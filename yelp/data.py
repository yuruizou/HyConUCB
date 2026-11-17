"""Prepare and load the fixed Yelp semi-synthetic experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy import sparse
from scipy.linalg import qr

MASTER_SEED = 6_842_193
SEEDS = (381274, 1965181828, 3886694563, 3669556475, 1726213960,
         1331522399, 2196341159, 1336312332, 2224772794, 1653692238)
S, DIMENSION = 2.0, 10
N_USERS, N_ARMS, N_KEYS = 100, 20_572, 500
USER_FILE = Path(__file__).with_name("selected_users.txt")
BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"


def all_pairs(count: int) -> np.ndarray:
    return np.column_stack(np.triu_indices(count, 1)).astype(int)


def barycentric_spanner(keys: np.ndarray, tolerance: float = 1e-8):
    _, singular_values, vh = np.linalg.svd(keys, full_matrices=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-10))
    coordinates = keys @ vh[:rank].T
    indices = np.array(qr(coordinates.T, pivoting=True, mode="economic")[2][:rank])
    for _ in range(10_000):
        coefficients = np.linalg.solve(coordinates[indices].T, coordinates.T).T
        i, j = np.unravel_index(np.abs(coefficients).argmax(), coefficients.shape)
        largest = float(abs(coefficients[i, j]))
        if largest <= 1 + tolerance:
            return indices, largest
        indices[j] = i
    raise RuntimeError("Barycentric spanner did not converge")


@dataclass
class World:
    X: np.ndarray
    theta_star: np.ndarray
    arm_utilities: np.ndarray
    key_features: np.ndarray
    key_utilities: np.ndarray
    business_ids: np.ndarray
    user_ids: np.ndarray
    key_names: np.ndarray
    all_key_pairs: np.ndarray
    spanner_indices: np.ndarray
    spanner_key_pairs: np.ndarray
    spanner_coefficient: float


def selected_users() -> np.ndarray:
    return np.asarray(USER_FILE.read_text(encoding="utf-8").splitlines(), dtype=str)


def load_world(path: Path) -> World:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in (
            "X", "theta_star", "arm_utilities", "key_features", "key_utilities",
            "business_ids", "user_ids", "key_names")}
    indices, certificate = barycentric_spanner(arrays["key_features"])
    pairs = all_pairs(N_KEYS)
    spanner_pairs = indices[all_pairs(len(indices))]
    return World(**arrays, all_key_pairs=pairs, spanner_indices=indices,
                 spanner_key_pairs=spanner_pairs, spanner_coefficient=certificate)


def _attribute_scalar(value) -> float:
    if value is None:
        return 0.
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(np.clip(value, 0., 1.))
    text = str(value).strip()
    lower = text.lower()
    if lower in {"true", "yes", "1", "free", "full_bar", "beer_and_wine"}:
        return 1.
    if lower in {"false", "no", "0", "none", "null", "no alcohol"}:
        return 0.
    if text.startswith("{") and text.endswith("}"):
        normalized = text.replace("'", '"').replace("None", "null")
        normalized = normalized.replace("True", "true").replace("False", "false")
        try:
            nested = json.loads(normalized)
        except (ValueError, TypeError):
            nested = None
        if isinstance(nested, dict) and nested:
            return float(np.mean([_attribute_scalar(v) for v in nested.values()]))
    return 1.


def load_businesses(path: Path) -> list[dict]:
    businesses = []
    with Path(path).open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            categories = tuple(sorted({x.strip() for x in (row.get("categories") or "").split(",")
                                       if x.strip()}))
            if "Restaurants" not in categories or int(row.get("review_count") or 0) < 50:
                continue
            businesses.append({"id": str(row["business_id"]),
                               "city": str(row.get("city") or "<missing>"),
                               "categories": categories,
                               "stars": float(row.get("stars") or 0.),
                               "reviews": int(row["review_count"]),
                               "attributes": dict(row.get("attributes") or {})})
    businesses.sort(key=lambda row: row["id"])
    return businesses


def build_features(businesses: list[dict]):
    import sklearn
    from sklearn.decomposition import PCA

    cities = sorted({b["city"] for b in businesses})
    categories = sorted({c for b in businesses for c in b["categories"]})
    counts = Counter(key for b in businesses for key in b["attributes"])
    attributes = sorted(counts, key=lambda key: (-counts[key], key))[:34]
    city_index = {name: i for i, name in enumerate(cities)}
    category_index = {name: i for i, name in enumerate(categories)}
    attribute_index = {name: i for i, name in enumerate(attributes)}
    raw = np.zeros((len(businesses), len(cities) + len(categories) + len(attributes) + 2),
                   dtype=np.float32)
    max_log_reviews = max(math.log1p(b["reviews"]) for b in businesses)
    for row, business in enumerate(businesses):
        raw[row, city_index[business["city"]]] = 1.
        for category in business["categories"]:
            raw[row, len(cities) + category_index[category]] = 1.
        for key, value in business["attributes"].items():
            index = attribute_index.get(key)
            if index is not None:
                raw[row, len(cities) + len(categories) + index] = _attribute_scalar(value)
        raw[row, -2] = business["stars"] / 5.
        raw[row, -1] = math.log1p(business["reviews"]) / max_log_reviews
    pca = PCA(n_components=DIMENSION, svd_solver="randomized",
              random_state=MASTER_SEED, iterated_power=7)
    X = np.asarray(pca.fit_transform(raw), dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    X /= norms[:, None]
    coverage = Counter(c for b in businesses for c in b["categories"])
    names = sorted(coverage, key=lambda key: (-coverage[key], key))[:N_KEYS]
    locations = {name: i for i, name in enumerate(names)}
    rows, columns, weights = [], [], []
    for arm, business in enumerate(businesses):
        for category in business["categories"]:
            column = locations.get(category)
            if column is not None:
                rows.append(arm)
                columns.append(column)
                weights.append(1. / len(business["categories"]))
    W = sparse.csr_matrix((weights, (rows, columns)), shape=(len(X), N_KEYS), dtype=np.float64)
    mass = np.asarray(W.sum(axis=0)).ravel()
    keys = np.asarray(W.T @ X) / mass[:, None]
    metadata = {"raw_dimension": raw.shape[1], "attribute_names": attributes,
                "scikit_learn_version": sklearn.__version__, "numpy_version": np.__version__,
                "pca_solver": "randomized", "pca_seed": MASTER_SEED, "pca_iterated_power": 7,
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                "arm_normalization": "unit Euclidean norm after PCA",
                "key_weight": "1 / number of all categories of the associated arm",
                "key_feature": "column-normalized weighted average of global arm features"}
    return X, keys, np.asarray(names, dtype=str), mass, metadata


def collect_reviews(path: Path, users: np.ndarray, business_ids: np.ndarray):
    positions = {user: i for i, user in enumerate(map(str, users))}
    arms = {business: i for i, business in enumerate(map(str, business_ids))}
    selected = [([], []) for _ in users]
    with Path(path).open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            user = positions.get(str(row["user_id"]))
            if user is None:
                continue
            arm = arms.get(str(row["business_id"]))
            if arm is not None:
                selected[user][0].append(arm)
                selected[user][1].append(int(float(row["stars"]) >= 4.))
    reviews = []
    for indices, labels in selected:
        reviews.append((np.asarray(indices, dtype=int), np.asarray(labels, dtype=float)))
    return reviews


def fit_user(X: np.ndarray, y: np.ndarray):
    A, b = X.T @ X, X.T @ y
    A = .5 * (A + A.T)
    eigenvalues, vectors = np.linalg.eigh(A)
    scale = max(1., float(np.max(np.abs(eigenvalues))))
    floor = 128 * np.finfo(float).eps * scale
    if eigenvalues.min() < -floor:
        raise RuntimeError("A review Gram matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.)
    coordinates = vectors.T @ b
    positive = eigenvalues > floor
    candidate = np.zeros(DIMENSION)
    candidate[positive] = coordinates[positive] / eigenvalues[positive]
    theta = vectors @ candidate
    null_residual = np.linalg.norm(coordinates[~positive])
    if null_residual > 256 * np.finfo(float).eps * max(1., np.linalg.norm(b)) or np.linalg.norm(theta) > S:
        low, high = 0., max(1., np.linalg.norm(b) / S)
        while np.linalg.norm(coordinates / (eigenvalues + high)) > S:
            high *= 2.
        for _ in range(100):
            middle = .5 * (low + high)
            if np.linalg.norm(coordinates / (eigenvalues + middle)) > S:
                low = middle
            else:
                high = middle
        theta = vectors @ (coordinates / (eigenvalues + high))
    norm = np.linalg.norm(theta)
    if norm > S:
        theta *= (S / norm) * (1 - 2 * np.finfo(float).eps)
    gradient = A @ theta - b
    gap = float(gradient @ theta + S * np.linalg.norm(gradient))
    if np.linalg.norm(theta) > S * (1 + 1e-11) or gap > 1e-8:
        raise RuntimeError("User regression failed its ball KKT certificate")
    return theta, max(0., gap)


def prepare(raw_dir: Path, output: Path) -> None:
    raw_dir, output = Path(raw_dir), Path(output)
    business_path, review_path = raw_dir / BUSINESS_FILE, raw_dir / REVIEW_FILE
    users = selected_users()
    businesses = load_businesses(business_path)
    business_ids = np.asarray([b["id"] for b in businesses], dtype=str)
    X, keys, key_names, key_mass, feature_metadata = build_features(businesses)
    print(f"Prepared {len(X)} arms and {len(keys)} key terms; reading selected reviews", flush=True)
    reviews = collect_reviews(review_path, users, business_ids)
    fits = [fit_user(X[arms], labels) for arms, labels in reviews]
    theta = np.stack([row[0] for row in fits])
    arrays = dict(X=X, theta_star=theta, arm_utilities=theta @ X.T,
                  key_features=keys, key_utilities=theta @ keys.T,
                  business_ids=business_ids, user_ids=users, key_names=key_names,
                  key_mass=key_mass)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {"dimension": DIMENSION, "S": S, "rho": 0., "users": N_USERS,
                "arms": N_ARMS, "key_terms": N_KEYS, "binary_label": "1{stars >= 4}",
                "features": feature_metadata, "max_regression_kkt_gap": max(r[1] for r in fits),
                "review_counts": [len(labels) for _, labels in reviews]}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved cache: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="directory containing Yelp business/review JSON")
    parser.add_argument("--output", type=Path, default=Path("data/yelp.npz"))
    arguments = parser.parse_args()
    prepare(arguments.raw, arguments.output)
