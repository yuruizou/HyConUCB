"""Synthetic data and interaction streams."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr


SEEDS = (2756048, 675510, 807110, 2165051, 9492253, 927, 218, 495, 515, 452)
CONDITIONS = {"S": (4, 6, 8), "B": (2, 4, 6)}


@dataclass
class ExperimentConfig:
    axis: str = "S"
    value: float = 4
    n_users: int = 100
    T: int = 5000
    r: int = 10
    n_arms: int = 5000
    n_keyterms: int = 500
    pool_size: int = 100
    max_terms_per_arm: int = 3
    feature_noise_std: float = 0.02
    dataset_seed: int = 6842193
    alpha: float = 1.5
    xi: float = 0.05
    delta: float = 0.05
    numerical_tol: float = 1e-6

    @property
    def d(self):
        return self.r + (self.axis == "B")

    @property
    def S(self):
        return float(self.value) if self.axis == "S" else float(np.sqrt(2 * (self.value**2 + 1)))

    @property
    def condition(self):
        return f"{self.axis}{self.value:g}"

    def to_dict(self):
        return {
            "d": self.d, "S": self.S, "delta": self.delta, "alpha": self.alpha,
            "numerical_tol": self.numerical_tol,
            "fisher_logmin_query": {"xi": self.xi, "certificate_tol": self.numerical_tol},
        }


def all_pairs(count):
    return np.column_stack(np.triu_indices(count, 1)).astype(int)


def span_basis(vectors, rtol=1e-10):
    _, singular_values, vh = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(singular_values > singular_values[0] * rtol))
    return vh[:rank].T


def barycentric_spanner(keys, tolerance=1e-8):
    basis = span_basis(keys)
    rank = basis.shape[1]
    coordinates = keys @ basis
    indices = np.array(qr(coordinates.T, pivoting=True, mode="economic")[2][:rank])
    for _ in range(10000):
        coefficients = np.linalg.solve(coordinates[indices].T, coordinates.T).T
        i, j = np.unravel_index(np.abs(coefficients).argmax(), coefficients.shape)
        largest = float(abs(coefficients[i, j]))
        if largest <= 1 + tolerance:
            return indices, largest
        indices[j] = i
    raise RuntimeError("Barycentric spanner did not converge")


@dataclass
class World:
    arms: np.ndarray
    weights: np.ndarray
    users: np.ndarray
    key_features: np.ndarray
    key_ids: np.ndarray
    spanner_indices: np.ndarray
    spanner_pairs: np.ndarray
    spanner_coefficient: float

def build_world(config):
    rng = np.random.default_rng(config.dataset_seed)
    pseudo = rng.uniform(-1, 1, (config.n_keyterms, config.r))
    weights = np.zeros((config.n_arms, config.n_keyterms))
    for arm in range(config.n_arms):
        count = int(rng.integers(1, config.max_terms_per_arm + 1))
        ids = rng.choice(config.n_keyterms, count, replace=False)
        weights[arm, ids] = 1.0 / count
    raw = weights @ pseudo + rng.normal(0, config.feature_noise_std, (config.n_arms, config.r))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    relative_arms = raw / norms
    directions = rng.standard_normal((config.n_users, config.r))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms
    if config.axis == "S":
        arms = relative_arms
        users = config.S * directions
    else:
        sqrt_two = np.sqrt(2.0)
        arms = np.empty((config.n_arms, config.d))
        arms[:, 0] = 1.0 / sqrt_two
        arms[:, 1:] = relative_arms / sqrt_two
        users = np.empty((config.n_users, config.d))
        users[:, 0] = sqrt_two * config.value
        users[:, 1:] = sqrt_two * directions
    mass = weights.sum(axis=0)
    keys = (weights.T @ arms) / mass[:, None]
    if config.axis == "B":
        keys[:, 0] = 1.0 / np.sqrt(2.0)
    indices, coefficient = barycentric_spanner(keys)
    pairs = indices[all_pairs(len(indices))]
    return World(*map(np.ascontiguousarray, (arms, weights, users, keys, np.arange(config.n_keyterms, dtype=int),
                                           indices, pairs)), float(coefficient))


def cumulative_budget(t):
    return 10 * (t // 50)


def make_stream(config, user, repeat):
    seed = np.random.SeedSequence([SEEDS[repeat], user, repeat])
    pool_seed, arm_seed, query_seed = seed.spawn(3)
    pool_rng = np.random.default_rng(pool_seed)
    pools = np.array([pool_rng.choice(config.n_arms, config.pool_size, replace=False)
                      for _ in range(config.T)])
    arm_uniforms = np.random.default_rng(arm_seed).random(config.T)
    query_uniforms = np.random.default_rng(query_seed).random(cumulative_budget(config.T))
    return pools, arm_uniforms, query_uniforms


def policy_rngs(config, user, repeat):
    return tuple(np.random.default_rng(np.random.SeedSequence([SEEDS[repeat], user, repeat, stream]))
                 for stream in (91, 92))


