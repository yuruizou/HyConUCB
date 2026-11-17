"""Rebuild the final xRouteBench features from published question/model records."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.linalg import helmert
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parent
DATASET = "ulab-ai/xRouteBench"
DATASET_REVISION = "ea4b6e1b29d9a734f55f0a637baf326bad6aa681"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TASK_NAMES = ("arc_challenge", "gsm8k", "math", "mmlu")
MODEL_NAMES = ("Qwen3-Coder-Next", "Llama-4-Maverick",
               "Mistral-Small-3-24B-Instruct", "Llama-3.3-70B-Instruct-Turbo")
MODEL_IDS = ("qwen3-coder-next", "llama-4-maverick",
             "mistral-small-3-24b-instruct", "llama-3.3-70b-instruct-turbo")
SPLIT_NAMESPACE = "xroute-combined-resplit-v3-query-group"
DEFAULT_DATA = ROOT / "data" / "benchmark.npz"


def read_records(directory: Path) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    records = []
    for source in ("train", "test"):
        path = directory / "llmrouter_generic" / f"{source}.parquet"
        table = pq.read_table(
            path, columns=["task_name", "embedding_id", "model_name", "query", "performance"],
            filters=[("task_name", "in", list(TASK_NAMES)), ("model_name", "in", list(MODEL_IDS))],
        )
        groups = {}
        for row in table.to_pylist():
            task, number = str(row["task_name"]), int(row["embedding_id"])
            group = groups.setdefault((task, number), {
                "task": task, "source": f"official_{source}", "number": number,
                "query": str(row["query"]), "cells": {},
            })
            model = str(row["model_name"])
            value = float(row["performance"])
            group["cells"][model] = int(value)
        for group in groups.values():
            group["rewards"] = [group["cells"][model] for model in MODEL_IDS]
            del group["cells"]
            group["id"] = f"{group['source']}:{group['task']}:{group['number']}"
            group["query_hash"] = hashlib.sha256(group["query"].encode("utf-8")).hexdigest()
            records.append(group)
    return split_records(records)


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    by_task = {task: {} for task in TASK_NAMES}
    for row in records:
        by_task[row["task"]].setdefault(row["query_hash"], []).append(row)
    assignments = {}
    for task, groups in by_task.items():
        ordered = sorted(groups, key=lambda digest: (
            hashlib.sha256(f"{SPLIT_NAMESPACE}|{task}|{digest}".encode("ascii")).hexdigest(), digest,
        ))
        n_train, n_val = int(np.floor(.6 * len(ordered) + .5)), int(np.floor(.2 * len(ordered) + .5))
        for i, digest in enumerate(ordered):
            assignments[task, digest] = "construction" if i < n_train else (
                "validation" if i < n_train + n_val else "final")
    hash_tasks = {}
    for task, digest in assignments:
        hash_tasks.setdefault(digest, []).append(task)
    for digest, tasks in hash_tasks.items():
        anchor = min(tasks, key=lambda task: (-len(by_task[task]), task))
        selected = assignments[anchor, digest]
        for task in tasks:
            assignments[task, digest] = selected
    splits = {name: [] for name in ("construction", "validation", "final")}
    for row in records:
        splits[assignments[row["task"], row["query_hash"]]].append(row)
    for rows in splits.values():
        rows.sort(key=lambda row: (TASK_NAMES.index(row["task"]), row["source"], row["number"]))
    return splits


def normalized(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    return values / norms[:, None]


def fit_classifier(embeddings: np.ndarray, tasks: np.ndarray) -> tuple:
    from sklearn.linear_model import LogisticRegression

    rows = normalized(embeddings)
    labels = np.asarray([TASK_NAMES.index(str(task)) for task in tasks])
    classifier = LogisticRegression(C=30.0, solver="lbfgs", max_iter=2000,
                                    tol=1e-10, fit_intercept=True, random_state=20261012).fit(rows, labels)
    coefficients, intercepts = np.asarray(classifier.coef_), np.asarray(classifier.intercept_)
    probabilities = softmax(normalized(rows) @ coefficients.T + intercepts, axis=1)
    mean = probabilities.mean(axis=0)
    scale = np.sqrt(3.0 / np.mean(np.sum((probabilities - mean) ** 2, axis=1)))
    return coefficients, intercepts, mean, float(scale)


def feature_map(embeddings: np.ndarray, coefficients: np.ndarray,
                intercepts: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    posterior = softmax(normalized(embeddings) @ coefficients.T + intercepts, axis=1)
    centered = posterior - mean
    centered[:, -1] = -np.sum(centered[:, :-1], axis=1)
    private = np.column_stack((np.ones(len(posterior)), scale * centered))
    private /= np.linalg.norm(private, axis=1)[:, None] * np.sqrt(2.0)
    arms = np.zeros((len(private), 4, 21), dtype=np.float64)
    for model in range(4):
        arms[:, model, model * 5:(model + 1) * 5] = private
    arms[:, :, -1] = 1.0 / np.sqrt(2.0)
    return arms


def fit_environment(arms: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    X, y = arms.reshape(-1, 21), rewards.reshape(-1).astype(np.float64)
    basis = np.zeros((21, 17), dtype=np.float64)
    for model in range(4):
        basis[model * 5, model * 4] = 1.0
        basis[model * 5 + 1:model * 5 + 5, model * 4 + 1:model * 4 + 4] = helmert(4, full=False).T
    basis[-1, -1] = 1.0
    Z, beta = X @ basis, np.zeros(17)
    for _ in range(100):
        eta = Z @ beta
        probability = expit(eta)
        gradient = Z.T @ (probability - y)
        if np.linalg.norm(gradient) <= 1e-10:
            theta = basis @ beta
            return theta
        hessian = Z.T @ ((probability * (1.0 - probability))[:, None] * Z)
        step = np.linalg.solve(hessian, gradient)
        objective = np.sum(np.logaddexp(0.0, eta) - y * eta)
        rate = 1.0
        while True:
            candidate = Z @ (beta - rate * step)
            value = np.sum(np.logaddexp(0.0, candidate) - y * candidate)
            if value <= objective - 1e-4 * rate * float(gradient @ step):
                break
            rate *= .5
            if rate < 2.0 ** -40:
                raise RuntimeError("Environment MLE line search failed")
        beta -= rate * step
    raise RuntimeError("Environment MLE did not converge")


def metadata(arrays: dict) -> dict:
    return {
        "dataset": DATASET, "configuration": "llmrouter_generic", "dataset_revision": DATASET_REVISION,
        "models": list(MODEL_NAMES), "model_ids": list(MODEL_IDS), "tasks": list(TASK_NAMES),
        "expert_model_indices": [0, 1, 2, 3], "dimension": 21,
        "split": {"fractions": [.6, .2, .2], "unit": "task and exact query text",
                  "construction_records": 2279, "validation_records": 761, "final_records": 760,
                  "official_train_and_test_combined": True},
        "embedding": {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
                      "text_template": "Query: {query}", "max_sequence_length": 512,
                      "normalized": True},
        "features": {"classifier": "construction-only multinomial logistic ridge", "classifier_C": 30.0,
                     "posterior": "four probabilities, construction-mean centered, one group scale to mean squared energy 3",
                     "model_block": "[1, scaled centered posterior] normalized to 1/sqrt(2)",
                     "global_intercept": float(1 / np.sqrt(2)), "unit_arm_norm": True},
        "theta_fit": {"scope": "construction-only binary correctness", "method": "unregularized minimum-norm logistic MLE",
                      "norm": float(np.linalg.norm(arrays["theta_environment"])), "online_bound_S": 6.0},
        "arm_feedback": "actual logged binary correctness",
        "conversation_feedback": "logistic-Bernoulli draws from theta_environment",
        "reference_versions": {"numpy": "2.4.6", "scipy": "1.17.1", "scikit-learn": "1.9.1"},
        "preparation_versions": {name: version(name) for name in (
            "numpy", "scipy", "scikit-learn", "pyarrow", "huggingface-hub",
            "sentence-transformers", "transformers", "tokenizers", "safetensors", "torch")},
    }


def build(output: Path, records_dir: Path | None = None, model_dir: Path | None = None,
          device: str = "cpu", batch_size: int = 64) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    from sentence_transformers import SentenceTransformer

    if records_dir is None:
        records_dir = ROOT / "data" / "raw"
        for source in ("train", "test"):
            hf_hub_download(DATASET, f"llmrouter_generic/{source}.parquet", repo_type="dataset",
                            revision=DATASET_REVISION, local_dir=records_dir)
    splits = read_records(Path(records_dir))
    train, final = splits["construction"], splits["final"]
    if model_dir is None:
        model_dir = Path(snapshot_download(EMBEDDING_MODEL, revision=EMBEDDING_REVISION))
    encoder = SentenceTransformer(str(model_dir), device=device, local_files_only=True)
    encoder.max_seq_length = 512
    texts = [f"Query: {row['query']}" for row in train + final]
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True)
    embeddings = normalized(embeddings).astype(np.float32)
    train_embeddings, final_embeddings = embeddings[:len(train)], embeddings[len(train):]
    train_rewards = np.asarray([row["rewards"] for row in train], dtype=np.uint8)
    transform = fit_classifier(train_embeddings, np.asarray([row["task"] for row in train]))
    train_arms, final_arms = feature_map(train_embeddings, *transform), feature_map(final_embeddings, *transform)
    theta = fit_environment(train_arms, train_rewards)
    tasks = np.asarray([row["task"] for row in final])
    rewards = np.asarray([row["rewards"] for row in final], dtype=np.uint8)
    arrays = {"arm_features": final_arms, "rewards": rewards, "tasks": tasks,
              "record_ids": np.asarray([row["id"] for row in final]), "theta_environment": theta,
              "accuracy_matrix": np.vstack([rewards[tasks == task].mean(axis=0) for task in TASK_NAMES])}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    description = metadata(arrays)
    description["embedding"].update(device=device, batch_size=batch_size)
    output.with_name("metadata.json").write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(final)} questions, d=21, theta norm={np.linalg.norm(theta):.9f}: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--records-dir", type=Path, help="directory containing llmrouter_generic/{train,test}.parquet")
    parser.add_argument("--model-dir", type=Path, help="local BGE snapshot; otherwise download the pinned revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args()
    build(arguments.output, arguments.records_dir, arguments.model_dir, arguments.device, arguments.batch_size)
