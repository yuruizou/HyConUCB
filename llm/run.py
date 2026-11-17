"""Run the five-algorithm xRouteBench experiments sequentially and save each trajectory."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from algorithms import ALGORITHMS, ALL_PAIRS, make_policy
from prepare import TASK_NAMES

ROOT = Path(__file__).resolve().parent
MASTER_SEED = 20261011
SEEDS = tuple(range(20261201, 20261401))
HORIZON = 5000
SETTINGS = {"S": 6.0, "alpha": 1.5, "xi": 0.05, "delta": 0.05, "tol": 1e-6}
W = np.eye(4)


def load_benchmark(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    data["task_index"] = np.asarray([TASK_NAMES.index(str(task)) for task in data["tasks"]], dtype=np.int8)
    return data


def replay_order(tasks: np.ndarray, seed: int, horizon: int = HORIZON) -> np.ndarray:
    groups = [np.flatnonzero(tasks == task).astype(np.int32) for task in TASK_NAMES]
    counts = np.full(4, horizon // 4, dtype=np.int64)
    allocation_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, seed, 301]))
    if horizon % 4:
        counts[allocation_rng.permutation(4)[:horizon % 4]] += 1
    queues = []
    for task, (group, count) in enumerate(zip(groups, counts)):
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, seed, 307, task]))
        blocks, remaining = [], int(count)
        while remaining:
            permutation = rng.permutation(group).astype(np.int32, copy=False)
            take = min(remaining, len(group))
            blocks.append(permutation[:take])
            remaining -= take
        queues.append(np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int32))
    schedule = np.repeat(np.arange(4, dtype=np.int8), counts)
    np.random.default_rng(np.random.SeedSequence([MASTER_SEED, seed, 313])).shuffle(schedule)
    cursors, order = np.zeros(4, dtype=np.int64), np.empty(horizon, dtype=np.int32)
    for position, task in enumerate(schedule):
        order[position] = queues[task][cursors[task]]
        cursors[task] += 1
    return order


def conversation_feedback(keys: np.ndarray, theta: np.ndarray, kind: str,
                          index: int, seed: int, round_index: int) -> int:
    if kind == "pair":
        left, right = map(int, ALL_PAIRS[index])
        feature, coordinates = keys[left] - keys[right], (seed, 2, round_index, 0, left, right)
    else:
        feature, coordinates = keys[index], (seed, 3, round_index, 0, index)
    payload = ":".join(map(str, coordinates)).encode("ascii")
    uniform = (int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") + 0.5) / 2**64
    return int(uniform < float(expit(float(feature @ theta))))


def result_path(output: Path, algorithm: str, seed: int) -> Path:
    return output / f"seed_{seed}" / f"{algorithm}.npz"


def metric_curves(arrays: dict, data: dict) -> dict:
    order, chosen = arrays["record_order"], arrays["chosen_model"]
    tasks = data["task_index"][order]
    rewards = data["rewards"][order]
    selected = rewards[np.arange(len(order)), chosen].astype(np.int64)
    expert = rewards[np.arange(len(order)), tasks].astype(np.int64)
    rates = data["accuracy_matrix"][tasks]
    return {
        "expert_regret": np.r_[0, np.cumsum(expert - selected)],
        "logged_regret": np.r_[0, np.cumsum(rewards.max(axis=1).astype(np.int64) - selected)],
        "task_pseudo_regret": np.r_[0.0, np.cumsum(rates.max(axis=1) - rates[np.arange(len(order)), chosen])],
        "theta_error": np.r_[np.linalg.norm(data["theta_environment"]), arrays["theta_error"]],
    }


def run_one(data: dict, algorithm: str, seed: int, stop: int = HORIZON) -> dict:
    order = replay_order(data["tasks"], seed)[:stop]
    policy = make_policy(algorithm, d=21, **SETTINGS)
    arm_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, seed, 211]))
    query_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, seed, 223]))
    arrays = {
        "record_order": order, "chosen_model": np.zeros(stop, dtype=np.int8),
        "arm_reward": np.zeros(stop, dtype=np.int8),
        "query_index": np.full(stop, -1, dtype=np.int8),
        "query_response": np.full(stop, -1, dtype=np.int8),
        "theta_error": np.zeros(stop), "theta_norm": np.zeros(stop),
    }
    for round_index in range(stop):
        record = int(order[round_index])
        X = np.asarray(data["arm_features"][record], dtype=np.float64)
        keys = W @ X
        if algorithm != "OFUGLB-UCB" and (round_index + 1) % 5 == 0:
            feature, kind, index = policy.query(keys, query_rng)
            response = conversation_feedback(keys, data["theta_environment"], kind, index, seed, round_index)
            policy.observe_query(feature, response)
            arrays["query_index"][round_index], arrays["query_response"][round_index] = index, response
        chosen = int(policy.choose_arm(X, arm_rng))
        reward = int(data["rewards"][record, chosen])
        policy.observe_arm(X[chosen], reward)
        theta = policy.theta
        arrays["chosen_model"][round_index], arrays["arm_reward"][round_index] = chosen, reward
        arrays["theta_error"][round_index] = np.linalg.norm(theta - data["theta_environment"])
        arrays["theta_norm"][round_index] = np.linalg.norm(theta)
    theta = policy.theta
    curves = metric_curves(arrays, data)
    summary = {
        "algorithm": algorithm, "seed": seed, "horizon": stop, "stream_horizon": HORIZON,
        "expert_regret": float(curves["expert_regret"][-1]),
        "logged_regret": float(curves["logged_regret"][-1]),
        "task_pseudo_regret": float(curves["task_pseudo_regret"][-1]),
        "accuracy": float(arrays["arm_reward"].mean()),
        "theta_error": float(arrays["theta_error"][-1]),
        "queries": int(np.count_nonzero(arrays["query_index"] >= 0)),
    }
    return dict(arrays, theta_final=theta, summary=np.asarray(json.dumps(summary)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/benchmark.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--stop", "--horizon", dest="stop", type=int, default=HORIZON,
                        help="Run 1..5000 rounds from scratch using the full replay stream")
    args = parser.parse_args()
    data = load_benchmark(str(args.data))
    for seed in dict.fromkeys(args.seeds):
        for algorithm in dict.fromkeys(args.algorithms):
            result = run_one(data, algorithm, seed, args.stop)
            target = result_path(args.output, algorithm, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **result)
            summary = json.loads(str(result["summary"]))
            print(f"{algorithm} seed={seed}: expert={summary['expert_regret']:.0f}, "
                  f"logged={summary['logged_regret']:.0f}", flush=True)


if __name__ == "__main__":
    main()
