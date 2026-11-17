"""Run the synthetic experiments sequentially and save each trajectory."""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from algorithms import AM, CDSPAN, FISHER, LEE, RANDOM, make_policy
from environment import CONDITIONS, ExperimentConfig, all_pairs, build_world, cumulative_budget, make_stream, policy_rngs


POLICIES = {"hyconucb": FISHER, "conduel": CDSPAN, "var-random": RANDOM,
            "var-am": AM, "ofuglb-ucb": LEE}
THETA_TIMING = "after_current_round_queries_before_arm_response"


def run_one(config, world, algorithm, user, seed, stop=None):
    stop = config.T if stop is None else stop
    with threadpool_limits(limits=1):
        pools, arm_uniforms, query_uniforms = make_stream(config, user, seed)
        arm_rng, query_rng = policy_rngs(config, user, seed)
        policy = POLICIES[algorithm]
        learner, design = make_policy(config.to_dict(), policy)
        arrays = {name: np.zeros(stop) for name in ("regret", "utility_regret", "parameter_error")}
        arrays.update(theta=np.zeros((stop, config.d)), actions=np.zeros(stop, np.int32),
                      reward=np.zeros(stop, np.int8))
        query_time, query_pair, query_reward = [], [], []
        keys, truth = world.key_features, world.users[user]
        pairs = None
        if policy == CDSPAN:
            pairs = world.spanner_pairs
        elif policy in (FISHER, RANDOM):
            pairs = all_pairs(len(keys))

        for index in range(stop):
            t = index + 1
            X = world.arms[pools[index]]
            count = 0 if policy == LEE else cumulative_budget(t) - cumulative_budget(t - 1)
            for _ in range(count):
                if policy == AM:
                    selected = design.choose(learner, keys, query_rng)
                    pair, vector = (int(selected), -1), keys[selected]
                else:
                    selected = design.choose(learner, keys, pairs, query_rng)
                    pair = tuple(map(int, pairs[selected]))
                    vector = keys[pair[0]] - keys[pair[1]]
                response = int(query_uniforms[len(query_time)] < expit(vector @ truth))
                query_time.append(t)
                query_pair.append(pair)
                query_reward.append(response)
                learner.add(vector, response)

            local = learner.choose_arm(X, arm_rng)
            arrays["theta"][index] = learner.last["theta"]
            arrays["parameter_error"][index] = np.linalg.norm(arrays["theta"][index] - truth)
            utilities = X @ truth
            means = expit(utilities)
            arrays["regret"][index] = max(0.0, float(means.max() - means[local]))
            arrays["utility_regret"][index] = max(0.0, float(utilities.max() - utilities[local]))
            arrays["actions"][index] = pools[index, local]
            arrays["reward"][index] = int(arm_uniforms[index] < means[local])
            learner.add(X[local], int(arrays["reward"][index]))

    metadata = dict(condition=config.condition, algorithm=algorithm, user=user, seed_index=seed,
                    T=config.T, t=stop, config=asdict(config), query_count=len(query_time),
                    theta_timing=THETA_TIMING)
    return dict(arrays, time=np.arange(1, stop + 1), user_parameter=truth,
                query_time=np.asarray(query_time, dtype=np.int32),
                query_pair=np.asarray(query_pair, dtype=np.int32).reshape(-1, 2),
                query_reward=np.asarray(query_reward, dtype=np.int8),
                metadata=np.array(json.dumps(metadata, sort_keys=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("S", "B", "both"), default="both")
    parser.add_argument("--values", type=int, nargs="+", help="Subset of values for one setting")
    parser.add_argument("--users", type=int, nargs="+", default=list(range(100)), help="User IDs, 0..99")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Seed indices, 0..9")
    parser.add_argument("--stop", type=int, default=5000, help="Run this many rounds from scratch, using the full 5000-round streams")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    axes = tuple(CONDITIONS) if args.setting == "both" else (args.setting,)
    with threadpool_limits(limits=1):
        for axis in axes:
            for value in sorted(set(args.values or CONDITIONS[axis])):
                config = ExperimentConfig(axis=axis, value=value)
                world = build_world(config)
                print(f"Running {config.condition}: {args.stop} rounds per trajectory", flush=True)
                for seed in sorted(set(args.seeds)):
                    for user in sorted(set(args.users)):
                        folder = args.output / config.condition / f"seed{seed:02d}" / f"user{user:03d}"
                        folder.mkdir(parents=True, exist_ok=True)
                        for algorithm in POLICIES:
                            result = run_one(config, world, algorithm, user, seed, args.stop)
                            np.savez_compressed(folder / f"{algorithm}.npz", **result)
                            print(f"{config.condition} seed={seed} user={user} {algorithm}: "
                                  f"regret={result['regret'].sum():.6f}", flush=True)


if __name__ == "__main__":
    main()
