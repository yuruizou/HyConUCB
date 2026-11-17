"""Run the Yelp experiment sequentially and save each trajectory."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from algorithms import ALGORITHMS, DISPLAY_NAMES, make_policy
from data import MASTER_SEED, SEEDS, load_world

CONFIG = dict(users=100, horizon=5000, dimension=10, key_terms=500,
              candidate_pool=100, S=2.0, alpha=1.5, xi=0.05,
              budget_count=10, budget_period=50)
ROUND_ARRAYS = ("utility_regret", "link_regret", "chosen_arm", "arm_reward",
                "queries_issued", "optimal_hit", "online_theta_norm")


def feedback_uniform(seed, kind, *coordinates):
    mask = (1 << 64) - 1

    def splitmix(value):
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    value = splitmix(MASTER_SEED ^ int(seed) ^ (int(kind) << 48))
    for coordinate in coordinates:
        value = splitmix(value ^ (int(coordinate) & mask))
    return ((value >> 11) + .5) * 2.0**-53


def candidate_pools(seed, user, n_arms):
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, seed, user, 101]))
    pools = np.empty((CONFIG["horizon"], CONFIG["candidate_pool"]), dtype=np.uint16)
    for t in range(CONFIG["horizon"]):
        pools[t] = rng.choice(n_arms, CONFIG["candidate_pool"], replace=False)
    return pools


def run_one(world, seed, user, algorithm, stop=5000):
    with threadpool_limits(limits=1):
        repeat_seed = SEEDS[seed]
        pools = candidate_pools(repeat_seed, user, len(world.X))
        learner, selector = make_policy(algorithm, d=10, S=2, alpha=1.5, xi=.05)
        arrays = {name: np.zeros(stop, dtype=np.float64) for name in ROUND_ARRAYS}
        for name, dtype in (("chosen_arm", np.int32), ("arm_reward", np.int8),
                            ("queries_issued", np.int16), ("optimal_hit", np.int8)):
            arrays[name] = np.zeros(stop, dtype=dtype)
        arm_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, repeat_seed, user, 211]))
        query_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, repeat_seed, user, 223]))
        query_round, query_choice, query_reward = [], [], []
        keys = world.key_features
        pairs = world.spanner_key_pairs if algorithm == "conduel" else world.all_key_pairs
        for local in range(stop):
            t = local + 1
            pool = pools[local].astype(np.int64)
            X = world.X[pool]
            due = 10 if algorithm != "ofuglb_ucb" and t % 50 == 0 else 0
            for slot in range(due):
                if algorithm == "var_am":
                    left = int(selector.choose(learner, keys, query_rng))
                    right, feature = -1, keys[left]
                    logit = float(world.key_utilities[user, left])
                    uniform = feedback_uniform(repeat_seed, 3, user, local, slot, left)
                else:
                    selected = selector.choose(learner, keys, pairs, query_rng)
                    left, right = map(int, pairs[selected])
                    feature = keys[left] - keys[right]
                    logit = float(world.key_utilities[user, left] - world.key_utilities[user, right])
                    uniform = feedback_uniform(repeat_seed, 2, user, local, slot, min(left, right), max(left, right))
                    if left > right:
                        uniform = 1.0 - uniform
                response = int(uniform < expit(logit))
                learner.add(feature, response)
                query_round.append(t)
                query_choice.append((left, right))
                query_reward.append(response)
            chosen = int(learner.choose_arm(X, arm_rng))
            arm = int(pool[chosen])
            utilities = world.arm_utilities[user, pool]
            best, value = float(utilities.max()), float(utilities[chosen])
            response = int(feedback_uniform(repeat_seed, 1, user, local, arm) < expit(value))
            learner.add(world.X[arm], response)
            learner.fit()
            arrays["online_theta_norm"][local] = np.linalg.norm(learner.theta)
            arrays["link_regret"][local] = max(0., float(expit(best) - expit(value)))
            arrays["utility_regret"][local] = max(0., best - value)
            arrays["chosen_arm"][local], arrays["arm_reward"][local] = arm, response
            arrays["queries_issued"][local], arrays["optimal_hit"][local] = due, np.isclose(value, best, rtol=0., atol=1e-12)
    metadata = dict(algorithm=algorithm, user=user, seed_index=seed, repeat_seed=repeat_seed,
                    t=stop, config=CONFIG)
    return dict(arrays, user_parameter=world.theta_star[user],
                query_round=np.asarray(query_round, dtype=np.int32),
                query_choice=np.asarray(query_choice, dtype=np.int16).reshape(-1, 2),
                query_reward=np.asarray(query_reward, dtype=np.int8),
                metadata=np.array(json.dumps(metadata, sort_keys=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yelp.npz"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--users", type=int, nargs="+", default=list(range(100)), help="User indices, 0..99")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Seed indices, 0..9")
    parser.add_argument("--algorithms", nargs="+", choices=[DISPLAY_NAMES[a] for a in ALGORITHMS], default=[DISPLAY_NAMES[a] for a in ALGORITHMS])
    parser.add_argument("--stop", type=int, default=5000, help="Run a prefix of 1..5000 rounds from scratch")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        world = load_world(args.data)
        for seed in sorted(set(args.seeds)):
            for user in sorted(set(args.users)):
                folder = args.output / f"seed_{SEEDS[seed]}" / f"user_{user:03d}"
                folder.mkdir(parents=True, exist_ok=True)
                for algorithm in ALGORITHMS:
                    if DISPLAY_NAMES[algorithm] in args.algorithms:
                        result = run_one(world, seed, user, algorithm, args.stop)
                        np.savez_compressed(folder / f"{algorithm}.npz", **result)
                        print(f"seed={seed} user={user} {DISPLAY_NAMES[algorithm]}: "
                              f"regret={result['link_regret'].sum():.6f}", flush=True)


if __name__ == "__main__":
    main()
