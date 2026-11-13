"""Profiling workload of tools/build_worker.sh (PGO=1): N envs of the training environment (v76 defaults) play
uniformly random legal actions from the bedroom for a while; the instrumented workers write their profiles when the
envs close.

    FIREREDSIM_WORKER=build/worker/fireredsim-worker PYTHONPATH=rl:python python3 tools/pgo_workload.py --seconds 180
"""
import argparse
import dataclasses
import time

import numpy as np

from pokerl import train as TR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = dataclasses.replace(TR.TrainConfig(), num_envs=a.envs, collector="sync")
    TR.ensure_native()
    env = TR.make_envs(cfg, a.seed)[0]
    rng = np.random.default_rng(a.seed)
    _, mask = env.reset()
    steps, t0 = 0, time.time()
    try:
        while time.time() - t0 < a.seconds:
            mask = np.asarray(mask, dtype=bool)
            # one legal action per env, uniformly: the k-th set bit of its mask row
            k = (rng.random(a.envs) * mask.sum(1)).astype(np.int64)
            actions = np.argmax(np.cumsum(mask, axis=1) > k[:, None], axis=1)
            _, mask, _, _, _, _ = env.step(actions)
            steps += a.envs
    finally:
        env.close()
    print(f"pgo workload: {steps:,} env steps in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
