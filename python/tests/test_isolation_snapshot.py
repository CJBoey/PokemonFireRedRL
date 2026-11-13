"""Env isolation (envs swapped inside one worker == envs alone) and snapshot / restore / clone."""

import numpy as np
import pytest

import fireredsim as pn
from fireredsim.vec_env import SimulatorError


def test_interleaved_equals_solo(worker_path):
    rng = np.random.default_rng(5)
    acts = rng.integers(0, 9, size=(30, 3))
    seeds = [101, 202, 303]
    # three envs in ONE worker process: every step swaps the state of all three
    with pn.PokeVecEnv(num_envs=3, num_workers=1, seed=0, worker_path=worker_path) as env:
        env.reset(seed=seeds)
        inter = []
        for a in acts:
            env.step(a)
            inter.append(env.state_hash())
        stats = env.worker_stats()
        assert stats["swap_ns"].sum() > 0  # really swapped
    inter = np.array(inter)
    # each env alone in its own process
    for i in range(3):
        with pn.PokeVecEnv(num_envs=1, num_workers=1, seed=0, worker_path=worker_path) as env:
            env.reset(seed=[seeds[i]])
            solo = []
            for a in acts:
                env.step(a[i:i + 1])
                solo.append(env.state_hash()[0])
        assert (np.array(solo) == inter[:, i]).all(), f"env {i}: interleaved != solo"


def test_snapshot_restore_continuation(worker_path, tmp_path):
    rng = np.random.default_rng(8)
    acts = rng.integers(0, 9, size=(25, 4))
    with pn.PokeVecEnv(num_envs=4, num_workers=2, seed=11, worker_path=worker_path) as env:
        env.reset()
        for a in acts[:5]:
            env.step(a)
        snaps = env.snapshot()
        fast = env.snapshot(fast=True)
        h0 = env.state_hash()
        first = []
        for a in acts[5:]:
            env.step(a)
            first.append(env.state_hash())
        # restore (verified) and continue
        env.restore(snaps)
        assert (env.state_hash() == h0).all()
        for k, a in enumerate(acts[5:]):
            env.step(a)
            assert (env.state_hash() == first[k]).all()
        # fast snapshots restore without verification
        env.restore(fast)
        assert (env.state_hash() == h0).all()
        # file round trip, into another env (other worker)
        p = tmp_path / "s.pnsnap"
        env.restore(snaps)
        env.save_snapshot(0, p)
        env.load_snapshot(3, p)
        h = env.state_hash()
        assert h[3] == h0[0]
        # the snapshot file is the Simulator's format v1
        blob = open(p, "rb").read()
        assert blob[:8] == b"PNSNAP\r\n" and len(blob) == env.snapshot_size


def test_clone_across_workers(worker_path):
    with pn.PokeVecEnv(num_envs=4, num_workers=4, seed=21, worker_path=worker_path) as env:
        env.reset()
        for _ in range(10):
            env.step(np.array([5, 6, 7, 8]))
        env.clone([0, 0], [2, 3])
        h = env.state_hash()
        assert h[2] == h[0] and h[3] == h[0] and h[1] != h[0]
        # clones continue identically
        for _ in range(10):
            env.step(np.array([1, 2, 1, 1]))
        h = env.state_hash()
        assert h[0] == h[2] == h[3]


def test_bad_snapshots_are_rejected(worker_path):
    with pn.PokeVecEnv(num_envs=1, seed=1, worker_path=worker_path) as env:
        env.reset()
        good = env.snapshot()[0]
        bad = bytearray(good)
        bad[200000] ^= 0xFF
        with pytest.raises(SimulatorError, match="hash"):
            env.restore([bytes(bad)])
        foreign = bytearray(good)
        foreign[16] ^= 1  # build id
        with pytest.raises(SimulatorError, match="build id"):
            env.restore([bytes(foreign)])
        with pytest.raises(SimulatorError):
            env.restore([good[:-10]])
        # the env is unchanged and still works
        env.restore([good])
        env.step(np.array([0]))


def test_start_from_snapshot_file(worker_path, tmp_path):
    with pn.PokeVecEnv(num_envs=1, seed=5, worker_path=worker_path) as env:
        env.reset()
        for _ in range(3):
            env.step(np.array([5]))
        p = tmp_path / "start.pnsnap"
        env.save_snapshot(0, p)
        h = env.state_hash()[0]
        f = env.frames()[0]
    with pn.PokeVecEnv(num_envs=3, num_workers=2, start=str(p), seed=1, worker_path=worker_path) as env:
        _, info = env.reset()
        assert (env.state_hash() == h).all()
        assert (info["frame"] == f).all()
    # noop_reset_max gives seed-dependent starts
    with pn.PokeVecEnv(num_envs=4, num_workers=2, start=str(p), seed=1, noop_reset_max=30,
                       worker_path=worker_path) as env:
        _, info = env.reset()
        assert len(set(info["frame"].tolist())) > 1
