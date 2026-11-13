"""Determinism: same seeds + actions => identical state hashes and observations,
across processes, worker counts and frameskip decompositions (spec.md section 8)."""
import numpy as np
import pytest

import fireredsim as pn


def run(worker_path, num_envs, num_workers, actions, seed=7, frameskip=24, **kw):
    hashes, obs = [], []
    with pn.PokeVecEnv(num_envs=num_envs, num_workers=num_workers, frameskip=frameskip, seed=seed,
                       worker_path=worker_path, **kw) as env:
        o, _ = env.reset()
        obs.append(o.copy())
        hashes.append(env.state_hash())
        for a in actions:
            o, *_ = env.step(a)
            obs.append(o.copy())
            hashes.append(env.state_hash())
    return np.array(hashes), obs


@pytest.fixture(scope="module")
def actions():
    rng = np.random.default_rng(123)
    return [rng.integers(0, 9, size=6) for _ in range(40)]


def test_same_seed_same_trajectory_across_processes(worker_path, actions):
    h1, o1 = run(worker_path, 6, 6, actions)
    h2, o2 = run(worker_path, 6, 6, actions)
    assert (h1 == h2).all()
    for a, b in zip(o1, o2):
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_worker_count_does_not_matter(worker_path, actions, workers):
    ref, ref_obs = run(worker_path, 6, 6, actions)
    h, o = run(worker_path, 6, workers, actions)
    assert (h == ref).all(), f"{workers} workers differ from 6"
    for a, b in zip(o, ref_obs):
        assert a.tobytes() == b.tobytes()


def test_seeds_differ(worker_path):
    h1, o1 = run(worker_path, 2, 2, [], seed=1)
    h2, o2 = run(worker_path, 2, 2, [], seed=2)
    assert (h1[0] != h2[0]).all()
    # different envs of one run have different episode seeds -> different trainer IDs / RNG
    assert h1[0][0] != h1[0][1]
    if "trainer_id_lo" in o1[0].dtype.names:
        assert o1[0]["trainer_id_lo"][0] != o1[0]["trainer_id_lo"][1]


def test_explicit_timer1_values(worker_path):
    with pn.PokeVecEnv(num_envs=2, worker_path=worker_path) as env:
        o, _ = env.reset(options={"timer1": [0x7C2A, 0x61C0]})
        h = env.state_hash()
        assert h[0] == h[1]
        if "trainer_id_lo" in o.dtype.names:
            assert int(o["trainer_id_lo"][0]) == 0x61C0
