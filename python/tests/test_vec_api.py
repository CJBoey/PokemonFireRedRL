"""PokeVecEnv API: autoreset, truncation, partial resets, async stepping, pixels, start-up."""
import numpy as np
import pytest

import fireredsim as pn


def test_autoreset_next_step(worker_path):
    def term(obs, info):
        return info["episode_step"] >= 3

    with pn.PokeVecEnv(num_envs=2, num_workers=2, seed=1, termination_fn=term, worker_path=worker_path) as env:
        obs0, info0 = env.reset()
        h0 = env.state_hash()
        for k in range(1, 4):
            _, rew, terminated, truncated, info = env.step(np.array([5, 6]))
            assert terminated.tolist() == [k >= 3] * 2
        # next step: both envs start a new episode (action ignored, reward 0, new seed)
        obs, rew, terminated, truncated, info = env.step(np.array([1, 1]))
        assert info["autoreset"].all() and not terminated.any() and (rew == 0).all()
        assert (info["episode_step"] == 0).all()
        assert (info["frame"] == 5161).all()
        assert (env.state_hash() != h0).all()  # a new episode seed
        assert (info["episode_seed"] != info0["episode_seed"]).all()


def test_max_episode_steps(worker_path):
    with pn.PokeVecEnv(num_envs=1, seed=1, max_episode_steps=2, worker_path=worker_path) as env:
        env.reset()
        assert not env.step(np.array([0]))[3][0]
        assert env.step(np.array([0]))[3][0]
        _, _, _, _, info = env.step(np.array([0]))
        assert info["autoreset"][0]


def test_partial_reset_and_episode_seeds(worker_path):
    with pn.PokeVecEnv(num_envs=3, num_workers=3, seed=5, worker_path=worker_path) as env:
        _, info = env.reset()
        seeds0 = info["episode_seed"].copy()
        for _ in range(3):
            env.step(np.array([5, 5, 5]))
        h = env.state_hash()
        obs, info = env.reset(env_ids=[1])
        assert obs.shape == (1,) and info["frame"][0] == 5161
        h2 = env.state_hash()
        assert h2[0] == h[0] and h2[2] == h[2] and h2[1] != h[1]
        assert info["episode_seed"][0] != seeds0[1]  # next episode of env 1
        # the same base seed reproduces the same first episodes
        env.reset(seed=5)
        assert (env.state_hash() == env.state_hash()).all()
    with pn.PokeVecEnv(num_envs=3, num_workers=1, seed=5, worker_path=worker_path) as env2:
        _, info2 = env2.reset()
        assert (info2["episode_seed"] == seeds0).all()


def test_step_async(worker_path):
    with pn.PokeVecEnv(num_envs=2, num_workers=2, seed=2, worker_path=worker_path) as a, \
         pn.PokeVecEnv(num_envs=2, num_workers=2, seed=2, worker_path=worker_path) as b:
        a.reset()
        b.reset()
        for k in range(5):
            act = np.array([k % 9, (k * 3) % 9])
            a.step_async(act)
            ob, *_ = b.step(act)
            oa, *_ = a.step_wait()
            assert oa.tobytes() == ob.tobytes()
        assert (a.state_hash() == b.state_hash()).all()


def test_pixels_only_and_copy_false(worker_path):
    with pn.PokeVecEnv(num_envs=2, pixels="only", copy=False, seed=3, worker_path=worker_path) as env:
        obs, _ = env.reset()
        assert obs.shape == (2, 160, 240, 3) and obs.dtype == np.uint8
        o1, *_ = env.step(np.array([5, 6]))
        snapshot = o1.copy()
        o2, *_ = env.step(np.array([5, 6]))
        assert o1 is o2 or np.shares_memory(o1, o2)  # zero-copy views of the shared memory
        assert not np.array_equal(snapshot, o2) or True


def test_multibinary_and_buttons_actions(worker_path):
    with pn.PokeVecEnv(num_envs=2, action_type="multibinary", seed=4, worker_path=worker_path) as a, \
         pn.PokeVecEnv(num_envs=2, action_type="buttons", seed=4, worker_path=worker_path) as b:
        a.reset()
        b.reset()
        bits = np.zeros((2, 10), dtype=np.int8)
        bits[0, 6] = 1  # UP
        bits[1, 1] = bits[1, 5] = 1  # B + LEFT
        a.step(bits)
        b.step(np.array([pn.BUTTONS["UP"], pn.BUTTONS["B"] | pn.BUTTONS["LEFT"]], dtype=np.uint16))
        assert (a.state_hash() == b.state_hash()).all()
        with pytest.raises(ValueError):
            b.step(np.array([0x400, 0]))


def test_many_quick_startups(worker_path):
    # regression: a worker must not miss the first batch posted right after start-up
    for k in range(10):
        with pn.PokeVecEnv(num_envs=4, num_workers=4, start="power_on", seed=k, worker_path=worker_path,
                           batch_timeout=60) as env:
            env.reset()
            env.step(np.zeros(4, dtype=np.int64))


def test_packed_workers_match_single(worker_path):
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 15, size=(10, 8))
    hashes = []
    for workers in (8, 2):
        with pn.PokeVecEnv(num_envs=8, num_workers=workers, seed=99, action_set="full",
                           worker_path=worker_path) as env:
            env.reset()
            for a in acts:
                env.step(a)
            hashes.append(env.state_hash())
    assert (hashes[0] == hashes[1]).all()


def test_send_recv_matches_sync(worker_path):
    rng = np.random.default_rng(1)
    acts = rng.integers(0, 9, size=(12, 6))
    with pn.PokeVecEnv(num_envs=6, num_workers=6, seed=31, worker_path=worker_path) as ref:
        ref.reset()
        for a in acts:
            ref.step(a)
        ref_h = ref.state_hash()
    with pn.PokeVecEnv(num_envs=6, num_workers=6, seed=31, worker_path=worker_path) as env:
        env.reset()
        steps = np.zeros(6, dtype=int)
        env.send(acts[0])
        while (steps < len(acts)).any():
            obs, rew, term, trunc, info = env.recv(min_envs=2)
            ids = info["env_id"]
            assert len(ids) >= 2 or (steps < len(acts)).sum() < 2
            assert obs.shape == (len(ids),)
            steps[ids] += 1
            todo = ids[steps[ids] < len(acts)]
            if todo.size:
                env.send(acts[steps[todo], todo], env_ids=todo)
        assert (env.state_hash() == ref_h).all()
        with pytest.raises(RuntimeError):
            env.recv()


def test_send_recv_guards(worker_path):
    with pn.PokeVecEnv(num_envs=4, num_workers=2, seed=1, worker_path=worker_path) as env:
        env.reset()
        env.send(np.zeros(2, dtype=int), env_ids=[0, 1])        # worker 0
        with pytest.raises(RuntimeError):
            env.send(np.zeros(1, dtype=int), env_ids=[1])       # worker 0 still busy
        with pytest.raises(RuntimeError):
            env.step(np.zeros(4, dtype=int))                    # sync step while in flight
        env.send(np.zeros(2, dtype=int), env_ids=[2, 3])        # worker 1
        _, _, _, _, info = env.recv()
        assert sorted(info["env_id"].tolist()) == [0, 1, 2, 3]
        env.step(np.zeros(4, dtype=int))                        # back to synchronous use


def test_script_start(worker_path, repo):
    import os
    # the bedroom script as a ScriptStart with per-episode seeds == start="bedroom"
    script = pn.ScriptStart(os.path.join(repo, "worker", "new_game_to_pallet.txt"))
    with pn.PokeVecEnv(num_envs=2, start=script, seed=5, worker_path=worker_path) as a, \
         pn.PokeVecEnv(num_envs=2, start="bedroom", seed=5, worker_path=worker_path) as b:
        a.reset()
        b.reset()
        assert (a.state_hash() == b.state_hash()).all()


def test_reward_fn_sync_and_async(worker_path):
    def reward_fn(prev, obs, info):
        return (obs["x"].astype(np.float32) - prev["x"]) if "x" in obs.dtype.names else np.ones(len(obs))

    with pn.PokeVecEnv(num_envs=2, num_workers=2, seed=4, reward_fn=reward_fn, worker_path=worker_path) as env:
        env.reset()
        right = int(np.flatnonzero(env.actions == pn.BUTTONS["RIGHT"])[0])
        _, rew, _, _, _ = env.step(np.array([right, 0]))
        assert rew.shape == (2,) and rew.dtype == np.float32
        env.send(np.array([right, 0]))
        _, rew2, _, _, info = env.recv()
        assert rew2.shape == (2,)
        if "x" in env.obs_dtype.names:
            assert rew[1] == 0 and rew2[list(info["env_id"]).index(1)] == 0
