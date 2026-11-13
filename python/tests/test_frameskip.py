"""Frame skip semantics: step(a, n) == n single-frame steps with a held (ARCHITECTURE.md 4)."""
import numpy as np
import pytest

import fireredsim as pn


@pytest.mark.parametrize("n", [8, 24, 48])
def test_step_n_equals_n_single_frames(worker_path, n):
    rng = np.random.default_rng(n)
    buttons = [int(b) for b in rng.choice(pn.action_table("basic"), size=12)]
    with pn.PokeVecEnv(num_envs=2, num_workers=2, frameskip=n, action_type="buttons", seed=3,
                       worker_path=worker_path) as env:
        env.reset(seed=[3, 3])  # the same episode seed for both envs
        assert env.state_hash()[0] == env.state_hash()[1]
        for b in buttons:
            # env 0: one step of n frames; env 1: n steps of one frame (env 0 idles meanwhile... it
            # must not: step the envs separately with per-step frame counts)
            env.step(np.array([b, b], dtype=np.uint16), n_frames=np.array([n, 0]))
            for _ in range(n):
                env.step(np.array([b, b], dtype=np.uint16), n_frames=np.array([0, 1]))
            h = env.state_hash()
            assert h[0] == h[1], f"step(n={n}) != {n} single frames"
            f = env.frames()
            assert f[0] == f[1]


def test_hold_frames(worker_path):
    n, hold = 24, 5
    with pn.PokeVecEnv(num_envs=2, num_workers=1, frameskip=n, action_type="buttons", seed=9,
                       worker_path=worker_path) as env:
        env.reset(seed=[9, 9])
        for b in (pn.BUTTONS["START"], pn.BUTTONS["DOWN"], pn.BUTTONS["A"], pn.BUTTONS["B"], pn.BUTTONS["B"]):
            bb = np.array([b, b], dtype=np.uint16)
            env.step(bb, n_frames=np.array([n, 0]), hold_frames=hold)
            env.step(bb, n_frames=np.array([0, hold]))
            env.step(np.zeros(2, np.uint16), n_frames=np.array([0, n - hold]))
            h = env.state_hash()
            assert h[0] == h[1]


def test_pixels_are_the_final_frame(worker_path):
    with pn.PokeVecEnv(num_envs=2, num_workers=2, frameskip=24, pixels=True, action_type="buttons", seed=4,
                       worker_path=worker_path) as env:
        env.reset(seed=[4, 4])
        b = np.array([pn.BUTTONS["UP"]] * 2, dtype=np.uint16)
        obs, *_ = env.step(b, n_frames=np.array([24, 0]))
        for _ in range(24):
            env.step(b, n_frames=np.array([0, 1]))
        img = env.render([1])[0]
        assert np.array_equal(obs["pixels"][0], img)
