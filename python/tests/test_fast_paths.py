"""The throughput fast paths change no trajectory.

- ``step``'s fast path (64-bit command words, one status reduction) against the general
  field-by-field path (taken for per-env ``n_frames`` arrays);
- asynchronous batches over futex-waiting workers, woken through a wake-up tree over the posted
  workers (WCMD_TREE_LIST), against synchronous steps;
- the skipped FUTEX_WAKE of spinning workers (Dekker handshake) under alternating sync/async use;
- the default worker counts above 3/4 of the CPUs (one env per process up to 16 per CPU).
"""
import numpy as np
import pytest

import fireredsim as pn
from fireredsim import _ipc


def _actions(n_steps, n_envs, seed=5):
    return np.random.default_rng(seed).integers(0, 9, size=(n_steps, n_envs))


def test_step_fast_path_matches_general_path(worker_path):
    acts = _actions(30, 3)
    out = []
    for general in (False, True):
        with pn.PokeVecEnv(num_envs=3, seed=11, max_episode_steps=7, worker_path=worker_path) as env:
            env.reset()
            rec = []
            for a in acts:
                n = np.full(3, 24) if general else None  # an array of n_frames takes the general path
                o, r, te, tr, info = env.step(a, n_frames=n)
                rec.append((o.tobytes(), r.tobytes(), te.tobytes(), tr.tobytes(),
                            [(k, np.asarray(v).dtype.str, np.asarray(v).tobytes()) for k, v in info.items()]))
            rec.append(env.state_hash().tobytes())
            out.append(rec)
    assert out[0] == out[1]


def test_step_fast_path_words():
    # the 64-bit words write exactly the fields the general path writes
    buf = np.zeros(2 * _ipc.CMD_DTYPE.itemsize, dtype=np.uint8)
    cmds = np.ndarray((2,), _ipc.CMD_DTYPE, buf, 0)
    w0 = np.ndarray((2,), np.dtype("<u8"), buf, 0, (_ipc.CMD_DTYPE.itemsize,))
    w8 = np.ndarray((2,), np.dtype("<u8"), buf, 8, (_ipc.CMD_DTYPE.itemsize,))
    w0[:] = _ipc.OP_STEP | (_ipc.F_OBS | _ipc.F_RNG_RESEED) << 32
    w8[:] = np.array([0x3FF, 0x41], dtype=np.uint64) | np.uint64((24 << 16) | (1 << 32))
    assert cmds["op"].tolist() == [_ipc.OP_STEP] * 2
    assert cmds["flags"].tolist() == [_ipc.F_OBS | _ipc.F_RNG_RESEED] * 2
    assert cmds["buttons"].tolist() == [0x3FF, 0x41]
    assert cmds["n_frames"].tolist() == [24, 24] and cmds["hold_frames"].tolist() == [1, 1]
    assert cmds["timer1_count"].tolist() == [0, 0]


@pytest.mark.parametrize("spin", [0, None])
def test_async_tree_wakeups_match_sync(worker_path, spin):
    """9 workers (> tree fanout 4): with spin 0 every async send of more than 4 workers goes
    through a wake-up tree over the posted workers."""
    n = 9
    acts = _actions(14, n, seed=8)
    with pn.PokeVecEnv(num_envs=n, seed=21, worker_path=worker_path, spin_us=spin, client_spin_us=spin) as ref:
        ref.reset()
        ref_obs = []
        for a in acts:
            o, *_ = ref.step(a)
            ref_obs.append(o.tobytes())
        ref_h = ref.state_hash()
    with pn.PokeVecEnv(num_envs=n, seed=21, worker_path=worker_path, spin_us=spin, client_spin_us=spin) as env:
        if spin == 0:
            assert env._ws._list_fanout == 4
        env.reset()
        steps = np.zeros(n, dtype=int)
        per_env = [[] for _ in range(n)]
        env.send(acts[0])
        while (steps < len(acts)).any():
            obs, _, _, _, info = env.recv(min_envs=1)
            ids = info["env_id"]
            for j, e in enumerate(ids.tolist()):
                per_env[e].append(obs[j].tobytes())
            steps[ids] += 1
            todo = ids[steps[ids] < len(acts)]
            if todo.size:
                env.send(acts[steps[todo], todo], env_ids=todo)
        assert (env.state_hash() == ref_h).all()
        rec = env._ws.obs_bytes.shape[1]
        for e in range(n):
            assert per_env[e] == [o[e * rec:(e + 1) * rec] for o in ref_obs]
        # a partial synchronous batch (reset of 6 envs) through the same tree, then sync steps again
        env.reset(env_ids=np.arange(6))
        env.step(acts[0])


def test_alternating_sync_async_and_sleep(worker_path):
    """Workers that went to sleep (spin window over) are still woken by the fast path."""
    import time
    with pn.PokeVecEnv(num_envs=3, seed=2, worker_path=worker_path, spin_us=50, client_spin_us=50) as env:
        env.reset()
        for k in range(6):
            env.step(np.zeros(3, dtype=int))
            time.sleep(0.002)  # > spin_us: the workers are in FUTEX_WAIT now
            env.send(np.zeros(3, dtype=int))
            env.recv()
        assert (env.frames() > 0).all()


def test_many_envs_one_process_each():
    from fireredsim.vec_env import default_num_workers
    assert default_num_workers(256, 32) == 256
    assert default_num_workers(512, 32) == 512
    assert default_num_workers(600, 32) == 512


def test_step_per_env_hold_words():
    # per-env hold_frames / n_frames arrays keep the fast path and write each env's fields (test_step_fast_path_words)
    buf = np.zeros(3 * _ipc.CMD_DTYPE.itemsize, dtype=np.uint8)
    cmds = np.ndarray((3,), _ipc.CMD_DTYPE, buf, 0)
    w8 = np.ndarray((3,), np.dtype("<u8"), buf, 8, (_ipc.CMD_DTYPE.itemsize,))
    env = pn.PokeVecEnv.__new__(pn.PokeVecEnv)
    env.action_type = "discrete"
    env.actions = np.array([0, 1, 2, 8, 64, 128, 32, 16], dtype=np.uint16)
    env._fast_words = None
    words = env._discrete_words(np.array([4, 1, 7]), 3, 24, np.array([1, 10, 0xFFFF]))
    assert words is not None
    w8[:] = words
    assert cmds["buttons"].tolist() == [64, 1, 16]
    assert cmds["n_frames"].tolist() == [24] * 3 and cmds["hold_frames"].tolist() == [1, 10, 0xFFFF]
    assert cmds["timer1_count"].tolist() == [0, 0, 0]
    # per-env n_frames with a scalar hold, and both arrays
    w8[:] = env._discrete_words(np.array([4, 1, 7]), 3, np.array([24, 25, 1]), 1)
    assert cmds["n_frames"].tolist() == [24, 25, 1] and cmds["hold_frames"].tolist() == [1, 1, 1]
    w8[:] = env._discrete_words(np.array([4, 1, 7]), 3, np.array([25, 24, 24]), np.array([10, 1, 4]))
    assert cmds["n_frames"].tolist() == [25, 24, 24] and cmds["hold_frames"].tolist() == [10, 1, 4]
    assert cmds["buttons"].tolist() == [64, 1, 16]
    # scalars after arrays and back: the cached table is re-keyed, the words are the same
    w8[:] = env._discrete_words(np.array([4, 1, 7]), 3, 24, 1)
    assert cmds["hold_frames"].tolist() == [1, 1, 1] and cmds["n_frames"].tolist() == [24] * 3
    assert cmds["buttons"].tolist() == [64, 1, 16]
    w8[:] = env._discrete_words(np.array([4, 1, 7]), 3, 24, np.array([2, 2, 2]))
    assert cmds["hold_frames"].tolist() == [2, 2, 2]
    # a wrong length, a negative hold or an n_frames above 0xFFFF: the general path
    assert env._discrete_words(np.array([4, 1, 7]), 3, 24, np.array([1, 2])) is None
    assert env._discrete_words(np.array([4, 1, 7]), 3, 24, np.array([1, -1, 2])) is None
    assert env._discrete_words(np.array([4, 1, 7]), 3, np.array([24, 70000, 24]), 1) is None


def test_per_env_hold_fast_path_matches_general_path(worker_path):
    # per-env holds and frame counts through the words against the field-by-field path
    acts = _actions(30, 3)
    holds = np.random.default_rng(9).integers(1, 12, size=(30, 3))
    out = []
    for general in (False, True):
        with pn.PokeVecEnv(num_envs=3, seed=11, max_episode_steps=7, worker_path=worker_path) as env:
            env.reset()
            rec = []
            for a, h in zip(acts, holds):
                # the general path: an n_frames array with an entry above 0xFFFF is refused by the words, so force
                # it with a float hold (the field path casts); both run 24 or 25 frames per env
                n = np.where(np.arange(3) == 1, 25, 24)
                o, r, te, tr, info = env.step(a, n_frames=n, hold_frames=h.astype(np.float64) if general else h)
                rec.append((o.tobytes(), r.tobytes(), te.tobytes(), tr.tobytes()))
            rec.append(env.state_hash().tobytes())
            out.append(rec)
    assert out[0] == out[1]
    # and the holds reach the game: in the bedroom LEFT (basic action 7) held 1 frame only turns, held 12 frames
    # turns and walks (a different state after one step), through the words and through the fields alike
    hashes = []
    for h in (np.array([1, 1, 12]), np.array([1, 1, 12]), 1):
        with pn.PokeVecEnv(num_envs=3, seed=11, start="bedroom", worker_path=worker_path) as env:
            env.reset()
            env.step(np.array([7, 7, 7]), n_frames=np.full(3, 24) if len(hashes) == 1 else None, hold_frames=h)
            hashes.append(env.state_hash().tolist())
    assert hashes[0] == hashes[1]
    assert hashes[0][0] == hashes[2][0] and hashes[0][1] == hashes[2][1] and hashes[0][2] != hashes[2][2]
