"""Terminal statuses and worker failures: an env that ends never takes the others down."""
import os
import signal
import time

import numpy as np
import pytest

import fireredsim as pn
from fireredsim import _ipc

GMAIN = 0x030030F0  # reference/rom_symbols.txt (mirror layout: game RAM at its ROM address)
HEADER = 72 + 8 * 8  # snapshot header + region table (8 regions)
IWRAM_OFFSET = 0x40000  # regions in order ewram (256 KiB), iwram, ...


def corrupt_callback2(blob: bytes) -> bytes:
    b = bytearray(blob)
    off = HEADER + IWRAM_OFFSET + (GMAIN - 0x03000000) + 4  # gMain.callback2
    b[off:off + 4] = (0x100).to_bytes(4, "little")  # an address in the GBA BIOS region
    b[32:40] = bytes(8)  # no state hash: restored without verification
    return bytes(b)


def test_terminal_status_is_reported_and_isolated(worker_path):
    with pn.PokeVecEnv(num_envs=3, num_workers=1, seed=2, worker_path=worker_path, autoreset=True) as env:
        env.reset()
        snap = env.snapshot([1])[0]
        env.restore([corrupt_callback2(snap)], [1])
        obs, rew, term, trunc, info = env.step(np.array([0, 0, 0]))
        assert info["sim_terminal"].tolist() == [False, True, False]
        assert trunc.tolist() == [False, True, False]
        assert _ipc.TERMINAL_KINDS[int(info["terminal_kind"][1])] == "CRASHED"
        assert "BIOS" in info["terminal_message"][1]
        assert info["frames_run"][0] == 24 and info["frames_run"][1] == 0
        # next step: env 1 is reset automatically (Gymnasium next-step autoreset), the others step
        obs, rew, term, trunc, info = env.step(np.array([5, 5, 5]))
        assert info["autoreset"].tolist() == [False, True, False]
        assert not info["sim_terminal"].any()
        obs, rew, term, trunc, info = env.step(np.array([5, 5, 5]))
        assert info["frames_run"].tolist() == [24, 24, 24]


def test_terminal_without_autoreset_stays_ended(worker_path):
    with pn.PokeVecEnv(num_envs=2, num_workers=2, seed=2, worker_path=worker_path, autoreset=False) as env:
        env.reset()
        env.restore([corrupt_callback2(env.snapshot([0])[0])], [0])
        env.step(np.array([0, 0]))
        h = env.state_hash()[0]
        _, _, _, trunc, info = env.step(np.array([1, 1]))
        assert trunc[0] and info["frames_run"][0] == 0 and env.state_hash()[0] == h
        env.reset(env_ids=[0])
        _, _, _, trunc, info = env.step(np.array([1, 1]))
        assert not trunc.any() and info["frames_run"][0] == 24


def test_soft_reset_is_reported(worker_path):
    combo = pn.BUTTONS["A"] | pn.BUTTONS["B"] | pn.BUTTONS["START"] | pn.BUTTONS["SELECT"]
    with pn.PokeVecEnv(num_envs=1, seed=3, action_type="buttons", worker_path=worker_path) as env:
        env.reset()
        _, _, _, _, info = env.step(np.array([combo], dtype=np.uint16), hold_frames=2)
        assert info["soft_reset"][0]
        assert not info["sim_terminal"][0]
        # the game keeps running (powered on again, flash kept)
        _, _, _, _, info = env.step(np.array([0], dtype=np.uint16))
        assert info["frames_run"][0] == 24 and not info["soft_reset"][0]


def test_worker_crash_raises(worker_path):
    env = pn.PokeVecEnv(num_envs=4, num_workers=2, seed=1, worker_path=worker_path)
    try:
        env.reset()
        os.kill(env.worker_pids[1], signal.SIGKILL)
        time.sleep(0.05)
        with pytest.raises(pn.WorkerError, match=r"worker 1 .*envs 2\.\.3"):
            env.step(np.zeros(4, dtype=np.int64))
    finally:
        env.close()


def test_bad_worker_path():
    with pytest.raises(FileNotFoundError):
        pn.PokeVecEnv(num_envs=1, worker_path="/nonexistent/fireredsim-worker")
