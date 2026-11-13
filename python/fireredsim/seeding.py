"""Seeds -> timer-1 values (the game's only source of randomness).

FireRed seeds its RNG and draws the trainer ID from the free-running hardware
timer 1 (``SeedRngAndSetTrainerId``: once when START is pressed on the title
screen, once when the player's naming screen closes). The simulator takes the
successive values of those reads from the reset command; this module derives
them from integer seeds, deterministically and independently of how the
environments are spread over worker processes:

* ``episode_seeds(seed, n)``: the first episode seed of env i (splitmix64 of
  the base seed and i);
* ``next_episode_seed(s)``: the seed of the next episode of the same env;
* ``timer1_for_seed(s)``: the two 16-bit timer-1 values of an episode;
* ``rng_stream_seed(s)``: the episode's initial host entropy stream state of the
  ``rng_reseed`` option, and ``rng_stream_values`` the values
  that stream writes into the game's PRNG at successive steps.

For the ``bedroom`` start (a new game replayed from power-on) both values
matter (the second one is the trainer ID and the RNG seed of the game). For
``power_on`` they are used when the agent reaches those screens itself. For a
snapshot start they only matter for reads that have not happened yet in that
snapshot; diversity then has to come from ``noop_reset_max`` or the policy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

_M64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def episode_seeds(seed: int, n: int, offset: int = 0) -> np.ndarray:
    """First episode seed of envs offset .. offset+n-1 for a base seed."""
    base = splitmix64(int(seed) & _M64)
    return np.array([splitmix64(base ^ splitmix64(i + offset)) for i in range(n)], dtype=np.uint64)


def next_episode_seed(s: int) -> int:
    return splitmix64(int(s) ^ 0xD1B54A32D192ED03)


def timer1_for_seed(s: int) -> Tuple[int, int]:
    """(title-screen read, naming-screen read) for an episode seed."""
    h = splitmix64(int(s) ^ 0x5EED_7131)
    return h & 0xFFFF, (h >> 16) & 0xFFFF


def rng_stream_seed(s: int) -> int:
    """Initial state of the ``rng_reseed`` host entropy stream of an episode (PnEnvCmd.rng_seed)."""
    return splitmix64(int(s) ^ 0x524E_4752_4553_4544)  # "RNGRESED"


def rng_stream_values(stream: int, n: int):
    """(values, next_state): the first ``n`` (gRngValue, sWildEncounterData.rngState) pairs that
    an env whose stream state is ``stream`` writes into the game at its next ``n`` steps with
    ``rng_reseed`` (sim/sim.cpp ``rng_stream_next``: one splitmix64 output per step, low 32 bits
    -> gRngValue, high 32 bits -> the wild-encounter RNG)."""
    out = np.zeros((n, 2), dtype=np.uint32)
    x = int(stream) & _M64
    for k in range(n):
        x = (x + 0x9E3779B97F4A7C15) & _M64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        z ^= z >> 31
        out[k, 0] = z & 0xFFFFFFFF
        out[k, 1] = z >> 32
    return out, x


def noop_frames_for_seed(s: int, noop_max: int) -> int:
    if noop_max <= 0:
        return 0
    return int(splitmix64(int(s) ^ 0x0F0F_0F0F) % (noop_max + 1))


def random_seed() -> int:
    return int.from_bytes(np.random.default_rng().bytes(8), "little")
