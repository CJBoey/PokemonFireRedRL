"""Rewind rings: per env, its own recent captures to retry from on stagnation (env.py "Rewind").

A **capture** of env i is a fast snapshot of its state right after a progress event, written by the worker into
the snapshot slot ``reserved_base + i * size + slot`` of the env group's shared memory (``PokeRLEnv``; the ring holds
no bytes), with ``ep_step`` (the life's step count when taken), a **slot** and a **serial**:

* ``slot`` in ``0 .. size-1`` is unique among env i's live captures: the PPO collector keeps the matching policy
  state (recurrent state, previous action) in a ``[N, size]`` store indexed by ``(env, slot)``. A new capture takes a
  free slot, or the slot of the oldest capture it evicts. (Not a running integer modulo ``size``: after a rewind
  past newer captures, the next capture would then land on the slot of a live older one.)
* ``serial`` is a running integer per env (0, 1, 2, ...): the collector checks it on every rewind, so a policy
  state stored for another capture can never be injected silently.

The ring holds at most ``size`` captures per env, newest first. :meth:`RewindRing.choose` picks one of the newest
``min(depth, count)`` uniformly and discards the captures newer than it (they are its future); the chosen one and
the older ones stay. :meth:`clear` empties the rings of ended lives.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

MAX_CAPTURES_PER_STEP = 8  # snapshot rows per env-group step at most (oldest last capture first)


class Capture:
    """One capture (its bytes are in the caller's worker slot)."""
    __slots__ = ("slot", "serial", "ep_step")

    def __init__(self, slot: int, serial: int, ep_step: int):
        self.slot, self.serial, self.ep_step = int(slot), int(serial), int(ep_step)


class RewindRing:
    """The capture rings of ``num_envs`` envs (one env group). ``count[i]`` (captures held) and ``last_step[i]``
    (``ep_step`` of the newest, -1 without) are kept as arrays for the env's vectorised capture rule."""

    def __init__(self, num_envs: int, size: int = 1, capture_bytes: int = 0):
        if size < 1:
            raise ValueError("RewindRing: size must be >= 1")
        self.num_envs, self.size = int(num_envs), int(size)
        self.capture_bytes = int(capture_bytes)  # bytes of one capture in its worker slot (memory_bytes)
        self.rings: List[List[Capture]] = [[] for _ in range(self.num_envs)]  # newest first
        self.count = np.zeros(self.num_envs, dtype=np.int64)
        self.last_step = np.full(self.num_envs, -1, dtype=np.int64)
        self.serial = np.zeros(self.num_envs, dtype=np.int64)  # next serial per env

    def push(self, rows: Sequence[int], ep_steps: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Adds one capture per row (distinct rows); returns their (slots, serials) (int64 [k]). The caller writes
        each one's bytes to its (row, slot) after this call."""
        rows = np.asarray(rows).tolist()
        k = len(rows)
        slots = np.zeros(k, dtype=np.int64)
        serials = np.zeros(k, dtype=np.int64)
        for j, (r, st) in enumerate(zip(rows, np.asarray(ep_steps).tolist())):
            ring = self.rings[r]
            if len(ring) >= self.size:
                slot = ring.pop().slot  # evict the oldest: its slot is reused
            else:
                used = {c.slot for c in ring}
                slot = next(s for s in range(self.size) if s not in used)
            serial = int(self.serial[r])
            c = Capture(slot, serial, st)
            self.serial[r] = serial + 1
            ring.insert(0, c)
            self.count[r] = len(ring)
            self.last_step[r] = c.ep_step
            slots[j], serials[j] = slot, serial
        return slots, serials

    def choose(self, row: int, depth: int, rng, now: Optional[int] = None) -> Capture:
        """The capture to rewind env ``row`` to: uniformly one of the newest ``min(depth, count)`` (``rng.integers``
        is drawn only when there is a choice); the captures newer than it are discarded. ``now`` (the life's step
        count at the rewind) becomes the gap reference (``last_step``), so the next capture waits the full
        ``rewind_min_gap`` after the restored point; None keeps the restored capture's own ``ep_step`` (the
        life's counter runs on across a rewind, so that would allow a capture right after the rewind)."""
        ring = self.rings[row]
        if not ring:
            raise ValueError(f"RewindRing.choose: env {row} has no capture")
        k = min(max(1, int(depth)), len(ring))
        j = int(rng.integers(k)) if k > 1 else 0
        c = ring[j]
        del ring[:j]
        self.count[row] = len(ring)
        self.last_step[row] = ring[0].ep_step if now is None else int(now)
        return c

    def captures(self, row: int) -> List[Capture]:
        """Env ``row``'s captures, newest first (a copy of the list)."""
        return list(self.rings[row])

    def clear(self, rows) -> None:
        for r in np.asarray(rows, dtype=np.int64).reshape(-1).tolist():
            self.rings[r] = []
        self.count[rows] = 0
        self.last_step[rows] = -1

    def memory_bytes(self) -> int:
        """Bytes of the captures held (``capture_bytes`` each, in the worker slots)."""
        return int(self.count.sum()) * self.capture_bytes
