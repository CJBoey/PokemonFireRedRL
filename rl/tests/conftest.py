"""Shared fixtures: synthetic observation records (fireredsim.obs_dtype.OBS_DTYPE) and the worker."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(ROOT, "rl"), os.path.join(ROOT, "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fireredsim.obs_dtype import OBS_DTYPE  # noqa: E402

PALLET = (3, 0)       # MAP_PALLET_TOWN: group 3, num 0 (24 x 20)
ROUTE1 = (3, 19)      # MAP_ROUTE1 (24 x 40)


def records(n=2, map_=PALLET, x=5, y=5):
    """n in-game records on the field, no party."""
    r = np.zeros(n, dtype=OBS_DTYPE)
    r["valid"] = 1 | 2
    r["map_group"], r["map_num"] = map_
    r["x"], r["y"] = x, y
    r["field_state"] = 1
    return r


def set_flag(r, flag, value=True, rows=slice(None)):
    byte, bit = flag >> 3, 1 << (flag & 7)
    if value:
        r["flags_priv"][rows, byte] |= bit
    else:
        r["flags_priv"][rows, byte] &= ~np.uint8(bit)


def set_mon(r, slot, species, level=5, hp=10, moves=(), egg=False, rows=slice(None)):
    p = r["party"]
    p["species"][rows, slot] = species
    p["level"][rows, slot] = level
    p["hp"][rows, slot] = hp
    p["max_hp"][rows, slot] = max(hp, 1)
    p["is_egg"][rows, slot] = 1 if egg else 0
    mv = list(moves) + [0] * (4 - len(moves))
    p["moves"][rows, slot] = mv


def worker_available():
    try:
        import fireredsim
        fireredsim.find_worker(None)
        return True
    except Exception:
        return False


needs_worker = pytest.mark.skipif(not worker_available(), reason="no fireredsim-worker built")
needs_cuda = pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA")
