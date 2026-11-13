"""pytest fixtures for python/fireredsim.

The tests need a built worker: cmake --build build-worker --target fireredsim-worker
(and fireredsim-render for the pixel parity test). FIREREDSIM_WORKER selects the
executable; otherwise the newest build*/worker/fireredsim-worker of the repo.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))

import fireredsim  # noqa: E402
from fireredsim import _paths  # noqa: E402


@pytest.fixture(scope="session")
def worker_path() -> str:
    try:
        return _paths.find_worker()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def repo() -> str:
    return REPO


def parse_script(path: str):
    """An input script (lines ``START END BUTTONS``) -> list of (start, end, mask)."""
    ranges = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            a, b, btn = line.split()
            ranges.append((int(a), int(b), fireredsim.parse_buttons(btn)))
    return ranges


def script_buttons(ranges, n_frames: int, offset: int = 0) -> np.ndarray:
    out = np.zeros(n_frames, dtype=np.uint16)
    for a, b, m in ranges:
        a, b = max(a - offset, 0), min(b - offset, n_frames)
        if a < b:
            out[a:b] |= m
    return out
