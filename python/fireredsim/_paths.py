"""Locating the native ``fireredsim-worker`` executable."""
from __future__ import annotations

import glob
import os
import shutil
from typing import List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def worker_candidates() -> List[str]:
    pats = [os.path.join(REPO_ROOT, "build*", "worker", "fireredsim-worker")]
    out = []
    for p in pats:
        out += [c for c in glob.glob(p) if os.access(c, os.X_OK)]
    w = shutil.which("fireredsim-worker")
    if w:
        out.append(w)
    return out


def find_worker(path: Optional[str] = None) -> str:
    """The worker executable: ``path``, else $FIREREDSIM_WORKER, else the most recently built
    ``build*/worker/fireredsim-worker`` of this repository, else ``fireredsim-worker`` on PATH."""
    if path:
        if not os.access(path, os.X_OK):
            raise FileNotFoundError(f"worker executable not found: {path}")
        return path
    env = os.environ.get("FIREREDSIM_WORKER")
    if env:
        return find_worker(env)
    cands = worker_candidates()
    if not cands:
        raise FileNotFoundError(
            "fireredsim-worker not found. Build it: cmake -S . -B build-worker -G Ninja -DCMAKE_BUILD_TYPE=Release "
            "&& cmake --build build-worker --target fireredsim-worker (or set FIREREDSIM_WORKER)")
    return max(cands, key=lambda c: os.stat(c).st_mtime)
