"""Observation records: the NumPy dtype of the worker's observation bytes.

The worker writes one fixed-size record per env (``PnObs``, obs/observation.h; its
dtype is generated into :mod:`fireredsim.obs_dtype` by ``tools/gen_obs_dtype.py``).
Fields ending in ``_priv`` (and everything under such a field) are privileged:
zero in ``visible`` mode, filled in ``privileged`` mode.

Before the observation library existed the worker used a 64-byte placeholder
record (``STUB_DTYPE``); a worker built that way still works with this package.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

# worker/obs_bridge.h PnObsStub
STUB_DTYPE = np.dtype({
    "names": ["magic", "version", "size", "mode", "in_battle", "map_group", "map_num", "x", "y", "facing",
              "text_printer_active", "script_enabled", "field_controls_locked", "held_keys", "new_keys",
              "vblank_counter2", "callback2_priv", "rng_priv"],
    "formats": ["<u4", "<u2", "<u2", "u1", "u1", "u1", "u1", "<i2", "<i2", "u1", "u1", "u1", "u1", "<u2", "<u2",
                "<u4", "<u4", "<u4"],
    "offsets": [0, 4, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 24, 28, 32],
    "itemsize": 64,
})

def resolve_obs_dtype(info: dict) -> Tuple[np.dtype, str]:
    """(dtype, kind) for the worker described by ``info`` (``fireredsim-worker --info``).

    kind: ``"obs"`` (the observation contract), ``"stub"`` (placeholder record) or
    ``"opaque"`` (bytes: the generated dtype does not match the worker; rebuild).
    """
    size = int(info["obs_size"])
    if info.get("obs_stub"):
        return STUB_DTYPE, "stub"
    try:
        from .obs_dtype import OBS_DTYPE  # generated
    except ImportError:  # pragma: no cover - generated file missing
        OBS_DTYPE = None
    if OBS_DTYPE is not None and OBS_DTYPE.itemsize == size:
        return OBS_DTYPE, "obs"
    warnings.warn(f"fireredsim.obs_dtype ({None if OBS_DTYPE is None else OBS_DTYPE.itemsize} bytes) does not "
                  f"match the worker's observation ({size} bytes): observations are raw bytes. Rebuild "
                  f"(tools/gen_obs_dtype.py) so that both come from the same obs/observation.h.")
    return np.dtype([("bytes", np.uint8, (size,))]), "opaque"


_CHARMAP: Optional[Dict[int, str]] = None


def decode_text(raw) -> str:
    """Game-encoded string (0xFF terminated) -> str (uses obs_dtype.decode_text when available)."""
    try:
        from .obs_dtype import decode_text as _dec  # generated
        return _dec(raw)
    except ImportError:  # pragma: no cover
        b = bytes(np.asarray(raw, dtype=np.uint8))
        return b.split(b"\xff", 1)[0].hex()
