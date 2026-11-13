"""Fast field access for observation records (helpers of events.py and the native spec).

Reading ~60 nested fields of a structured array one by one costs a few µs of numpy overhead each.
Instead, a :class:`Gather` plan turns a list of (possibly nested, array-valued) leaf fields of one
scalar type into byte offsets computed once from the dtype, and reads all of them with one
fancy-index of the records' raw bytes (``[N, itemsize]`` view, no copy for contiguous records such as
``PokeVecEnv(copy=False)`` views). The result is a fresh contiguous array.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def leaf_offsets(dtype: np.dtype, path: Sequence[str]) -> Tuple[np.ndarray, np.dtype]:
    """Byte offsets (shaped like the leaf's sub-array dims, outermost first) and scalar dtype of a
    leaf field, e.g. ``("party", "moves")`` -> offsets [6, 4], ``<u2``."""
    offs = np.zeros((), dtype=np.int64)
    dt = dtype
    for name in path:
        fdt, off = dt.fields[name][:2]
        offs = offs + off
        if fdt.subdtype is not None:
            base, shape = fdt.subdtype
            k = (np.arange(int(np.prod(shape))) * base.itemsize).reshape(shape)
            offs = offs.reshape(offs.shape + (1,) * len(shape)) + k
            dt = base
        else:
            dt = fdt
    return offs, dt


def as_bytes(obs: np.ndarray) -> np.ndarray:
    """[N, itemsize] uint8 view of records (a copy only if they are not contiguous)."""
    if not obs.flags.c_contiguous:
        obs = np.ascontiguousarray(obs)
    return obs.view(np.uint8).reshape(obs.shape[0], obs.dtype.itemsize)


class Gather:
    """Reads leaf fields of one scalar type (``scalar``: a numpy dtype such as ``np.uint16``) from
    records' bytes: ``g(rec_bytes)`` -> [N, K] array; ``g.cols[name]`` = (start, stop, shape)."""

    def __init__(self, dtype: np.dtype, scalar, fields: Sequence[Tuple[str, Sequence[str]]]):
        self.scalar = np.dtype(scalar)
        size = self.scalar.itemsize
        idx: List[np.ndarray] = []
        self.cols: Dict[str, Tuple[int, int, tuple]] = {}
        o = 0
        for name, path in fields:
            offs, dt = leaf_offsets(dtype, path)
            if dt.itemsize != size or dt.kind != self.scalar.kind:
                raise TypeError(f"field {path}: {dt}, not {self.scalar}")
            if np.any(offs % size):
                raise ValueError(f"field {path}: unaligned for {self.scalar}")
            idx.append((offs // size).reshape(-1))
            self.cols[name] = (o, o + offs.size, offs.shape)
            o += offs.size
        self.index = np.concatenate(idx).astype(np.intp)
        self.width = o

    def __call__(self, rec: np.ndarray) -> np.ndarray:
        view = rec if self.scalar.itemsize == 1 and self.scalar == rec.dtype else rec.view(self.scalar)
        return view[:, self.index]

    def get(self, out: np.ndarray, name: str) -> np.ndarray:
        a, b, shape = self.cols[name]
        return out[:, a:b].reshape(out.shape[0], *shape)


def byte_range(dtype: np.dtype, name: str) -> Tuple[int, int]:
    """(start, stop) byte range of a top-level field."""
    fdt, off = dtype.fields[name][:2]
    return off, off + fdt.itemsize


def prefix_dtype(dtype: np.dtype, size: int):
    """The fields of ``dtype`` that lie entirely in the first ``size`` bytes, as a dtype of itemsize
    ``size`` (an older observer layout whose fields are a prefix of the current one, e.g. layout v2 =
    v3 without ``tiles``), or None if a field straddles ``size``."""
    names, formats, offsets = [], [], []
    for name in dtype.names:
        fdt, off = dtype.fields[name][:2]
        if off + fdt.itemsize <= size:
            names.append(name)
            formats.append(fdt)
            offsets.append(off)
        elif off < size:
            return None
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": size})
