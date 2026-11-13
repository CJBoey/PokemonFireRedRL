"""Shared-memory protocol with the 32-bit ``fireredsim-worker`` processes.

Mirror of ``worker/pn_ipc.h``. The NumPy dtypes below
give the byte layout of the C structs; :func:`check_layout` compares them with
the offsets the worker executable reports (``fireredsim-worker --info``), so a
mismatch fails loudly at start-up instead of corrupting memory.

Synchronisation: x86 total store order. Python writes the per-env commands,
then each participating worker's ``seq``, then ``pending``, then increments
``go`` and FUTEX_WAKEs it; workers decrement ``pending`` atomically and the last
one FUTEX_WAKEs it. Python waits on ``pending`` (optionally spinning first).
"""
from __future__ import annotations

import ctypes
import platform
import json
import mmap
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

SHM_MAGIC = 0x48534E50
SHM_VERSION = 3
PAGE = 4096
PIXEL_W, PIXEL_H = 240, 160
PIXEL_BYTES = PIXEL_W * PIXEL_H * 3
TIMER1_MAX = 8


class PixelFormat(NamedTuple):
    """One pixel format: its code (``PnShmHeader.pixels``, per-op format in
    ``PnEnvCmd.flags`` bits 24-31), the uint8 shape of one frame and its bytes."""
    code: int
    shape: Tuple[int, ...]
    bytes: int


def _gray_format(code: int) -> PixelFormat:
    k = code & 0x0F
    h, w = -(-PIXEL_H // k), -(-PIXEL_W // k)
    return PixelFormat(code, (h, w), h * w)


# name -> PixelFormat. Code: bits 0-3 = scale k, bit 4 = gray box (mean of each k x k block), bit 5 = gray
# sub-sampled lines (only scanlines 0, k, 2k, ... composed; mean of k pixels); 1 = RGB8 [160][240][3].
# Luma Y = (77 R + 150 G + 29 B + 128) >> 8 of the RGB frame; mean of n lumas = (sum + n // 2) // n.
PIXEL_FORMATS: Dict[str, PixelFormat] = {
    "rgb": PixelFormat(1, (PIXEL_H, PIXEL_W, 3), PIXEL_BYTES),
    "gray1": _gray_format(0x11),
    "gray2": _gray_format(0x12),
    "gray3": _gray_format(0x13),
    "gray4": _gray_format(0x14),
    "gray2s": _gray_format(0x22),
    "gray3s": _gray_format(0x23),
    "gray4s": _gray_format(0x24),
}
PIXEL_FORMAT_NAMES = {f.code: n for n, f in PIXEL_FORMATS.items()}
F_FORMAT_SHIFT = 24  # PnEnvCmd.flags bits 24-31: this op's pixel format code (0 = the header's)


def pixel_format(name: str) -> PixelFormat:
    """The :class:`PixelFormat` of a format name (``PIXEL_FORMATS``); ValueError for an unknown name."""
    try:
        return PIXEL_FORMATS[name]
    except (KeyError, TypeError):
        raise ValueError(f"unknown pixel format {name!r} (one of {', '.join(PIXEL_FORMATS)})") from None


def pixel_shape(name: str) -> Tuple[int, ...]:
    """The uint8 shape of one frame of pixel format ``name``: (160, 240, 3) for "rgb", (H, W) for the gray ones."""
    return pixel_format(name).shape


def pixel_bytes(name: str) -> int:
    """Bytes of one frame of pixel format ``name``."""
    return pixel_format(name).bytes

# worker commands / states
WCMD_RUN = 1
WCMD_SHUTDOWN = 2
WCMD_TREE = 0x100
WCMD_TREE_LIST = 0x200  # wake the workers listed in the block's tree_child first (partial batches)
TREE_LIST_MAX = 8
WSTATE_STARTING, WSTATE_READY, WSTATE_EXITED, WSTATE_FAILED = 0, 1, 2, 3

# per-env ops and flags
OP_NONE, OP_STEP, OP_RESET, OP_SNAPSHOT, OP_RESTORE, OP_OBSERVE = 0, 1, 2, 3, 4, 5
F_OBS, F_RENDER, F_HASH, F_FAST = 1, 2, 4, 8
F_RNG_RESEED = 16  # STEP reseeds the game's PRNG first; RESET sets the stream (rng_seed)
F_AUTO_ADVANCE = 32  # STEP: decision-point stepping (max_frames, advance_frames)

# templates
TEMPLATE_POWER_ON = 0
TEMPLATE_BEDROOM = 1
TEMPLATE_SAVE = 2  # .sav path in the env's PnEnvInfo.message (written before the batch)
TEMPLATE_FILE0 = 16
TEMPLATE_SCRIPT0 = 1024

# result status bits
ST_SOFT_RESET = 1
ST_TERMINAL = 2
ST_ERROR = 4
ST_BEDROOM_CACHED = 8

ERRORS = {
    0: "NONE", 1: "BAD_OP", 2: "BAD_SLOT", 3: "BAD_TEMPLATE", 4: "BAD_SNAPSHOT",
    5: "NO_PIXELS", 6: "START_FAILED", 7: "EXCEPTION",
}

# platform/fireredsim_port.h PORT_TERMINAL_*
TERMINAL_KINDS = {0: "NONE", 1: "HUNG", 2: "CRASHED", 3: "UNSUPPORTED", 4: "INEXACT", 5: "INTERNAL"}

HEADER_DTYPE = np.dtype({
    "names": ["magic", "version", "num_envs", "num_workers", "obs_size", "obs_mode", "pixels",
              "snapshot_slots", "snapshot_size", "snapshot_stride", "state_size", "flags",
              "off_workers", "off_cmds", "off_results", "off_info", "off_obs", "off_pixels",
              "off_snapshots", "total_size", "build_id", "spin_us", "ready", "failed", "tree_fanout",
              "pixel_stride", "completions", "go", "pending"],
    "formats": ["<u4"] * 12 + ["<u8"] * 9 + ["<u4"] * 5 + ["<u4", "<u4", "<u4"],
    "offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
                48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 124, 128, 132, 136, 192, 256, 320],
    "itemsize": 384,
})
HDR_RENDER_HBLANK = 1
HDR_WAKE_EACH = 2
HDR_TRACE = 4  # workers write PnWorkerBlock.trace_* (CLOCK_MONOTONIC ns; bench/trace_batches.py)

WORKER_DTYPE = np.dtype({
    "names": ["cmd", "seq", "done_seq", "status", "env_begin", "env_end", "pid", "state",
              "busy_ns", "batches", "frames", "swap_ns", "wake", "sleeping",
              "trace_wake_ns", "trace_begin_ns", "trace_end_ns", "tree_child", "tree_children"],
    "formats": ["<u4"] * 8 + ["<u8"] * 4 + ["<u4", "<u4"] + ["<u8"] * 3 + [("<u2", (8,)), "<u4"],
    "offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 68, 72, 80, 88, 96, 112],
    "itemsize": 128,
})

CMD_DTYPE = np.dtype({
    "names": ["op", "flags", "buttons", "n_frames", "hold_frames", "timer1_count", "timer1",
              "template_id", "slot", "noop_frames", "rng_seed", "max_frames", "advance_frames"],
    "formats": ["<u4", "<u4", "<u2", "<u2", "<u2", "<u2", ("<u2", (TIMER1_MAX,)), "<u4", "<u4", "<u4", "<u8",
                "<u2", "<u2"],
    "offsets": [0, 4, 8, 10, 12, 14, 16, 32, 36, 40, 48, 56, 58],
    "itemsize": 64,
})

RESULT_DTYPE = np.dtype({
    "names": ["frame", "state_hash", "status", "terminal", "frames_run", "soft_resets", "error", "seq", "op_ns"],
    "formats": ["<u8", "<u8", "<u4", "<u4", "<u4", "<u4", "<u4", "<u4", "<u4"],
    "offsets": [0, 8, 16, 20, 24, 28, 32, 36, 40],
    "itemsize": 64,
})

INFO_DTYPE = np.dtype([("message", "S256")])

_STRUCTS = {
    "PnShmHeader": HEADER_DTYPE,
    "PnWorkerBlock": WORKER_DTYPE,
    "PnEnvCmd": CMD_DTYPE,
    "PnEnvResult": RESULT_DTYPE,
    "PnEnvInfo": INFO_DTYPE,
}


def supports_tree_list(info: dict) -> bool:
    """Whether the worker executable implements WCMD_TREE_LIST (``PnWorkerBlock.tree_child`` in its
    ``--info``; an older worker would not wake the listed children)."""
    return "tree_child" in info.get("structs", {}).get("PnWorkerBlock", {}).get("fields", {})


def supports_rng_reseed(info: dict) -> bool:
    """Whether the worker executable implements PN_F_RNG_RESEED (``PnEnvCmd.rng_seed`` in its
    ``--info``; older workers ignore the flag, so ``rng_reseed=True`` must not run on them)."""
    return "rng_seed" in info.get("structs", {}).get("PnEnvCmd", {}).get("fields", {})


def supports_pixel_formats(info: dict) -> bool:
    """Whether the worker executable renders pixel formats other than RGB (``pixel_formats`` in its ``--info``:
    ``PnShmHeader.pixel_stride`` and the per-op format of ``PnEnvCmd.flags`` bits 24-31)."""
    return "pixel_formats" in info


def supports_pixel_format(info: dict, name: str) -> bool:
    """Whether the worker executable renders pixel format ``name``. "rgb" works on every worker; the gray formats
    need one that lists their code in ``pixel_formats`` (an older worker would render RGB into the slot)."""
    fmt = pixel_format(name)
    if name == "rgb":
        return True
    return fmt.code in info.get("pixel_formats", ())


def supports_auto_advance(info: dict) -> bool:
    """Whether the worker executable implements PN_F_AUTO_ADVANCE (``PnEnvCmd.max_frames`` in its
    ``--info``; older workers ignore the flag and would return every non-waiting step, so
    ``auto_advance=True`` must not run on them)."""
    return "max_frames" in info.get("structs", {}).get("PnEnvCmd", {}).get("fields", {})


def supports_start_preset(info: dict) -> bool:
    """Whether the worker executable takes ``--preset-flag`` / ``--preset-var`` (the new-game preset of script
    starts; ``preset`` in its ``--info``: the id ranges it accepts). An older worker rejects the options at
    start-up, so a preset must not be requested from it."""
    return isinstance(info.get("preset"), dict)


def check_start_preset(info: dict, flags: Sequence[int], vars: Mapping[int, int], worker_path: str = "") -> None:
    """Raises WorkerError when a preset (``flags`` / ``vars`` not both empty) is requested from a worker without
    the options, ValueError for an id outside the worker's ranges (``--info`` ``preset``) or a value outside
    0..65535."""
    if not flags and not vars:
        return
    if not supports_start_preset(info):
        raise WorkerError(f"a new-game preset was requested but the worker {worker_path or '(this worker)'} does "
                          f"not implement --preset-flag / --preset-var (no 'preset' in its --info: rebuild it)")
    r = info["preset"]
    for f in flags:
        if not int(r["flag_min"]) <= int(f) <= int(r["flag_max"]):
            raise ValueError(f"preset flag {int(f):#x}: not a saved event flag ({int(r['flag_min']):#x}.."
                             f"{int(r['flag_max']):#x}; special flags from 0x4000 are not saved)")
    for v, val in vars.items():
        if not int(r["var_min"]) <= int(v) <= int(r["var_max"]):
            raise ValueError(f"preset var {int(v):#x}: not a saved game variable ({int(r['var_min']):#x}.."
                             f"{int(r['var_max']):#x})")
        if not 0 <= int(val) <= 0xFFFF:
            raise ValueError(f"preset var {int(v):#x} = {val}: values are 0..65535")


class WorkerError(RuntimeError):
    """A worker process died or failed; the message carries its last stderr lines."""


class LayoutError(RuntimeError):
    """The Python package and the worker executable disagree on the shared-memory layout."""


def check_layout(info: dict) -> None:
    """Compares the dtypes above with the worker's struct offsets (``--info``)."""
    structs = info.get("structs", {})
    for name, dt in _STRUCTS.items():
        c = structs.get(name)
        if c is None:
            raise LayoutError(f"worker --info lacks struct {name}")
        if c["size"] != dt.itemsize:
            raise LayoutError(f"{name}: C size {c['size']} != NumPy itemsize {dt.itemsize}")
        for fname, (off, size) in c["fields"].items():
            if fname not in dt.names:
                raise LayoutError(f"{name}.{fname} missing in the NumPy dtype")
            sub, doff = dt.fields[fname][0], dt.fields[fname][1]
            if doff != off or sub.itemsize != size:
                raise LayoutError(f"{name}.{fname}: C (offset {off}, size {size}) != NumPy "
                                  f"(offset {doff}, size {sub.itemsize})")
    # fields in formerly reserved space, listed apart (an older Python rejects unknown "structs" fields)
    for name, fields in info.get("struct_ext", {}).items():
        dt = _STRUCTS.get(name)
        if dt is None:
            continue
        for fname, (off, size) in fields.items():
            if fname not in dt.names:
                continue  # a newer worker's extension this Python does not use (the area stays 0)
            sub, doff = dt.fields[fname][0], dt.fields[fname][1]
            if doff != off or sub.itemsize != size:
                raise LayoutError(f"{name}.{fname}: C (offset {off}, size {size}) != NumPy "
                                  f"(offset {doff}, size {sub.itemsize})")
    if info.get("shm_version") != SHM_VERSION:
        raise LayoutError(f"shared-memory version: worker {info.get('shm_version')}, Python {SHM_VERSION}")


# ---- futex ----------------------------------------------------------------------------------

_libc = ctypes.CDLL(None, use_errno=True)
_SYS_FUTEX = 202  # x86_64
FUTEX_WAIT, FUTEX_WAKE = 0, 1


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


_syscall = _libc.syscall
_syscall.restype = ctypes.c_long


def futex_wait(addr: int, expected: int, timeout_s: Optional[float]) -> int:
    ts = None
    if timeout_s is not None:
        sec = int(timeout_s)
        ts = ctypes.byref(_Timespec(sec, int((timeout_s - sec) * 1e9)))
    return _syscall(ctypes.c_long(_SYS_FUTEX), ctypes.c_void_p(addr), ctypes.c_int(FUTEX_WAIT),
                    ctypes.c_uint(expected), ts, ctypes.c_void_p(0), ctypes.c_uint(0))


# A separate function object with declared argument types: no ctypes objects built per call.
_syscall_wake = _libc["syscall"]
_syscall_wake.restype = ctypes.c_long
_syscall_wake.argtypes = (ctypes.c_long, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_uint)


def futex_wake(addr: int, count: int = 0x7FFFFFFF) -> int:
    return _syscall_wake(_SYS_FUTEX, addr, FUTEX_WAKE, count, None, None, 0)


# Full memory barrier for the Dekker handshake with a worker going to sleep (pn_ipc.h:
# PnWorkerBlock.sleeping): after incrementing workers' `wake` words, a waker may skip their
# FUTEX_WAKE syscalls if, after a full barrier, `sleeping` is 0 (the worker re-checks `wake` after
# announcing its sleep with an xchg). glibc's pthread_spin_trylock is one locked instruction (lock
# cmpxchg, or xchg in the generic version), which on x86 orders the earlier stores before the later
# loads; on a lock word that stays 0 the x86 version always fails, i.e. it never writes. ~0.1 us,
# versus ~1-4 us for a FUTEX_WAKE syscall that finds no sleeper.
# POSIX does not promise a barrier from a failed trylock on a never-initialised word, so this is used
# only where the executed code was checked (glibc on x86-64: an unconditional locked cmpxchg, also on
# the EBUSY path). Elsewhere every wake is a
# FUTEX_WAKE (always correct, only slower).
_spin_trylock = None
if platform.machine() in ("x86_64", "AMD64") and platform.libc_ver()[0] == "glibc":
    try:
        _spin_trylock = _libc["pthread_spin_trylock"]
        _spin_trylock.restype = ctypes.c_int
        _spin_trylock.argtypes = (ctypes.c_void_p,)
        _FENCE_WORD = ctypes.c_int(0)
        _FENCE_ADDR = ctypes.addressof(_FENCE_WORD)
    except AttributeError:  # pragma: no cover (no pthread_spin_trylock: always issue FUTEX_WAKE)
        _spin_trylock = None


def full_barrier() -> bool:
    """A full memory barrier (False if unavailable: the caller must then not skip wake-ups)."""
    if _spin_trylock is None:
        return False
    _spin_trylock(_FENCE_ADDR)
    return True


# ---- worker executable --------------------------------------------------------------------------

_INFO_CACHE: Dict[tuple, dict] = {}


def worker_info(path: str) -> dict:
    """``fireredsim-worker --info`` (cached per path and modification time)."""
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_mtime_ns, st.st_size)
    info = _INFO_CACHE.get(key)
    if info is None:
        out = subprocess.run([path, "--info"], check=True, capture_output=True, text=True).stdout
        info = json.loads(out)
        check_layout(info)
        _INFO_CACHE[key] = info
    return info


def _align(v: int, a: int = PAGE) -> int:
    return (v + a - 1) // a * a


@dataclass
class ShmLayout:
    num_envs: int
    num_workers: int
    obs_size: int
    pixels: bool
    snapshot_slots: int
    snapshot_size: int
    pixel_stride: int = 0  # bytes per env slot of the pixel section (0: PIXEL_BYTES, an RGB frame)
    off_workers: int = 0
    off_cmds: int = 0
    off_results: int = 0
    off_info: int = 0
    off_obs: int = 0
    off_pixels: int = 0
    off_snapshots: int = 0
    snapshot_stride: int = 0
    total_size: int = 0

    def __post_init__(self):
        n, w = self.num_envs, self.num_workers
        off = PAGE
        self.off_workers = off
        off = _align(off + w * WORKER_DTYPE.itemsize)
        self.off_cmds = off
        off = _align(off + n * CMD_DTYPE.itemsize)
        self.off_results = off
        off = _align(off + n * RESULT_DTYPE.itemsize)
        self.off_info = off
        off = _align(off + n * INFO_DTYPE.itemsize)
        self.off_obs = off
        off = _align(off + n * self.obs_size)
        if self.pixels:
            if not self.pixel_stride:
                self.pixel_stride = PIXEL_BYTES
            self.off_pixels = off
            off = _align(off + n * self.pixel_stride)
        else:
            self.pixel_stride = 0
        self.snapshot_stride = _align(self.snapshot_size)
        if self.snapshot_slots:
            self.off_snapshots = off
            off = _align(off + self.snapshot_slots * self.snapshot_stride)
        self.total_size = off


class WorkerSet:
    """The shared memory file plus W worker processes, each hosting a contiguous env range."""

    def __init__(self, worker_path: str, info: dict, num_envs: int, num_workers: int, *,
                 obs_mode: int = 0, pixels: bool = False, pixel_code: int = 1, pixel_stride: int = 0,
                 snapshot_slots: int = 0,
                 templates: Sequence[str] = (), scripts: Sequence[str] = (), spin_us: int = 0,
                 client_spin_us: int = 0,
                 render_hblank: bool = True, bedroom_cache: int = 16, save_cache: int = 16,
                 preset_flags: Sequence[int] = (), preset_vars: Optional[Mapping[int, int]] = None,
                 cpus: Optional[Sequence[int]] = None,
                 log_dir: Optional[str] = None, startup_timeout: float = 60.0, quiet_faults: bool = False,
                 batch_timeout: Optional[float] = 600.0, tree_fanout: int = 4):
        if num_workers < 1 or num_envs < num_workers:
            raise ValueError("need 1 <= num_workers <= num_envs")
        preset_vars = dict(preset_vars or {})
        check_start_preset(info, preset_flags, preset_vars, worker_path)
        self.info = info
        self.num_envs, self.num_workers = num_envs, num_workers
        # pixels: the pixel section exists, one slot of pixel_stride bytes (0: one frame of pixel_code) per env;
        # pixel_code = PnShmHeader.pixels, the format the ops render in unless they name their own.
        if pixels:
            fmt = PIXEL_FORMAT_NAMES.get(int(pixel_code))
            if fmt is None:
                raise ValueError(f"unknown pixel format code {pixel_code}")
            if not pixel_stride:
                pixel_stride = PIXEL_FORMATS[fmt].bytes
            if pixel_stride < PIXEL_FORMATS[fmt].bytes:
                raise ValueError(f"pixel_stride {pixel_stride} < one {fmt} frame ({PIXEL_FORMATS[fmt].bytes} bytes)")
            if (int(pixel_code) != 1 or pixel_stride != PIXEL_BYTES) and not supports_pixel_formats(info):
                raise WorkerError(f"pixel format {fmt} / pixel_stride {pixel_stride}: the worker {worker_path} "
                                  f"renders RGB slots only (no pixel_formats in its --info: rebuild it)")
        self.pixel_code = int(pixel_code) if pixels else 0
        self.layout = L = ShmLayout(num_envs, num_workers, int(info["obs_size"]), bool(pixels),
                                    int(snapshot_slots), int(info["snapshot_size"]),
                                    int(pixel_stride) if pixels else 0)
        self.client_spin_us = int(client_spin_us)
        self.batch_timeout = batch_timeout
        self._closed = False
        self._procs: List[subprocess.Popen] = []
        self._logs: List[str] = []
        self._fd = os.memfd_create("fireredsim", 0)
        os.ftruncate(self._fd, L.total_size)
        self._mm = mmap.mmap(self._fd, L.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        # np.ndarray(buffer=mmap) keeps the mmap object alive but does not keep
        # a buffer export: mmap.close() can unmap observations still held by a
        # caller. frombuffer retains an export through its memoryview, so all
        # derived arrays keep the mapping valid until their last view is gone.
        buf = np.frombuffer(self._mm, dtype=np.uint8)
        self.header = np.ndarray((), HEADER_DTYPE, buf, 0)
        self.workers = np.ndarray((num_workers,), WORKER_DTYPE, buf, L.off_workers)
        self.cmds = np.ndarray((num_envs,), CMD_DTYPE, buf, L.off_cmds)
        self.results = np.ndarray((num_envs,), RESULT_DTYPE, buf, L.off_results)
        self.infos = np.ndarray((num_envs,), INFO_DTYPE, buf, L.off_info)
        self.obs_bytes = np.ndarray((num_envs, L.obs_size), np.uint8, buf, L.off_obs)
        # pixel_slots: every env's slot as raw bytes [N, pixel_stride]; pixels: the slots as frames of the header's
        # format [N, *shape] (strided when the slot is larger than a frame)
        self.pixel_slots = (np.ndarray((num_envs, L.pixel_stride), np.uint8, buf, L.off_pixels)
                            if pixels else None)
        self.pixels = None
        if pixels:
            shape = PIXEL_FORMATS[PIXEL_FORMAT_NAMES[self.pixel_code]].shape
            inner = tuple(int(np.prod(shape[i + 1:], dtype=np.int64)) for i in range(len(shape)))
            self.pixels = np.ndarray((num_envs,) + shape, np.uint8, buf, L.off_pixels,
                                     strides=(L.pixel_stride,) + inner)
        self.slots = (np.ndarray((snapshot_slots, L.snapshot_stride), np.uint8, buf, L.off_snapshots)
                      if snapshot_slots else None)
        # field views used on every batch
        self.c_op = self.cmds["op"]
        self.c_flags = self.cmds["flags"]
        self.c_buttons = self.cmds["buttons"]
        self.c_n = self.cmds["n_frames"]
        self.c_hold = self.cmds["hold_frames"]
        # The same bytes as two 64-bit words per command (little endian): offset 0 = op | flags << 32,
        # offset 8 = buttons | n_frames << 16 | hold_frames << 32 | timer1_count << 48 (one strided
        # write each instead of one per field; PokeVecEnv.step's fast path).
        self.c_w0 = np.ndarray((num_envs,), np.dtype("<u8"), buf, L.off_cmds, (CMD_DTYPE.itemsize,))
        self.c_w8 = np.ndarray((num_envs,), np.dtype("<u8"), buf, L.off_cmds + 8, (CMD_DTYPE.itemsize,))
        # offset 56 = max_frames | advance_frames << 16 (PN_F_AUTO_ADVANCE; constant per env, written
        # only when it changes)
        self.c_w56 = np.ndarray((num_envs,), np.dtype("<u4"), buf, L.off_cmds + 56, (CMD_DTYPE.itemsize,))
        self.r_status = self.results["status"]
        self.r_frame = self.results["frame"]
        self.r_frames_run = self.results["frames_run"]
        self.r_terminal = self.results["terminal"]
        self._w_cmd = self.workers["cmd"]
        self._w_seq = self.workers["seq"]
        self._w_tree_child = self.workers["tree_child"]
        self._w_tree_children = self.workers["tree_children"]
        h = self.header
        h["magic"], h["version"] = SHM_MAGIC, SHM_VERSION
        h["num_envs"], h["num_workers"] = num_envs, num_workers
        h["obs_size"], h["obs_mode"], h["pixels"] = L.obs_size, obs_mode, self.pixel_code
        h["pixel_stride"] = L.pixel_stride
        h["snapshot_slots"], h["snapshot_size"], h["snapshot_stride"] = snapshot_slots, L.snapshot_size, L.snapshot_stride
        h["state_size"] = int(info["state_size"])
        h["flags"] = HDR_RENDER_HBLANK if render_hblank else 0
        for k in ("off_workers", "off_cmds", "off_results", "off_info", "off_obs", "off_pixels", "off_snapshots",
                  "total_size"):
            h[k] = getattr(L, k)
        h["build_id"] = int(info["build_id"], 16)
        h["spin_us"] = int(spin_us)
        h["tree_fanout"] = int(tree_fanout)
        self.spin_us = int(spin_us)
        self.tree_fanout = int(tree_fanout)
        # Partial / asynchronous batches: a wake-up tree over the posted workers (WCMD_TREE_LIST)
        # when the workers sleep between batches (no spin: each FUTEX_WAKE of a sleeping worker
        # costs the waker ~12-17 us under WSL2); spinning workers are woken directly (no syscall).
        F = min(int(tree_fanout), TREE_LIST_MAX)
        self._list_fanout = F if (F > 0 and int(spin_us) == 0 and num_workers <= 0xFFFF
                                  and supports_tree_list(info)) else 0
        if self._list_fanout:
            k = np.arange(num_workers, dtype=np.int64)
            self._tree_idx = F * (k[:, None] + 1) + np.arange(F)[None, :]  # child positions of position k
        begins = (np.arange(num_workers, dtype=np.int64) * num_envs) // num_workers
        ends = (np.arange(1, num_workers + 1, dtype=np.int64) * num_envs) // num_workers
        self.workers["env_begin"], self.workers["env_end"] = begins, ends
        self.env_worker = np.repeat(np.arange(num_workers), ends - begins)
        self.worker_begin, self.worker_end = begins, ends
        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        self._go_addr = self._base_addr + HEADER_DTYPE.fields["go"][1]
        self._pending_addr = self._base_addr + HEADER_DTYPE.fields["pending"][1]
        self._ready_addr = self._base_addr + HEADER_DTYPE.fields["ready"][1]
        self._pending = ctypes.c_uint32.from_address(self._pending_addr)
        self._go = ctypes.c_uint32.from_address(self._go_addr)
        self._ready = ctypes.c_uint32.from_address(self._ready_addr)
        woff = self._base_addr + L.off_workers + WORKER_DTYPE.fields["wake"][1]
        soff = self._base_addr + L.off_workers + WORKER_DTYPE.fields["sleeping"][1]
        self._wake_addr = [woff + w * WORKER_DTYPE.itemsize for w in range(num_workers)]
        self._wake = [ctypes.c_uint32.from_address(a) for a in self._wake_addr]
        self._sleeping = [ctypes.c_uint32.from_address(soff + w * WORKER_DTYPE.itemsize) for w in range(num_workers)]
        F = int(tree_fanout)
        self._roots = list(range(F)) if 0 < F < num_workers else list(range(num_workers))
        self._seq = 0
        self.log_dir = log_dir or tempfile.mkdtemp(prefix="fireredsim-workers-")
        try:
            for w in range(num_workers):
                args = [worker_path, "--shm-fd", str(self._fd), "--worker", str(w),
                        "--bedroom-cache", str(int(bedroom_cache))]
                for t in templates:
                    args += ["--template", str(t)]
                for t in scripts:
                    args += ["--template-script", str(t)]
                if int(save_cache) != 16:  # (the default: also works with workers built before save starts)
                    args += ["--save-cache", str(int(save_cache))]
                for f in preset_flags:  # (none by default: also works with workers built before the preset)
                    args += ["--preset-flag", str(int(f))]
                for v, val in preset_vars.items():
                    args += ["--preset-var", f"{int(v)}={int(val)}"]
                if cpus:
                    args += ["--cpu", str(cpus[w % len(cpus)])]
                if quiet_faults:
                    args += ["--quiet-faults"]
                log = os.path.join(self.log_dir, f"worker{w}.log")
                self._logs.append(log)
                with open(log, "wb") as lf:
                    self._procs.append(subprocess.Popen(args, pass_fds=(self._fd,), stdin=subprocess.DEVNULL,
                                                        stdout=lf, stderr=lf, close_fds=True))
            self._wait_ready(startup_timeout)
        except BaseException:
            self.close(kill=True)
            raise

    # -- process management ------------------------------------------------------------------

    def _tail(self, w: int, lines: int = 20) -> str:
        try:
            with open(self._logs[w], "rb") as f:
                data = f.read().decode(errors="replace").strip().splitlines()
            return "\n".join(data[-lines:])
        except OSError:
            return ""

    def _dead_workers(self) -> List[int]:
        return [w for w, p in enumerate(self._procs) if p.poll() is not None]

    def _raise_dead(self, dead: List[int]) -> None:
        parts = []
        for w in dead:
            p = self._procs[w]
            b, e = int(self.worker_begin[w]), int(self.worker_end[w])
            parts.append(f"worker {w} (pid {p.pid}, envs {b}..{e - 1}) exited with code {p.returncode}; "
                         f"log {self._logs[w]}:\n{self._tail(w)}")
        raise WorkerError("\n".join(parts))

    def _wait_ready(self, timeout: float) -> None:
        t0 = time.monotonic()
        while True:
            r = self._ready.value
            if r >= self.num_workers:
                break
            dead = self._dead_workers()
            if dead:
                self._raise_dead(dead)
            if time.monotonic() - t0 > timeout:
                raise WorkerError(f"workers did not start within {timeout} s (logs in {self.log_dir})")
            futex_wait(self._ready_addr, r, 0.05)
        if int(self.header["failed"]):
            failed = [w for w in range(self.num_workers) if int(self.workers["state"][w]) == WSTATE_FAILED]
            raise WorkerError("worker(s) failed to start:\n" + "\n".join(
                f"worker {w}: {self._tail(w)}" for w in failed))

    @property
    def pids(self) -> List[int]:
        return [p.pid for p in self._procs]

    # -- batches -----------------------------------------------------------------------------------

    def _wake_worker(self, w: int) -> None:
        self._wake_workers((w,))

    def _wake_workers(self, targets) -> None:
        """Wakes workers: increments their ``wake`` words, then (Dekker, see full_barrier) issues
        FUTEX_WAKE only for those that announced a sleep. All increments come first, so spinning
        workers start together instead of one syscall apart."""
        wake = self._wake
        for w in targets:
            v = wake[w]
            v.value = (v.value + 1) & 0xFFFFFFFF
        if full_barrier():
            sleeping = self._sleeping
            for w in targets:
                if sleeping[w].value:
                    futex_wake(self._wake_addr[w], 1)
        else:
            for w in targets:
                futex_wake(self._wake_addr[w], 1)

    def _set_cmds(self, workers: np.ndarray, cmd: int) -> list:
        """Sets cmd and bumps seq of ``workers`` (a partial batch); returns the workers to wake:
        all of them, or the roots of a wake-up tree over them (WCMD_TREE_LIST)."""
        F = self._list_fanout
        n = int(workers.size)
        self._w_seq[workers] += 1
        if F and n > F:
            idx = self._tree_idx[:n]
            self._w_tree_child[workers, :F] = workers[np.minimum(idx, n - 1)]
            self._w_tree_children[workers] = np.clip(n - idx[:, 0], 0, F)
            self._w_cmd[workers] = cmd | WCMD_TREE_LIST
            return workers[:F].tolist()
        self._w_cmd[workers] = cmd
        return workers.tolist()

    def post(self, workers: Optional[np.ndarray] = None, cmd: int = WCMD_RUN) -> bool:
        """Starts the ops currently in ``cmds`` on the given workers (default: all); returns
        False if there is nothing to run. Call :meth:`wait` before touching the shared memory."""
        if self._closed:
            raise RuntimeError("environment is closed")
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        # Every post increments the worker's own sequence number, so seq != done_seq holds for
        # exactly the posted workers (also when synchronous and asynchronous batches alternate).
        if workers is None:
            F = self.tree_fanout
            self._w_cmd[:] = cmd | (WCMD_TREE if 0 < F < self.num_workers else 0)
            self._w_seq += 1
            n = self.num_workers
            targets = self._roots
        else:
            workers = np.asarray(workers)
            if workers.size == 0:
                return False
            targets = self._set_cmds(workers, cmd)
            n = int(workers.size)
        self._pending.value = n
        self._go.value = (self._go.value + 1) & 0xFFFFFFFF
        self._wake_workers(targets)
        return True

    # -- asynchronous batches (send/recv): per-worker sequence numbers, no `pending` --------------

    def enable_async(self) -> None:
        self.header["flags"] = int(self.header["flags"]) | HDR_WAKE_EACH
        self._completions_addr = self._base_addr + HEADER_DTYPE.fields["completions"][1]
        self._completions = ctypes.c_uint32.from_address(self._completions_addr)

    def post_async(self, workers: np.ndarray) -> np.ndarray:
        """Starts the ops in ``cmds`` on ``workers`` (none of them may be busy); returns their
        new sequence numbers (the batch is done when ``workers['done_seq']`` equals them)."""
        if self._closed:
            raise RuntimeError("environment is closed")
        targets = self._set_cmds(workers, WCMD_RUN)
        seqs = self._w_seq[workers]
        self._go.value = (self._go.value + 1) & 0xFFFFFFFF
        self._wake_workers(targets)
        return seqs

    def wait_any(self, workers: np.ndarray, seqs: np.ndarray, at_least: int,
                 spin_us: int = 0, timeout: Optional[float] = None,
                 weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Waits until ``workers`` that finished their batch ``seqs`` weigh at least
        ``at_least`` (weights: default 1 per worker); returns the mask of finished ones."""
        if timeout is None:
            timeout = self.batch_timeout
        done_seq = self.workers["done_seq"]
        t_spin = time.perf_counter() + spin_us * 1e-6
        t0 = time.monotonic()
        while True:
            c = self._completions.value
            done = done_seq[workers] == seqs
            n = int(done.sum()) if weights is None else int(weights[done].sum())
            if n >= at_least:
                return done
            if time.perf_counter() < t_spin:
                continue
            futex_wait(self._completions_addr, c, 0.25)
            dead = self._dead_workers()
            if dead:
                self._closed = True
                self._raise_dead(dead)
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise WorkerError(f"async batch did not finish within {timeout} s (logs in {self.log_dir})")

    def run(self, workers: Optional[np.ndarray] = None, timeout: Optional[float] = None) -> None:
        """Executes the ops currently in ``cmds`` on the given workers (default: all)."""
        if self.post(workers):
            self.wait(timeout)

    def wait(self, timeout: Optional[float] = None) -> None:
        if timeout is None:
            timeout = self.batch_timeout
        pending = self._pending
        if self.client_spin_us:
            t_end = time.perf_counter() + self.client_spin_us * 1e-6
            while pending.value:
                if time.perf_counter() > t_end:
                    break
            else:
                return
        t0 = time.monotonic()
        while True:
            p = pending.value
            if p == 0:
                return
            futex_wait(self._pending_addr, p, 0.25)
            if pending.value == 0:
                return
            dead = self._dead_workers()
            if dead:
                self._closed = True
                self._raise_dead(dead)
            if timeout is not None and time.monotonic() - t0 > timeout:
                late = [w for w in range(self.num_workers)
                        if int(self.workers["done_seq"][w]) != int(self.workers["seq"][w])]
                raise WorkerError(f"batch {self._seq} did not finish within {timeout} s; workers still busy: "
                                  f"{late} (logs in {self.log_dir})")

    def workers_of(self, env_ids: np.ndarray) -> np.ndarray:
        return np.unique(self.env_worker[env_ids])

    def message(self, env: int) -> str:
        return bytes(self.infos["message"][env]).split(b"\0", 1)[0].decode(errors="replace")

    def worker_stats(self) -> np.ndarray:
        return self.workers.copy()

    # -- shutdown ----------------------------------------------------------------------------------

    def close(self, kill: bool = False) -> None:
        if getattr(self, "_closed_done", False):
            return
        self._closed_done = True
        alive = [w for w, p in enumerate(self._procs) if p.poll() is None]
        if alive and not kill and hasattr(self, "header"):
            try:
                self.post(np.array(alive), cmd=WCMD_SHUTDOWN)
                t0 = time.monotonic()
                while self._pending.value and time.monotonic() - t0 < 5.0:
                    futex_wait(self._pending_addr, self._pending.value, 0.1)
                    if self._dead_workers() and len(self._dead_workers()) == len(self._procs):
                        break
            except Exception:
                pass
        for p in self._procs:
            try:
                if kill:
                    p.kill()
                p.wait(timeout=5.0)
            except Exception:
                p.kill()
                p.wait()
        self._closed = True
        # drop every view into the mapping before closing it
        for name in ("header", "workers", "cmds", "results", "infos", "obs_bytes", "pixels", "pixel_slots", "slots",
                     "c_op", "c_flags", "c_buttons", "c_n", "c_hold", "c_w0", "c_w8", "c_w56", "r_status", "r_frame",
                     "r_frames_run", "r_terminal", "_w_cmd", "_w_seq",
                     "_w_tree_child", "_w_tree_children", "_pending", "_go", "_ready", "_wake", "_sleeping",
                     "_completions"):
            if hasattr(self, name):
                setattr(self, name, None)
        try:
            self._mm.close()
        except (BufferError, AttributeError):
            pass  # a caller still holds a view (zero-copy observations); the mapping goes with it
        try:
            os.close(self._fd)
        except OSError:
            pass

    def __del__(self):
        try:
            self.close(kill=False)
        except Exception:
            pass
