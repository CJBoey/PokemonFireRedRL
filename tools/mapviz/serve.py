#!/usr/bin/env python3
"""Map visualizer server: serves the page (tools/mapviz/www) and streams a running training's player positions.

The positions come from the trainer's worker shared memory (the memfd of python/fireredsim), mapped
read-only: the trainer is not changed and does no work for the stream. The memory is read only while at
least one page is open and visible (the page reports document.visibilityState). With no visible page the
server samples nothing, and `--linger` seconds later it unmaps the memory and sleeps in the event loop
(0 CPU) until a page becomes visible again.

    python3 tools/mapviz/build_world.py        # once: renders www/world/ from game/data
    python3 tools/mapviz/serve.py              # then open http://localhost:8780/

Training is found as the newest `python -m pokerl.train` process (or `--pid`). Its `fireredsim-worker`
children name the shared memory in `--shm-fd`; each env group (the pipelined collector has two) is one
memfd, read through /proc/<worker>/fd/<fd>. Field offsets come from the trainer's own
fireredsim/obs_dtype.py (found through its PYTHONPATH), else the obs v4/v5 offsets below.

Stdlib + NumPy only: a minimal HTTP/1.1 file server and RFC 6455 WebSocket (/receive) on one asyncio port.
"""
import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
import mimetypes
import mmap
import os
import pickle
import struct
import sys
import time
import urllib.parse
import zipfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WWW = os.path.realpath(os.path.join(HERE, "www"))
SHM_MAGIC, SHM_VERSION, OBS_MAGIC = 0x48534E50, 3, 0x424F4E50  # python/fireredsim/_ipc.py, obs_dtype.py
SHM_HEADER = np.dtype({"names": ["magic", "version", "num_envs", "num_workers", "obs_size", "off_obs"],
                       "formats": ["<u4", "<u4", "<u4", "<u4", "<u4", "<u8"],
                       "offsets": [0, 4, 8, 12, 16, 80], "itemsize": 384})  # _ipc.HEADER_DTYPE (version 3)
FIELDS = {"magic": (0, "<u4"), "in_battle": (36, "u1"), "field_state": (38, "u1"), "map_group": (40, "u1"),
          "map_num": (41, "u1"), "x": (44, "<i2"), "y": (46, "<i2"), "badges": (590, "u1")}  # obs v4 and v5 offsets
HEAT_DIR = os.path.join(HERE, "heat")  # saved dwell tables, one per run name (git-ignored)
HEAT_MAGIC, HEAT_VERSION = 0x54414548, 1  # "HEAT": the binary frame of a dwell table (viewer.js onHeat)
NOV_MAGIC, NOV_VERSION = 0x4C564F4E, 1    # "NOVL": the binary frame of the tile counts (viewer.js onNovelty)
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"  # RFC 6455 section 1.3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- reading a training's shared memory

def _stat(pid):
    """(ppid, start time in clock ticks) of a process."""
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[1]), int(f[19])


def _cmdline(pid):
    return open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")


def find_trainer():
    """The newest `python -m pokerl.train` process."""
    best = None
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            argv = _cmdline(int(p))
            if b"pokerl.train" not in argv or not os.path.basename(argv[0]).startswith(b"python"):
                continue
            start = _stat(int(p))[1]
        except OSError:
            continue
        if best is None or start > best[0]:
            best = (start, int(p))
    if best is None:
        raise LookupError("no training process (python -m pokerl.train) is running")
    return best[1]


def _trainer_fields(pid, obs_size):
    """Field offsets from the trainer's own fireredsim/obs_dtype.py (its PYTHONPATH, relative to its cwd)."""
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
        env = dict(kv.split(b"=", 1) for kv in open(f"/proc/{pid}/environ", "rb").read().split(b"\0") if b"=" in kv)
    except OSError:
        return None, "cannot read the trainer's environment"
    for entry in env.get(b"PYTHONPATH", b"").decode(errors="replace").split(":") + ["python"]:
        path = os.path.join(cwd, entry, "fireredsim", "obs_dtype.py")
        if not entry or not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location(f"_mapviz_obs_dtype_{pid}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # (a generated file: NumPy dtypes only)
        dt = mod.OBS_DTYPE
        if dt.itemsize != obs_size:
            return None, f"{path}: record size {dt.itemsize} != shared memory {obs_size}"
        return {k: (dt.fields[k][1], dt.fields[k][0].str) for k in FIELDS}, path
    return None, "no fireredsim/obs_dtype.py on the trainer's PYTHONPATH"


class Run:
    """Read-only views of one training's env groups."""

    def __init__(self, pid):
        self.pid = pid
        argv = _cmdline(pid)
        self.name = next((argv[argv.index(o) + 1].decode() for o in (b"--run-name", b"--resume") if o in argv),
                         f"pid {pid}")
        shms = {}  # memfd inode -> [first worker pid, path]
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                if _stat(int(p))[0] != pid:
                    continue
                a = _cmdline(int(p))
                if not a[0].endswith(b"fireredsim-worker") or b"--shm-fd" not in a:
                    continue
                path = f"/proc/{p}/fd/{int(a[a.index(b'--shm-fd') + 1])}"
                ino = os.stat(path).st_ino
            except (OSError, ValueError):
                continue
            if ino not in shms or int(p) < shms[ino][0]:
                shms[ino] = [int(p), path]
        if not shms:
            raise LookupError(f"training {self.name} (pid {pid}) has no fireredsim-worker processes yet")
        self.groups = []  # (mmap, records, a worker pid); group order = worker start order (pid)
        self.source = None
        try:
            for wpid, path in sorted(shms.values()):
                fd = os.open(path, os.O_RDONLY)
                try:
                    head = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ)
                    h = np.frombuffer(head, SHM_HEADER, 1)[0].copy()
                    head.close()
                    if h["magic"] != SHM_MAGIC or h["version"] != SHM_VERSION:
                        raise LookupError(f"{path}: not a version-{SHM_VERSION} fireredsim shared memory")
                    n, osz, off = int(h["num_envs"]), int(h["obs_size"]), int(h["off_obs"])
                    mm = mmap.mmap(fd, off + n * osz, mmap.MAP_SHARED, mmap.PROT_READ)
                finally:
                    os.close(fd)  # (the mapping keeps its own reference)
                if self.source is None:
                    fields, self.source = _trainer_fields(pid, osz)
                    self.fields = fields or FIELDS
                    if fields is None:
                        log(f"field offsets: obs v4/v5 defaults ({self.source})")
                dt = np.dtype({"names": list(self.fields), "formats": [f for _, f in self.fields.values()],
                               "offsets": [o for o, _ in self.fields.values()], "itemsize": osz})
                rec = np.frombuffer(mm, dt, n, off)
                self.groups.append((mm, rec, wpid))
                if (rec["magic"] != OBS_MAGIC).mean() > 0.5:
                    raise LookupError(f"{path}: the records are not PnObs (magic) - layout mismatch?")
        except BaseException:
            self.close()
            raise
        self.num_envs = sum(len(r) for _, r, _ in self.groups)

    def alive(self):
        return all(os.path.exists(f"/proc/{w}") for _, _, w in self.groups)

    def sample(self, envs):
        """int32 [len(envs), 5]: x, y, map_group * 256 + map_num, in battle, field_state."""
        out = np.concatenate([np.stack([r["x"], r["y"], r["map_group"].astype(np.int32) * 256 + r["map_num"],
                                        (r["in_battle"] != 0) | (r["field_state"] == 4), r["field_state"]], 1)
                              for _, r, _ in self.groups])
        return out if envs is None else out[envs]

    def extras(self, envs):
        """(badge count, in battle) per env."""
        b = np.concatenate([r["badges"] for _, r, _ in self.groups])
        f = np.concatenate([r["in_battle"] for _, r, _ in self.groups]) != 0
        b = np.unpackbits(b[:, None], axis=1).sum(1)
        return (b, f) if envs is None else (b[envs], f[envs])

    def close(self):
        mms = [mm for mm, _, _ in self.groups]
        self.groups = []  # (drops the record arrays: the mmaps' only buffer exports)
        for mm in mms:
            try:
                mm.close()
            except BufferError:  # a view is still referenced: the mapping goes when it is collected
                pass


class Heat:
    """Dwell table of one run: env samples per tile (map id << 16 | y << 8 | x), split into field and battle time.

    Every position sample the server takes while a page is visible counts, so a tile's count / samples is
    the share of the sampled env time spent on it, and count * seconds / ticks its env-seconds (one sample
    of one env stands for the window's wall time / its samples). Not counted: the warp window
    (field_state 2: the map is already the destination, x/y are not) and Quest Log playback (3). Saved in
    heat/<run name>.npz, so it survives server restarts and continues across `--resume`.
    """

    def __init__(self, run):
        self.run = run
        self.path = os.path.join(HEAT_DIR, "".join(c if c.isalnum() or c in "-_." else "_" for c in run) + ".npz")
        self.clear()
        try:
            with np.load(self.path) as d:
                self.keys, self.counts = d["keys"].astype(np.int64), d["counts"].astype(np.int64)
                self.ticks, self.seconds, self.samples = int(d["ticks"]), float(d["seconds"]), int(d["samples"])
                self.since = float(d["since"])
            log(f"dwell heatmap of {run}: {len(self.keys)} tiles, {self.seconds / 60:.1f} min from {self.path}")
        except FileNotFoundError:
            pass
        except (OSError, KeyError, ValueError) as e:
            log(f"{self.path}: unreadable ({e}); starting a new dwell heatmap")

    def clear(self):
        self.keys = np.zeros(0, np.int64)            # sorted tile keys
        self.counts = np.zeros((0, 2), np.int64)     # [tile, (field, battle)] env samples
        self.ticks, self.seconds, self.samples = 0, 0.0, 0  # samples per env, their wall time, all env samples
        self.since = time.time()
        self.dirty = True

    def add(self, s, seconds):
        """s: int32 [T, envs, 5] of Run.sample over a window of `seconds`."""
        self.ticks += s.shape[0]
        self.seconds += seconds
        s = s.reshape(-1, s.shape[-1])
        self.samples += len(s)
        x, y, m, b, fs = s.T.astype(np.int64)
        ok = (fs != 2) & (fs != 3) & (x >= 0) & (x < 256) & (y >= 0) & (y < 256)
        u, c = np.unique(((m << 16 | y << 8 | x) << 1 | b)[ok], return_counts=True)
        keys, inv = np.unique(np.concatenate([self.keys, u >> 1]), return_inverse=True)
        counts = np.zeros((len(keys), 2), np.int64)
        counts[inv[:len(self.keys)]] = self.counts
        np.add.at(counts, (inv[len(self.keys):], u & 1), c)
        self.keys, self.counts = keys, counts
        self.dirty = True

    def frame(self):
        """The binary WebSocket payload: a 48-byte header, then u32 keys, f32 field and f32 battle counts."""
        head = struct.pack("<IIIIdddd", HEAT_MAGIC, HEAT_VERSION, len(self.keys), 0, self.seconds, self.ticks,
                           self.samples, self.since)
        return (head + self.keys.astype("<u4").tobytes() + self.counts[:, 0].astype("<f4").tobytes() +
                self.counts[:, 1].astype("<f4").tobytes())

    def save(self):
        if not self.dirty:
            return
        os.makedirs(HEAT_DIR, exist_ok=True)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, keys=self.keys, counts=self.counts, ticks=self.ticks, seconds=self.seconds,
                 samples=self.samples, since=self.since)
        os.replace(tmp, self.path)
        self.dirty = False


class _CkptUnpickler(pickle.Unpickler):
    """Unpickles a checkpoint's data.pkl without torch: NumPy arrays / scalars / dtypes and plain containers are
    rebuilt, every other class (torch tensors, storages: the model and the optimizer) becomes a stub, and no tensor
    data is read. Only the whitelisted constructors run (protocol-2 pickles name builtins ``__builtin__``)."""

    class Stub:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return None
        def __setstate__(self, s): pass

    SAFE = {"builtins": {"set", "frozenset", "bytes", "bytearray", "dict", "list", "tuple", "int", "float", "complex",
                         "str", "bool", "slice", "range", "object"},
            "collections": {"OrderedDict", "defaultdict", "deque"},
            "copyreg": {"_reconstructor"}, "_codecs": {"encode"},
            "numpy": {"ndarray", "dtype"},
            "numpy.core.multiarray": {"_reconstruct", "scalar"}, "numpy._core.multiarray": {"_reconstruct", "scalar"}}
    ALIAS = {"__builtin__": "builtins", "copy_reg": "copyreg"}

    def find_class(self, module, name):
        module = self.ALIAS.get(module, module)
        if name in self.SAFE.get(module, ()) or (module == "numpy.dtypes" and name.endswith("DType")):
            return super().find_class(module, name)
        return self.Stub

    def persistent_load(self, pid):
        return None


def read_checkpoint_dict(path):
    """The top-level dict of a pokerl checkpoint (a torch zip; only its data.pkl is read; tensors are stubs)."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
        with z.open(name) as f:
            return _CkptUnpickler(f).load()


def read_checkpoint(path):
    """(update, global_step, reward dict) of a pokerl checkpoint."""
    d = read_checkpoint_dict(path)
    return int(d["update"]), int(d["global_step"]), d.get("reward")


def _trainer_package(pid):
    """Directory of the trainer's pokerl package (its PYTHONPATH, relative to its cwd), or None."""
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
        env = dict(kv.split(b"=", 1) for kv in open(f"/proc/{pid}/environ", "rb").read().split(b"\0") if b"=" in kv)
    except OSError:
        return None
    for entry in [""] + env.get(b"PYTHONPATH", b"").decode(errors="replace").split(":") + ["rl"]:
        d = os.path.join(cwd, entry, "pokerl")
        if os.path.isfile(os.path.join(d, "tables.py")):
            return os.path.normpath(d)
    return None


def trainer_run(pid):
    """(run directory, pokerl package directory) of a trainer process (its --runs-dir, --resume / --run-name)."""
    argv = [a.decode(errors="replace") for a in _cmdline(pid)]
    pkg = _trainer_package(pid)
    if pkg is None:
        raise LookupError("no pokerl package on the trainer's PYTHONPATH")
    cwd = os.readlink(f"/proc/{pid}/cwd")
    runs = argv[argv.index("--runs-dir") + 1] if "--runs-dir" in argv else os.path.join(pkg, "..", "runs")
    name = next((argv[argv.index(o) + 1] for o in ("--resume", "--run-name") if o in argv), None)
    if name is None:
        raise LookupError("the trainer has no --run-name or --resume")
    return os.path.normpath(os.path.join(cwd, runs, name)), pkg


class Novelty:
    """The run-long tile counts of the tile-count bonus (pokerl reward.py, RewardV0.tile_count): per tile, the
    episodes that first-visited it. The trainer keeps them in its own memory only, but every checkpoint saves
    them (reward.state_dict: nonzero indices and counts), so they are read from the run's newest checkpoint,
    polled only while a page is visible. A first visit to a tile with count N pays W / sqrt(max(N + 1, S) / S),
    the share min(1, sqrt(S / (N + 1))) of W (the run's --tile-count-reward and --tile-count-scale).

    The checkpoints seen are kept in memory (at most HISTORY), so the page can show the first visits since
    about `window` seconds ago: the counts of the newest one minus those of the newest one at least that old
    (else the oldest one). The history starts from the checkpoints on disk when the server attaches.
    """
    HISTORY = 400

    def __init__(self, run, window):
        self.run_name, self.window = run.name, window
        self.error = None
        self.snaps = []  # (update, mtime, global_step, idx int64, counts int64), by update
        self.sent_update = None
        self.next_poll = 0.0
        self.dir, pkg = trainer_run(run.pid)
        spec = importlib.util.spec_from_file_location(f"_mapviz_tables_{run.pid}", os.path.join(pkg, "tables.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # (a generated file: data tables only)
        maps = np.array([(k, w, h) for k, _, w, h in mod.MAPS], np.int64)
        self.layout = zlib.crc32(maps.tobytes())  # reward.py TILE_LAYOUT
        self.map_key, self.map_w = maps[:, 0], maps[:, 1]
        self.map_off = np.concatenate([[0], np.cumsum(maps[:, 1] * maps[:, 2])])  # (last: TILE_COUNT)
        cfg = {}
        try:
            cfg = json.load(open(os.path.join(self.dir, "config.json")))
        except (OSError, ValueError):
            pass
        self.scale = float(cfg.get("tile_count_scale") or 1.0)
        w = cfg.get("tile_count_reward")
        self.weight = float("nan") if w is None else float(w)  # None: the reward preset's (not known here)

    def poll(self, now):
        """Read the checkpoints not seen yet (rate-limited); True when a newer one was added."""
        if now < self.next_poll:
            return False
        self.next_poll = now + 5.0
        try:
            files = [f for f in os.listdir(self.dir) if f.startswith("ckpt_") and f.endswith(".pt")]
        except OSError as e:
            self.error = f"{self.dir}: {e.strerror}"
            return False
        seen = {s[0] for s in self.snaps}
        newest = self.snaps[-1][0] if self.snaps else -1
        added = False
        for f in sorted(files):
            try:
                upd = int(f[5:-3])
            except ValueError:
                continue
            if upd in seen or (self.snaps and upd < self.snaps[0][0]):
                continue
            path = os.path.join(self.dir, f)
            try:
                mtime = os.stat(path).st_mtime
                u, step, rw = read_checkpoint(path)
            except (OSError, KeyError, ValueError, EOFError, StopIteration, pickle.UnpicklingError, zipfile.BadZipFile) as e:
                self.error = f"{f}: unreadable ({e})"  # (rotated away while reading: the next poll skips it)
                continue
            if not rw or "tile_count_idx" not in rw:
                self.error = f"{f}: no tile counts (a run without the tile reward?)"
                continue
            if int(rw["tile_layout"]) != self.layout or int(rw["tile_count_size"]) != int(self.map_off[-1]):
                self.error = f"{f}: tile layout {int(rw['tile_layout']):#x} != the trainer's tables.py {self.layout:#x}"
                continue
            if u < newest:  # a resume from an older checkpoint: the history no longer applies
                self.snaps = []
            self.snaps.append((u, mtime, step, rw["tile_count_idx"].astype(np.int64), rw["tile_count_val"].astype(np.int64)))
            self.snaps.sort(key=lambda s: s[0])
            del self.snaps[:-self.HISTORY]
            newest = self.snaps[-1][0]
            self.error = None
            added = True
        return added

    def frame(self):
        """The binary WebSocket payload (viewer.js onNovelty): an 80-byte header, then u32 tile keys
        (map id << 16 | y << 8 | x), u32 counts and u32 first visits since the base checkpoint."""
        u1, t1, s1, idx, cnt = self.snaps[-1]
        base = next((s for s in reversed(self.snaps[:-1]) if s[1] <= t1 - self.window), None) or self.snaps[0]
        u0, t0, s0, bidx, bcnt = base
        prev = np.zeros(len(idx), np.int64)
        pos = np.searchsorted(bidx, idx)
        hit = pos < len(bidx)
        hit[hit] = bidx[pos[hit]] == idx[hit]
        prev[hit] = bcnt[pos[hit]]
        delta = np.maximum(cnt - prev, 0)
        m = np.searchsorted(self.map_off, idx, side="right") - 1
        rel = idx - self.map_off[m]
        w = self.map_w[m]
        keys = self.map_key[m] << 16 | (rel // w) << 8 | rel % w
        head = struct.pack("<IIIIdddddddd", NOV_MAGIC, NOV_VERSION, len(idx), 0, self.scale, self.weight,
                           t1, s1, u1, t0, s0, u0)
        return (head + keys.astype("<u4").tobytes() + cnt.astype("<u4").tobytes() + delta.astype("<u4").tobytes())


PROBE = os.path.join(HERE, "party_probe.py")


class PartyApi:
    """/api/party/snapshot (every env's party; cached 10 s unless ?fresh=1) and /api/party/sample (?secs=N starts a
    behaviour sample when none runs; without it: the sample's state and its last result). Both run party_probe.py as a
    subprocess on request only (the trainer's own record layout and pokerl tables; nothing is read between requests)."""

    def __init__(self, args):
        self.args = args
        self.snap, self.snap_t = None, 0.0
        self.lock = asyncio.Lock()
        self.task, self.result, self.started, self.secs = None, None, None, 0.0

    async def _probe(self, argv, timeout):
        pid = ["--pid", str(self.args.pid)] if self.args.pid else []
        proc = await asyncio.create_subprocess_exec(sys.executable, "-I", PROBE, *argv, *pid,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise LookupError(f"party probe timed out after {timeout:.0f} s")
        if proc.returncode != 0:
            raise LookupError(f"party probe failed: {err.decode(errors='replace')[-600:]}")
        return json.loads(out)

    async def _sample(self, secs, hz):
        try:
            self.result = await self._probe(["sample", "--secs", str(secs), "--hz", str(hz)], secs + 90)
        except (LookupError, OSError, ValueError) as e:
            self.result = {"error": f"{type(e).__name__}: {e}"}

    async def handle(self, path, query):
        q = urllib.parse.parse_qs(query)
        if path == "/api/party/snapshot":
            async with self.lock:
                if self.snap is None or "error" in self.snap or q.get("fresh") or time.monotonic() - self.snap_t > 10:
                    self.snap = await self._probe(["snapshot"], 60)
                    self.snap_t = time.monotonic()
            return self.snap
        if path == "/api/party/sample":
            running = self.task is not None and not self.task.done()
            if "secs" in q and not running:
                self.secs = max(5.0, min(float(q["secs"][0]), 600.0))
                hz = max(1.0, min(float(q.get("hz", ["10"])[0]), 20.0))
                self.started, self.result = time.time(), None
                self.task = asyncio.create_task(self._sample(self.secs, hz))
                running = True
            if running:
                return {"state": "running", "elapsed": time.time() - self.started, "secs": self.secs}
            return {"state": "done" if self.result is not None else "idle", "result": self.result}
        raise LookupError(f"{path}: unknown")


def parse_envs(spec):
    if spec in (None, "", "all"):
        return None
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return np.array(sorted(set(out)), dtype=np.int64)


# ---------------------------------------------------------------- HTTP + WebSocket

def ws_frame(op, payload: bytes) -> bytes:
    n = len(payload)
    if n < 126:
        head = struct.pack(">BB", 0x80 | op, n)
    elif n < 1 << 16:
        head = struct.pack(">BBH", 0x80 | op, 126, n)
    else:
        head = struct.pack(">BBQ", 0x80 | op, 127, n)
    return head + payload


async def ws_read(reader):
    """(opcode, payload) of one client frame (clients mask their frames; ours are small: no fragments)."""
    b0, b1 = await reader.readexactly(2)
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack(">H", await reader.readexactly(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", await reader.readexactly(8))[0]
    if n > 1 << 16:
        raise ValueError("client frame too large")
    mask = await reader.readexactly(4) if b1 & 0x80 else b"\0\0\0\0"
    data = np.frombuffer(await reader.readexactly(n), np.uint8) ^ np.resize(np.frombuffer(mask, np.uint8), n)
    return b0 & 0x0F, data.tobytes()


class Viewer:
    def __init__(self, writer, peer):
        self.writer, self.peer = writer, peer
        self.active = False
        self.heat = False  # the page shows the dwell heatmap: send it the table
        self.nov = False   # the page shows the novelty view: send it the tile counts

    def send(self, text: str):
        self.send_frame(1, text.encode())

    def send_frame(self, op, payload: bytes):
        tr = self.writer.transport
        if tr.is_closing() or tr.get_write_buffer_size() > 16 << 20:  # a stalled page skips batches
            return
        self.writer.write(ws_frame(op, payload))


class Server:
    def __init__(self, args):
        self.args = args
        self.envs = parse_envs(args.envs)
        self.viewers = set()
        self.wake = asyncio.Event()
        self.run = None
        self.error = None
        self.cpu = 0.0  # % of one core over the last batch window
        self.heat = None  # Heat of the attached (or last attached) run
        self.heat_sent = 0.0
        self.heat_saved = time.monotonic()
        self.nov = None   # Novelty of the attached (or last attached) run
        self.nov_error = None
        self.party = PartyApi(args)  # the party page's JSON API (party.html)

    # viewers
    def n_active(self):
        return sum(v.active for v in self.viewers)

    def update_active(self):
        if self.n_active():
            self.wake.set()
        else:
            self.wake.clear()

    def status(self):
        r = self.run
        return {"t": "status", "run": r.name if r else None, "pid": r.pid if r else None,
                "envs": (r.num_envs if self.envs is None else len(self.envs)) if r else 0,
                "hz": self.args.hz, "viewers": self.n_active(), "cpu": round(self.cpu, 2) if self.n_active() else 0.0,
                "error": self.error, "nov_error": (self.nov.error if self.nov else None) or self.nov_error}

    def send_heat(self, viewers=None):
        if self.heat is None:
            return
        targets = [v for v in (self.viewers if viewers is None else viewers) if v.active and v.heat]
        if targets:
            payload = self.heat.frame()
            for v in targets:
                v.send_frame(2, payload)
        self.heat_sent = time.monotonic()

    def nov_viewers(self):
        return [v for v in self.viewers if v.active and v.nov]

    def update_nov(self, force=None):
        """Poll the run's checkpoints while some visible page shows the novelty view; send new counts."""
        targets = self.nov_viewers()
        if self.nov is None or not targets:
            return
        added = self.nov.poll(time.monotonic())
        if not self.nov.snaps:
            return
        for v in (targets if added else (force or [])):
            v.send_frame(2, self.nov.frame())

    def save_heat(self):
        if self.heat is not None:
            try:
                self.heat.save()
            except OSError as e:
                log(f"cannot save the dwell heatmap: {e}")
        self.heat_saved = time.monotonic()

    def broadcast(self, msg):
        text = json.dumps(msg, separators=(",", ":"))
        for v in list(self.viewers):
            if v.active:
                v.send(text)

    # sampling
    def detach(self, why):
        if self.run is not None:
            log(f"detached from {self.run.name} ({why})")
            self.run.close()
            self.run = None
            self.save_heat()

    async def sampler(self):
        loop = asyncio.get_running_loop()
        period = 1.0 / self.args.hz
        while True:
            if not self.wake.is_set():
                if self.run is not None:
                    try:
                        await asyncio.wait_for(self.wake.wait(), self.args.linger)
                    except asyncio.TimeoutError:
                        self.detach("no visible page")
                else:
                    await self.wake.wait()
                continue
            if self.run is None or not self.run.alive():
                if self.run is not None:
                    self.detach("training ended")
                try:
                    pid = self.args.pid or find_trainer()
                    self.run = Run(pid)
                    self.error = None
                    if self.heat is None or self.heat.run != self.run.name:
                        self.save_heat()
                        self.heat = Heat(self.run.name)
                    if self.nov is None or self.nov.run_name != self.run.name:
                        try:
                            self.nov, self.nov_error = Novelty(self.run, self.args.novelty_window), None
                        except (LookupError, OSError, ValueError, AttributeError) as e:
                            self.nov, self.nov_error = None, f"novelty: {e}"
                            log(self.nov_error)
                    log(f"attached to {self.run.name} (pid {pid}): {self.run.num_envs} envs in "
                        f"{len(self.run.groups)} groups; offsets from {self.run.source}")
                except (LookupError, OSError, ValueError) as e:
                    self.error = str(e)
                    self.broadcast(self.status())
                    await asyncio.sleep(3.0)
                    continue
                self.broadcast(self.status())
            # one batch window: sample at --hz, then send every env's path
            cpu0, t0 = time.process_time(), loop.time()
            buf = []
            nxt = t0
            while loop.time() - t0 < self.args.flush and self.wake.is_set():
                buf.append(self.run.sample(self.envs))
                nxt += period
                await asyncio.sleep(max(0.0, nxt - loop.time()))
            if not buf or not self.wake.is_set():
                continue
            dt = loop.time() - t0
            samples = np.stack(buf)  # [T, env, 5]
            self.heat.add(samples, dt)
            paths = samples[:, :, :3].transpose(1, 0, 2).reshape(samples.shape[1], -1)  # [env, T * 3] = x, y, map
            badges, battle = self.run.extras(self.envs)
            ids = np.arange(self.run.num_envs) if self.envs is None else self.envs
            batch = {"t": "batch", "dt": round(dt, 3),
                     "envs": [[int(e), int(b), int(f), p] for e, b, f, p in
                              zip(ids, badges, battle, paths.tolist())]}
            self.broadcast(batch)
            now = time.monotonic()
            if now - self.heat_sent >= self.args.heat_every:
                self.send_heat()
            if now - self.heat_saved >= 60:
                self.save_heat()
            self.update_nov()
            self.cpu = 100.0 * (time.process_time() - cpu0) / max(loop.time() - t0, 1e-6)
            self.broadcast(self.status())

    # connections
    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            lines = head.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            hdr = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
            url = urllib.parse.urlsplit(target)
            path = url.path
            if path == "/receive" and hdr.get("upgrade", "").lower() == "websocket":
                await self.websocket(reader, writer, hdr, peer)
            elif path.startswith("/api/party/"):
                await self.api(writer, method, path, url.query, self.party)
            else:
                await self.static(writer, method, path)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError,
                ValueError):
            pass
        finally:
            writer.close()

    async def static(self, writer, method, path):
        rel = urllib.parse.unquote(path).lstrip("/") or "index.html"
        full = os.path.realpath(os.path.join(WWW, rel))
        if method not in ("GET", "HEAD"):
            code, body, ctype, cache = "405 Method Not Allowed", b"", "text/plain", "no-cache"
        elif not full.startswith(WWW + os.sep) or not os.path.isfile(full):
            hint = b" (run tools/mapviz/build_world.py first)" if rel.startswith("world/") else b""
            code, body, ctype, cache = "404 Not Found", b"not found" + hint, "text/plain", "no-cache"
        else:
            code, body = "200 OK", open(full, "rb").read()
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            # world images are requested with ?v=<build>, so they may be cached
            cache = "max-age=604800" if rel.startswith("world/") and rel.endswith(".png") else "no-cache"
        writer.write((f"HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                      f"Cache-Control: {cache}\r\nConnection: close\r\n\r\n").encode())
        if method != "HEAD":
            writer.write(body)
        await writer.drain()

    async def api(self, writer, method, path, query, handler):
        """The party page's JSON requests (PartyApi); errors are JSON
        {"error": ...} with status 200 so the page can show them."""
        if method != "GET":
            body, code = b'{"error":"GET only"}', "405 Method Not Allowed"
        else:
            try:
                res = await handler.handle(path, query)
                body = json.dumps(res, separators=(",", ":"), allow_nan=False).encode()
            except (LookupError, OSError, ValueError, KeyError, TypeError) as e:
                log(f"{path}: {type(e).__name__}: {e}")
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            code = "200 OK"
        writer.write((f"HTTP/1.1 {code}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                      "Cache-Control: no-store\r\nConnection: close\r\n\r\n").encode() + body)
        await writer.drain()

    async def websocket(self, reader, writer, hdr, peer):
        origin = hdr.get("origin")
        if origin and urllib.parse.urlsplit(origin).netloc != hdr.get("host"):  # only our own page
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        accept = base64.b64encode(hashlib.sha1((hdr["sec-websocket-key"] + WS_GUID).encode()).digest()).decode()
        writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        v = Viewer(writer, peer)
        self.viewers.add(v)
        log(f"page connected {peer}")
        try:
            while True:
                op, data = await ws_read(reader)
                if op == 8:  # close
                    writer.write(ws_frame(8, data[:2]))
                    break
                if op == 9:  # ping
                    writer.write(ws_frame(10, data))
                elif op == 1:
                    try:
                        msg = json.loads(data)
                    except ValueError:
                        continue
                    if "heat" in msg:
                        v.heat = bool(msg["heat"])
                    if "nov" in msg:
                        v.nov = bool(msg["nov"])
                    if msg.get("heat_reset") and self.heat is not None:
                        log(f"dwell heatmap of {self.heat.run} reset by {peer}")
                        self.heat.clear()
                        self.save_heat()
                        self.send_heat()
                    if "active" in msg:
                        v.active = bool(msg["active"])
                        self.update_active()
                        v.send(json.dumps(self.status(), separators=(",", ":")))
                    if v.heat and ("heat" in msg or "active" in msg):
                        self.send_heat([v])
                    if v.nov and ("nov" in msg or "active" in msg):
                        self.update_nov(force=[v])
        finally:
            self.viewers.discard(v)
            self.update_active()
            log(f"page disconnected {peer}; {self.n_active()} visible")


async def amain(args):
    srv = Server(args)
    server = await asyncio.start_server(srv.handle, args.host, args.port)
    if not os.path.exists(os.path.join(WWW, "world", "story", "world.json")):
        log("www/world/ is missing: run tools/mapviz/build_world.py first")
    log(f"serving http://{'localhost' if args.host in ('127.0.0.1', '::1') else args.host}:{args.port}/ "
        f"(sampling at {args.hz} Hz only while a page is visible)")
    try:
        async with server:
            await asyncio.gather(server.serve_forever(), srv.sampler())
    finally:
        srv.save_heat()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to let other machines view)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--pid", type=int, default=None, help="trainer pid (default: the newest pokerl.train)")
    ap.add_argument("--hz", type=float, default=30.0, help="position samples per second while a page is visible")
    ap.add_argument("--flush", type=float, default=1.0, help="seconds of samples per batch sent to the page")
    ap.add_argument("--envs", default="all", help="envs to stream, e.g. 0-63,256-319 (default all)")
    ap.add_argument("--linger", type=float, default=10.0,
                    help="seconds to keep the shared memory mapped after the last page hides or closes")
    ap.add_argument("--heat-every", type=float, default=5.0,
                    help="seconds between dwell heatmap updates to pages that show it")
    ap.add_argument("--novelty-window", type=float, default=600.0,
                    help="novelty view: show the first visits of about the last this many seconds of checkpoints")
    args = ap.parse_args()
    if not 0 < args.hz <= 240 or args.flush <= 0:
        ap.error("--hz must be in (0, 240] and --flush > 0")
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
