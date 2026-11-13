"""Batched environments: ``PokeVecEnv`` (N envs in W worker processes).

Every call is one round trip through shared memory: Python writes the per-env
commands, the workers step their envs (no Python in the inner loop) and write
observations (and pixels, if enabled) straight into NumPy-mapped memory.

Semantics (ARCHITECTURE.md section 4): a step runs ``frameskip`` logical
frames with the action's buttons held for the first ``hold_frames`` of them
(default: all) and returns the observation at the boundary of the last frame.
Frame skip never changes the game: ``step(a)`` with frameskip 24 is exactly 24
single-frame steps with ``a`` held. Pixels are rendered for the final frame of
a step only.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from . import _ipc
from ._ipc import WorkerSet
from ._paths import find_worker
from .actions import action_table, multibinary_to_buttons
from .obs import resolve_obs_dtype
from .seeding import (episode_seeds, next_episode_seed, noop_frames_for_seed, random_seed, rng_stream_seed,
                      timer1_for_seed)
from .starts import SaveStart, ScriptStart

StartSpec = Union[str, os.PathLike, ScriptStart, SaveStart]

RewardFn = Callable[[Any, Any, Dict[str, np.ndarray]], np.ndarray]
TerminationFn = Callable[[Any, Dict[str, np.ndarray]], np.ndarray]

# Envs per worker process. One env per process avoids swapping the 564 KiB state in and out
# of the fixed GBA addresses (2-3x more steps/s); beyond this many
# processes per CPU the envs are packed several per process instead. 16: with many processes
# packing is memory-bandwidth bound (256 envs as 128 workers x 2: 25k steps/s; as 256 workers:
# 259k; 512 envs as 512 workers: 257k). Each
# process costs ~12 MiB plus its start-state cache (bedroom_cache).
MAX_PROCESSES_PER_CPU = 16


_NO_IDS = np.empty(0, dtype=np.int64)


def spin_worker_limit(cpu_count: Optional[int] = None) -> int:
    """Workers that may busy-wait between batches: 3/4 of the CPUs (with every vCPU spinning,
    WSL2 preemption stalls each batch for milliseconds)."""
    return max(1, (3 * (cpu_count or os.cpu_count() or 1)) // 4)


def default_num_workers(num_envs: int, cpu_count: Optional[int] = None) -> int:
    """Default worker processes for ``num_envs`` envs (measured on 32 vCPUs under WSL2):

    - up to 3/4 of the CPUs: one env per worker, busy-waiting (fastest per env);
    - up to twice that: two envs per worker (or one for the remainder), so that the workers still
      fit 3/4 of the CPUs and busy-wait: about one futex-waiting worker per vCPU (25-40 envs on
      32 vCPUs) was the worst configuration, 17-31 % slower than packing two per worker;
    - beyond: one env per worker again, oversubscribed (several processes per CPU; futexes), up
      to MAX_PROCESSES_PER_CPU (16) per CPU, then packed.
    """
    cpus = cpu_count or os.cpu_count() or 1
    n = max(1, int(num_envs))
    spin_cap = spin_worker_limit(cpus)
    if n <= spin_cap:
        return n
    if n <= 2 * spin_cap:
        return (n + 1) // 2
    return max(1, min(n, MAX_PROCESSES_PER_CPU * cpus))


def _as_save_start(start):
    """A path to a flash save (``.sav``, or 128 KiB or more without the snapshot magic) means
    ``SaveStart(path)``."""
    if isinstance(start, (str, os.PathLike)) and not (isinstance(start, str) and start in ("bedroom", "power_on")):
        path = os.fspath(start)
        if path.lower().endswith(".sav"):
            return SaveStart(path)
        try:
            if os.path.getsize(path) >= 128 * 1024:
                with open(path, "rb") as f:
                    if f.read(8) != b"PNSNAP\r\n":
                        return SaveStart(path)
        except OSError:
            pass
    return start


class SimulatorError(RuntimeError):
    """An operation failed inside a worker (bad snapshot, bad template, start script failed...)."""


def _u16_array(v, count: int) -> Optional[np.ndarray]:
    """``v`` as a uint64 [count] when it is an int array of that length with every entry in 0..0xFFFF (a
    per-env ``n_frames`` or ``hold_frames``), else None (a scalar, or something the general path handles)."""
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "iu" or v.shape != (count,):
        return None
    if v.size and (v.min() < 0 or v.max() > 0xFFFF):
        return None
    return v.astype(np.uint64)


class PokeVecEnv:
    """``num_envs`` independent FireRed environments stepped in batches.

    Parameters
    ----------
    num_envs, num_workers:
        environments and worker processes (default: ``default_num_workers``: one
        process per env up to 3/4 of the CPUs, two envs per process up to 3/2 of
        the CPUs, then one process per env again up to 16 x CPUs).
    frameskip:
        logical frames per step (1, 4, 8, 24, 48 ...).
    hold_frames:
        frames of the step during which the buttons are held (default: all);
        e.g. 1 = tap, then release for the rest of the step. ``step`` /
        ``step_async`` / ``send`` also take an int array with one hold per
        (sent) env, and likewise one ``n_frames`` per env, e.g. a longer D-pad
        hold toward a new direction (rl/pokerl/env.py ``dpad_hold``); both stay
        on the command-word fast path.
    obs_mode:
        ``"visible"`` (what a player can see) or ``"privileged"`` (adds the
        ``*_priv`` fields: RNG, exact enemy data, flags, vars ...).
    pixels, pixel_format, render:
        ``pixels``: False (state only), True (state + the last frame of each
        step, reset, restore) or ``"only"`` (pixels only). ``pixel_format``
        (``_ipc.PIXEL_FORMATS``): ``"rgb"``
        (default, uint8 [160, 240, 3]), ``"gray1"`` (luma [160, 240]),
        ``"gray2"`` / ``"gray3"`` / ``"gray4"`` (mean luma of k x k boxes,
        [80, 120] / [54, 80] / [40, 60]) or ``"gray2s"`` / ``"gray3s"`` /
        ``"gray4s"`` (same shapes, only every k-th scanline composed: cheaper);
        the gray formats need a worker that lists them in ``--info``
        ``pixel_formats``. ``render`` (default True) keeps every env's pixel slot
        large enough for an RGB frame, so ``render(fmt="rgb")`` works whatever the
        format (e.g. for videos); ``render=False`` with ``pixels`` makes the slots
        exactly one frame of ``pixel_format``. Attributes: ``pixel_format``,
        ``pixel_shape`` (one frame), ``pixels`` (the mapped uint8 [N, *pixel_shape]
        view of the slots, strided when a slot is larger than a frame; None
        without a pixel section) and ``pixels_mode`` (the ``pixels`` argument).
        With ``pixels`` on, every op that returns an observation renders (step,
        reset, restore / ``restore_from_slots`` / clone, observe), so the view is
        never stale.
    start:
        ``"bedroom"`` (new game: worker/new_game_to_pallet.txt replayed
        from power-on with the episode's timer-1 seeds; ~8 ms per new seed,
        cached per worker), ``"power_on"``, a :class:`~fireredsim.starts.SaveStart`
        (a 128 KiB flash save continued from power-on: title, CONTINUE, Quest Log
        recap skipped, player free in the overworld; cached per save and seeds; a
        path ending in ``.sav`` means the same), a :class:`~fireredsim.starts.ScriptStart`
        (input scripts replayed from power-on in every worker), or the path of a
        snapshot written by this worker executable (``save_snapshot``). Save starts,
        ``"bedroom"`` and ``"power_on"`` can also be chosen per env at reset
        (``options={"start": ...}``).
    seed:
        base seed of the episode seeds (see :mod:`fireredsim.seeding`); None =
        random.
    action_set, action_type:
        see :mod:`fireredsim.actions` (``"discrete"`` over ``action_set``,
        ``"multibinary"`` or raw ``"buttons"``).
    reward_fn(prev_obs, obs, info) -> float[N]:
        user reward (default: 0). ``obs`` are the structured records.
    termination_fn(obs, info) -> bool[N]:
        user termination (default: never).
    max_episode_steps:
        truncation after this many steps (default: none).
    autoreset:
        reset ended envs on the next ``step`` (Gymnasium's next-step mode): the
        action is ignored for them, they return the first observation of the new
        episode with reward 0.
    noop_reset_max:
        after each reset run a seed-dependent number (0..noop_reset_max) of NOOP
        frames (diversity for snapshot starts).
    copy:
        return copies of the observation arrays (default). ``False`` returns
        views of the shared memory, overwritten by the next call (zero copy).
    spin_us, client_spin_us:
        busy-wait of the workers / of this process before sleeping on a futex
        (latency vs. CPU; defaults depend on whether the workers fit the CPUs).
    bedroom_cache, save_cache:
        start states each worker keeps (LRU, 564 KiB each): per script start
        (bedroom included) per timer-1 sequence, and for all save starts
        together per (save, timer-1 sequence).
    preset_flags, preset_vars:
        new-game preset (default none):
        event flag ids (1..0x8FF) to set and ``{var id (0x4000..0x40FF): value}``
        to write in save block 1 of every start replayed from power-on by an
        input script (``"bedroom"`` and :class:`~fireredsim.starts.ScriptStart`),
        right after the replay and before the worker caches the start, so every
        such episode starts with them. Not applied to ``"power_on"``, save starts,
        snapshot starts or ``restore`` (their state carries its own flags). Ids
        are numbers (rl/pokerl resolves ``FLAG_*`` / ``VAR_*`` names). Needs a
        worker that implements it (``preset`` in its ``--info``; WorkerError
        otherwise); without a preset nothing is passed, so older workers start.
    rng_reseed:
        False (default): the game's RNG runs as on hardware (one LCG chain per
        episode from the timer-1 seed, advanced by the agent's own timing). True:
        before every step's first frame the game's PRNG state (``gRngValue`` and
        the wild-encounter RNG) is replaced by fresh values from a per-env host
        entropy stream, so the agent cannot learn to manipulate random outcomes by
        timing its inputs. Game code and probabilities are unchanged; a step of
        ``n`` frames reseeds once, so frameskip changes how often.
        The stream is seeded from the episode seed at every reset (reproducible
        given the seed) and is part of snapshots/clones (a restored env draws the
        same values as the original).
    auto_advance, max_frames, advance_frames:
        decision-point stepping (PN_F_AUTO_ADVANCE, worker/pn_ipc.h). False
        (default): a step is exactly its ``n_frames`` frames. True: after them the
        worker keeps running NOOP chunks of ``advance_frames`` frames (default
        None: the step's own ``n_frames``, i.e. ``frameskip``; 1 = check every
        frame) while the observation would report ``waiting_for_input == 0`` at
        the chunk boundary, until the step's total reaches ``max_frames``
        (default 16 x frameskip) or a terminal / soft reset. The game is
        untouched: such a step equals the requested step followed by NOOP steps
        of ``advance_frames`` frames up to the first waiting boundary (with
        ``rng_reseed`` each chunk reseeds, as each NOOP step would).
        ``info["frames_run"]`` is the step's total; a returned observation is
        non-waiting only at the cap, after a soft reset / terminal, or on reset.
        ``step``/``step_async``/``send`` take per-call overrides of all three.
        Needs a worker that implements it (RuntimeError otherwise).
    snapshot_slots, reserved_slots:
        shared-memory snapshot slots (one snapshot file each, ``snapshot_size``
        bytes in a page-aligned stride). ``snapshot()`` / ``restore()`` /
        ``clone()`` use the first ``snapshot_slots`` (default min(num_envs, 64))
        as scratch space. ``reserved_slots`` (default 0) more follow them, slots
        ``reserved_base .. reserved_base + reserved_slots - 1``, which those
        methods never touch: the caller's own snapshots kept in place
        (``snapshot_to_slots`` / ``restore_from_slots`` / ``slot_bytes``; e.g. the
        RL trainer's rewind captures). Every worker maps all slots; memory is
        committed per slot as it is first written.
    """

    def __init__(self, num_envs: int = 1, num_workers: Optional[int] = None, frameskip: int = 24,
                 hold_frames: Optional[int] = None, obs_mode: str = "visible", pixels: Union[bool, str] = False,
                 start: StartSpec = "bedroom", seed: Optional[int] = None,
                 action_set: Union[str, Sequence] = "basic", action_type: str = "discrete",
                 reward_fn: Optional[RewardFn] = None, termination_fn: Optional[TerminationFn] = None,
                 max_episode_steps: Optional[int] = None, autoreset: bool = True, noop_reset_max: int = 0,
                 terminate_on_soft_reset: bool = False, copy: bool = True,
                 snapshot_slots: Optional[int] = None, render: bool = True, render_hblank: bool = True,
                 worker_path: Optional[str] = None, spin_us: Optional[int] = None,
                 client_spin_us: Optional[int] = None, bedroom_cache: int = 16, save_cache: int = 16,
                 preset_flags: Sequence[int] = (), preset_vars: Optional[Mapping[int, int]] = None,
                 cpus: Optional[Sequence[int]] = None, log_dir: Optional[str] = None, quiet_faults: bool = False,
                 batch_timeout: Optional[float] = 600.0, tree_fanout: int = 4, rng_reseed: bool = False,
                 auto_advance: bool = False, max_frames: Optional[int] = None,
                 advance_frames: Optional[int] = None, reserved_slots: int = 0, pixel_format: str = "rgb"):
        if num_envs < 1:
            raise ValueError("num_envs >= 1")
        if not 1 <= int(frameskip) <= 0xFFFF:
            raise ValueError("frameskip must be 1..65535")
        if obs_mode not in ("visible", "privileged"):
            raise ValueError("obs_mode: 'visible' or 'privileged'")
        if pixels not in (False, True, "only"):
            raise ValueError("pixels: False, True or 'only'")
        if action_type not in ("discrete", "multibinary", "buttons"):
            raise ValueError("action_type: 'discrete', 'multibinary' or 'buttons'")
        self.num_envs = int(num_envs)
        self.num_workers = int(num_workers or default_num_workers(self.num_envs))
        self.frameskip = int(frameskip)
        self.hold_frames = None if hold_frames is None else int(hold_frames)
        self.obs_mode = obs_mode
        self.pixels_mode = pixels
        self.pixels = None  # the mapped pixel view (below)
        self.action_type = action_type
        self.actions = action_table(action_set)
        self.reward_fn, self.termination_fn = reward_fn, termination_fn
        self.max_episode_steps = max_episode_steps
        self.autoreset = autoreset
        self.noop_reset_max = int(noop_reset_max)
        self.terminate_on_soft_reset = terminate_on_soft_reset
        self.copy = copy
        self.worker_path = find_worker(worker_path)
        self.info = _ipc.worker_info(self.worker_path)
        self.obs_dtype, self.obs_kind = resolve_obs_dtype(self.info)
        fmt = _ipc.pixel_format(pixel_format)  # ValueError for an unknown name
        self.pixel_format = pixel_format
        self.pixel_shape = fmt.shape
        if (pixels or render) and not _ipc.supports_pixel_format(self.info, pixel_format):
            raise RuntimeError(f"pixel_format={pixel_format!r}: the worker {self.worker_path} renders RGB only "
                               f"(no pixel_formats in its --info: rebuild it)")
        self.rng_reseed = bool(rng_reseed)
        if self.rng_reseed and not _ipc.supports_rng_reseed(self.info):
            raise RuntimeError(f"rng_reseed=True: the worker {self.worker_path} does not implement it (rebuild it)")
        # new-game preset (validated against the worker's --info before any worker starts)
        self.preset_flags = tuple(int(f) for f in preset_flags)
        self.preset_vars = {int(k): int(v) for k, v in (preset_vars or {}).items()}
        _ipc.check_start_preset(self.info, self.preset_flags, self.preset_vars, self.worker_path)
        self._aa_supported = _ipc.supports_auto_advance(self.info)
        self.auto_advance = bool(auto_advance)
        self.max_frames = min(16 * self.frameskip, 0xFFFF) if max_frames is None else int(max_frames)
        self.advance_frames = 0 if advance_frames is None else int(advance_frames)
        self._aa_word(self.max_frames, self.advance_frames)  # validates
        if self.auto_advance:
            self._check_aa()

        templates: List[str] = []
        scripts: List[str] = []
        self._tmpdir = None
        self._save_links: Dict[str, bytes] = {}
        start = _as_save_start(start)
        if isinstance(start, ScriptStart):
            path = os.path.join(self._make_tmpdir(), "start_script.txt")
            with open(path, "w") as f:
                f.write(start.render())
            scripts.append(path)
            self._template = _ipc.TEMPLATE_SCRIPT0
            self._fixed_timer1 = start.timer1 is not None
        elif isinstance(start, SaveStart):
            self._template = _ipc.TEMPLATE_SAVE
        elif isinstance(start, os.PathLike) or (isinstance(start, str) and start not in ("bedroom", "power_on")):
            path = os.fspath(start)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"start snapshot not found: {path}")
            templates.append(os.path.abspath(path))
            self._template = _ipc.TEMPLATE_FILE0
        elif start == "bedroom":
            self._template = _ipc.TEMPLATE_BEDROOM
        else:
            self._template = _ipc.TEMPLATE_POWER_ON
        self.start = start
        # Per-env start: (template id, .sav path bytes or None, fixed timer-1 values or None);
        # reset(options={"start": ...}) changes it for the reset envs (sticky).
        self._default_start = self._start_spec(start, constructor=True)
        self._env_start = [self._default_start] * self.num_envs

        # Busy-wait only when every worker has a CPU of its own and some CPUs stay free: then a
        # batch round trip costs ~1 us instead of two futex wake-ups (~20 us each under WSL2);
        # with more workers than CPUs spinning would steal their time. The
        # workers spin 1 ms after a batch (covers this process's own ~50 us per step; then they
        # sleep), this process up to 2 ms per batch.
        # At most 3/4 of the CPUs busy-wait (with 30 spinning workers on 32 WSL2 vCPUs batches
        # stalled for milliseconds).
        cpu_count = os.cpu_count() or 1
        avail = len(set(cpus)) if cpus else spin_worker_limit(cpu_count)
        fits = self.num_workers <= avail <= spin_worker_limit(cpu_count)
        if spin_us is None:
            spin_us = 1000 if fits else 0
        if client_spin_us is None:
            client_spin_us = 2000 if fits else 0
        if snapshot_slots is None:
            snapshot_slots = min(self.num_envs, 64)
        if int(reserved_slots) < 0:
            raise ValueError("reserved_slots >= 0")
        # scratch slots of snapshot()/restore(); the reserved ones follow them
        self._scratch_slots = int(snapshot_slots)
        self._reserved_slots = int(reserved_slots)
        need_pixels = bool(pixels) or render
        # One slot per env: a frame of pixel_format (the format of every render unless an op names its own), at
        # least an RGB frame with render=True (render(fmt="rgb")).
        pixel_stride = max(fmt.bytes if pixels else 0, _ipc.PIXEL_BYTES if render else 0)
        self._ws = WorkerSet(self.worker_path, self.info, self.num_envs, self.num_workers,
                             obs_mode=1 if obs_mode == "privileged" else 0, pixels=need_pixels,
                             pixel_code=fmt.code, pixel_stride=pixel_stride,
                             snapshot_slots=self._scratch_slots + self._reserved_slots, templates=templates,
                             scripts=scripts, spin_us=spin_us,
                             client_spin_us=client_spin_us, render_hblank=render_hblank,
                             bedroom_cache=bedroom_cache, save_cache=save_cache,
                             preset_flags=self.preset_flags, preset_vars=self.preset_vars, cpus=cpus, log_dir=log_dir,
                             quiet_faults=quiet_faults,
                             batch_timeout=batch_timeout, tree_fanout=tree_fanout)
        ws = self._ws
        self.snapshot_size = int(self.info["snapshot_size"])
        # Structured views are slow to copy field by field (~100x a memcpy for this dtype):
        # copies always go through the byte view and are re-viewed (_records()).
        self._obs_b = ws.obs_bytes
        self._obs = self._obs_b.view(self.obs_dtype).reshape(self.num_envs)
        self._pix = ws.pixels
        self.pixels = ws.pixels
        self._all = np.arange(self.num_envs)
        self._base_seed = random_seed() if seed is None else int(seed)
        self._seeds = episode_seeds(self._base_seed, self.num_envs)
        self._episode_step = np.zeros(self.num_envs, dtype=np.int64)
        self._needs_reset = np.zeros(self.num_envs, dtype=bool)
        self._started = np.zeros(self.num_envs, dtype=bool)
        self._all_started = False
        self._prev_obs = None
        # rng_reseed: the flag on STEP reseeds the game's PRNG, on RESET sets the stream (rng_seed).
        # pixels on: every op that returns an observation also renders (in the header's format: flags bits 24-31 = 0)
        self._render_flag = _ipc.F_RENDER if pixels else 0
        self._step_flags = _ipc.F_OBS | self._render_flag | (_ipc.F_RNG_RESEED if self.rng_reseed else 0)
        # auto_advance: PnEnvCmd.max_frames | advance_frames << 16 as written into every env's
        # command (None: the envs differ, rewrite before each use)
        self._aa_written = None
        self._pending_step = None
        self._async = None
        self._fast_words = None  # ((n_frames, hold_frames), per-action command words): step fast path

    # ---- spaces --------------------------------------------------------------------------------

    @property
    def single_action_space(self):
        from gymnasium import spaces
        if self.action_type == "discrete":
            return spaces.Discrete(len(self.actions))
        if self.action_type == "multibinary":
            return spaces.MultiBinary(10)
        return spaces.Box(low=0, high=0x3FF, shape=(), dtype=np.uint16)

    # ---- helpers ---------------------------------------------------------------------------------

    def _make_tmpdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="fireredsim-start-")
        return self._tmpdir

    def _save_path_bytes(self, path: str) -> bytes:
        """The .sav path as the worker reads it from the env's message slot (at most 255 bytes; a
        longer path gets a short symlink in this env's temporary directory)."""
        b = self._save_links.get(path)
        if b is None:
            b = os.fsencode(path)
            if len(b) > _ipc.INFO_DTYPE["message"].itemsize - 1 or 0 in b:
                link = os.path.join(self._make_tmpdir(), f"save{len(self._save_links)}.sav")
                os.symlink(path, link)
                b = os.fsencode(link)
            self._save_links[path] = b
        return b

    def _start_spec(self, start, constructor: bool = False):
        """(template id, .sav path bytes, fixed timer-1 values) of a start."""
        start = _as_save_start(start)
        if isinstance(start, SaveStart):
            start.check()
            return (_ipc.TEMPLATE_SAVE, self._save_path_bytes(start.path), start.timer1)
        if isinstance(start, str) and start == "bedroom":
            return (_ipc.TEMPLATE_BEDROOM, None, None)
        if isinstance(start, str) and start == "power_on":
            return (_ipc.TEMPLATE_POWER_ON, None, None)
        if constructor:
            return (self._template, None, None)
        same = start is self.start or start == self.start or (
            isinstance(start, (str, os.PathLike)) and isinstance(self.start, (str, os.PathLike))
            and os.path.abspath(os.fspath(start)) == os.path.abspath(os.fspath(self.start)))
        if same:
            return self._default_start
        raise ValueError("options['start']: a SaveStart (or .sav path), 'bedroom', 'power_on', or the env's "
                         "constructor start (snapshot and script starts are registered with the workers when "
                         "they start)")

    def _ids(self, env_ids) -> np.ndarray:
        if env_ids is None:
            return self._all
        ids = np.atleast_1d(np.asarray(env_ids, dtype=np.int64))
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_envs):
            raise IndexError("env id out of range")
        return ids

    def _buttons(self, actions) -> np.ndarray:
        a = np.asarray(actions)
        if self.action_type == "discrete":
            a = a.reshape(self.num_envs)
            if a.size and (a.min() < 0 or a.max() >= len(self.actions)):
                raise ValueError(f"discrete actions are 0..{len(self.actions) - 1}")
            return self.actions[a.astype(np.int64)]
        if self.action_type == "multibinary":
            return multibinary_to_buttons(a.reshape(self.num_envs, 10))
        a = a.reshape(self.num_envs).astype(np.int64)
        if a.size and (a.min() < 0 or a.max() > 0x3FF):
            raise ValueError("button masks are 10-bit (0..0x3FF)")
        return a.astype(np.uint16)

    def _check_errors(self, ids: np.ndarray) -> None:
        st = self._ws.results["status"] if ids is self._all else self._ws.results["status"][ids]
        err = (st & _ipc.ST_ERROR) != 0
        if not err.any():
            return
        bad = ids[err]
        if bad.size:
            msgs = [f"env {int(e)}: {_ipc.ERRORS.get(int(self._ws.results['error'][e]), '?')}: {self._ws.message(int(e))}"
                    for e in bad[:8]]
            raise SimulatorError("; ".join(msgs))

    def _set_reset_cmds(self, ids: np.ndarray, timer1=None) -> None:
        c = self._ws.cmds
        c["op"][ids] = _ipc.OP_RESET
        c["flags"][ids] = self._step_flags
        t1 = np.zeros((ids.size, _ipc.TIMER1_MAX), dtype=np.uint16)
        count = np.zeros(ids.size, dtype=np.uint16)
        tids = np.zeros(ids.size, dtype=np.uint32)
        noop = np.zeros(ids.size, dtype=np.uint32)
        rng_seeds = np.zeros(ids.size, dtype=np.uint64)
        messages = self._ws.infos["message"]
        for k, e in enumerate(ids):
            tid, save_path, fixed = self._env_start[e]
            tids[k] = tid
            if save_path is not None:
                messages[e] = save_path  # PN_TEMPLATE_SAVE: the worker reads it before writing results
            # Snapshot starts keep their own timer-1 values unless given explicitly (the reads have
            # usually happened already; overwriting would change the state bytes for nothing);
            # script starts with "#! timer1" use theirs.
            from_file = _ipc.TEMPLATE_FILE0 <= tid < _ipc.TEMPLATE_SCRIPT0 or (
                tid >= _ipc.TEMPLATE_SCRIPT0 and getattr(self, "_fixed_timer1", False))
            s = int(self._seeds[e])
            if timer1 is not None:
                v = np.atleast_1d(np.asarray(timer1[k], dtype=np.uint16))
                if not 1 <= v.size <= _ipc.TIMER1_MAX:
                    raise ValueError(f"timer1: 1..{_ipc.TIMER1_MAX} values")
                t1[k, :v.size] = v
                count[k] = v.size
            elif fixed is not None:
                t1[k, :len(fixed)] = fixed
                count[k] = len(fixed)
            elif not from_file:
                t1[k, :2] = timer1_for_seed(s)
                count[k] = 2
            noop[k] = noop_frames_for_seed(s, self.noop_reset_max)
            if self.rng_reseed:
                rng_seeds[k] = rng_stream_seed(s)
        c["template_id"][ids] = tids
        c["rng_seed"][ids] = rng_seeds
        c["timer1"][ids] = t1
        c["timer1_count"][ids] = count
        c["noop_frames"][ids] = noop
        self._episode_step[ids] = 0
        self._needs_reset[ids] = False
        self._started[ids] = True

    def _records(self, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Copies of the observation records (all envs or ``ids``), via a byte copy."""
        b = self._obs_b.copy() if ids is None else self._obs_b[ids]
        return np.ndarray((b.shape[0],), dtype=self.obs_dtype, buffer=b)

    def _obs_out(self, ids: Optional[np.ndarray] = None):
        if ids is None:
            state = self._records() if self.copy else self._obs
        else:
            state = self._records(ids)
        pix = None
        if self.pixels_mode:
            pix = self._pix if ids is None else self._pix[ids]
            if self.copy and ids is None:
                pix = pix.copy()
        if self.pixels_mode == "only":
            return pix
        if self.pixels_mode:
            return {"state": state, "pixels": pix}
        return state

    def _results_info(self, ids: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        r = self._ws.results if ids is None else self._ws.results[ids]
        status = r["status"]
        info = {
            "frame": r["frame"].copy(),
            "frames_run": r["frames_run"].copy(),
            "soft_reset": (status & _ipc.ST_SOFT_RESET) != 0,
            "sim_terminal": (status & _ipc.ST_TERMINAL) != 0,
            "terminal_kind": r["terminal"].copy(),
            "episode_step": (self._episode_step if ids is None else self._episode_step[ids]).copy(),
            "episode_seed": (self._seeds if ids is None else self._seeds[ids]).copy(),
        }
        if info["sim_terminal"].any():
            base = self._all if ids is None else ids
            info["terminal_message"] = np.array(
                [self._ws.message(int(base[i])) if info["sim_terminal"][i] else "" for i in range(len(base))],
                dtype=object)
        return info

    # ---- the RL API --------------------------------------------------------------------------------

    def reset(self, seed: Optional[Union[int, Sequence[int]]] = None, env_ids=None,
              options: Optional[Dict[str, Any]] = None):
        """Resets envs (default: all). ``seed``: base seed (int) or one episode seed per env.
        ``options={"timer1": values}``: explicit timer-1 values (one list per env, or one list
        for all). ``options={"start": S}``: the start of the reset envs from now on (also for their
        autoresets): a :class:`~fireredsim.starts.SaveStart` (or a ``.sav`` path), ``"bedroom"``,
        ``"power_on"`` or the constructor's start; one start for all or a list with one per reset
        env. Returns (obs, info) of the reset envs."""
        self._finish_async()
        ids = self._ids(env_ids)
        if options and "start" in options:
            st = options["start"]
            if isinstance(st, (list, tuple)):
                if len(st) != ids.size:
                    raise ValueError("options['start']: one start, or one per reset env")
                specs = [self._start_spec(x) for x in st]
            else:
                specs = [self._start_spec(st)] * ids.size
            for e, sp in zip(ids, specs):
                self._env_start[int(e)] = sp
        if seed is not None:
            if np.ndim(seed) == 0:
                self._base_seed = int(seed)
                self._seeds[ids] = episode_seeds(self._base_seed, self.num_envs)[ids]
            else:
                s = np.asarray(seed, dtype=np.uint64).reshape(-1)
                if s.size != ids.size:
                    raise ValueError("one seed per reset env")
                self._seeds[ids] = s
        else:
            started = ids[self._started[ids]]
            for e in started:
                self._seeds[e] = next_episode_seed(int(self._seeds[e]))
        timer1 = None
        if options and "timer1" in options:
            t = options["timer1"]
            timer1 = [t] * ids.size if np.ndim(t) == 1 else list(t)
            if len(timer1) != ids.size:
                raise ValueError("options['timer1']: one list of values, or one per reset env")
        self._set_reset_cmds(ids, timer1)
        self._ws.run(self._ws.workers_of(ids))
        self._check_errors(ids)
        self._ws.c_op[ids] = _ipc.OP_NONE
        if self.reward_fn is not None:
            if self._prev_obs is None:
                self._prev_obs = self._records()
            self._prev_obs.view(np.uint8).reshape(self.num_envs, -1)[ids] = self._obs_b[ids]
        obs = self._obs_out(None if ids is self._all else ids)
        return obs, self._results_info(None if ids is self._all else ids)

    def _check_aa(self) -> None:
        if not self._aa_supported:
            raise RuntimeError(f"auto_advance=True: the worker {self.worker_path} does not implement it "
                               f"(PnEnvCmd.max_frames missing in its --info: rebuild it)")

    @staticmethod
    def _aa_word(max_frames, advance_frames) -> int:
        m, a = int(max_frames), int(advance_frames)
        if not 0 <= m <= 0xFFFF:
            raise ValueError("max_frames must be 0..65535")
        if not 0 <= a <= 0xFFFF:
            raise ValueError("advance_frames must be 0..65535 (0: the step's n_frames)")
        return m | (a << 16)

    def _aa_flags(self, ids: Optional[np.ndarray], auto_advance, max_frames, advance_frames) -> int:
        """The STEP flags of this call; with auto_advance writes max_frames / advance_frames into the
        commands of ``ids`` (None: every env) when they differ from what is there."""
        aa = self.auto_advance if auto_advance is None else bool(auto_advance)
        if not aa:
            return self._step_flags
        self._check_aa()
        word = self._aa_word(self.max_frames if max_frames is None else max_frames,
                             self.advance_frames if advance_frames is None else advance_frames)
        if self._aa_written != word:
            if ids is None:
                self._ws.c_w56[:] = word
                self._aa_written = word
            else:
                self._ws.c_w56[ids] = word
                self._aa_written = word if ids.size == self.num_envs else None
        return self._step_flags | _ipc.F_AUTO_ADVANCE

    def _words(self, buttons_u64: np.ndarray, count: int, n, h) -> Optional[np.ndarray]:
        """``buttons_u64`` | n_frames << 16 | hold_frames << 32 for a scalar ``n`` (0..0xFFFF) or a per-env int
        array ``n`` [count] and a scalar ``h`` (>= 0, clamped to 0xFFFF) or a per-env int array ``h`` [count]
        (0..0xFFFF); None when ``n`` or ``h`` is neither (the general path)."""
        nn = _u16_array(n, count)
        if nn is None and not (type(n) is int and 0 <= n <= 0xFFFF):
            return None
        hh = _u16_array(h, count)
        if hh is None and not (type(h) is int and h >= 0):
            return None
        words = buttons_u64 | (nn << np.uint64(16) if nn is not None else np.uint64(n << 16))
        return words | (hh << np.uint64(32) if hh is not None else np.uint64(min(h, 0xFFFF) << 32))

    def _discrete_words(self, actions, count: int, n, h) -> Optional[np.ndarray]:
        """Command words (``_ipc.WorkerSet.c_w8``: buttons | n_frames << 16 | hold_frames << 32,
        timer1_count 0, which STEP ignores) of ``count`` integer discrete actions for scalar or per-env
        (int array [count]) ``n`` / ``h`` (``_words``): the bytes the field-by-field path writes. None: use
        the general path. Scalar ``n`` and ``h``: a cached table of the action set's words."""
        if self.action_type != "discrete":
            return None
        a = np.asarray(actions)
        if a.dtype.kind not in "iu":
            return None
        a = a.reshape(count)
        if a.dtype.kind == "i" and a.size and a.min() < 0:
            raise ValueError(f"discrete actions are 0..{len(self.actions) - 1}")
        if type(n) is int and type(h) is int:
            if not (0 <= n <= 0xFFFF and h >= 0):
                return None
            key = (n, h)
            if self._fast_words is None or self._fast_words[0] != key:
                words = self.actions.astype(np.uint64) | np.uint64((n << 16) | (min(h, 0xFFFF) << 32))
                self._fast_words = (key, words)
            table = self._fast_words[1]
        else:  # per-env n_frames and / or hold_frames: the action's buttons, the rest per env
            key = (None, None)
            if self._fast_words is None or self._fast_words[0] != key:
                self._fast_words = (key, self.actions.astype(np.uint64))
            table = self._fast_words[1]
        try:
            buttons = table.take(a)
        except IndexError:
            raise ValueError(f"discrete actions are 0..{len(self.actions) - 1}") from None
        return buttons if key != (None, None) else self._words(buttons, count, n, h)

    def _step_words(self, actions, n, h) -> Optional[np.ndarray]:
        """Fast path of step_async: the command words of every env (see _discrete_words; other
        action types through _buttons); None: use the general path."""
        if self.action_type == "discrete":
            return self._discrete_words(actions, self.num_envs, n, h)
        return self._words(self._buttons(actions).astype(np.uint64), self.num_envs, n, h)

    def step_async(self, actions, n_frames=None, hold_frames=None, auto_advance=None, max_frames=None,
                   advance_frames=None) -> None:
        """Posts a step of every env; :meth:`step_wait` returns its results. ``auto_advance``,
        ``max_frames``, ``advance_frames``: per-call overrides (None: the constructor's)."""
        self._finish_async()
        self._check_idle()
        flags = self._aa_flags(None, auto_advance, max_frames, advance_frames)
        n = self.frameskip if n_frames is None else n_frames
        h = (self.hold_frames if self.hold_frames is not None else n) if hold_frames is None else hold_frames
        ws = self._ws
        words = self._step_words(actions, n, h)
        buttons = self._buttons(actions) if words is None else None
        if self.autoreset and self._needs_reset.any():
            reset_ids = np.flatnonzero(self._needs_reset)
        else:
            reset_ids = _NO_IDS
        if not self._all_started:
            if not self._started.all():
                raise RuntimeError("call reset() before step()")
            self._all_started = True
        if words is not None:
            ws.c_w0[:] = _ipc.OP_STEP | (flags << 32)
            ws.c_w8[:] = words
        else:
            ws.c_op[:] = _ipc.OP_STEP
            ws.c_flags[:] = flags
            ws.c_buttons[:] = buttons
            ws.c_n[:] = n
            ws.c_hold[:] = np.minimum(h, 0xFFFF)
        if reset_ids.size:
            for e in reset_ids:
                self._seeds[e] = next_episode_seed(int(self._seeds[e]))
            self._set_reset_cmds(reset_ids)
        ws.post()
        self._pending_step = reset_ids

    def step_wait(self):
        if self._pending_step is None:
            raise RuntimeError("step_wait() without step_async()")
        reset_ids = self._pending_step
        self._pending_step = None
        ws = self._ws
        ws.wait()
        # One reduction tells whether any env reported a status bit (error, terminal, soft reset);
        # usually none did and the status-derived arrays are all False.
        any_status = int(np.bitwise_or.reduce(ws.r_status))
        if any_status & _ipc.ST_ERROR:
            self._check_errors(self._all)
        ws.c_op[:] = _ipc.OP_NONE
        self._episode_step += 1
        if reset_ids.size:
            self._episode_step[reset_ids] = 0
        if any_status:
            info = self._results_info()
        else:
            n = self.num_envs
            info = {
                "frame": ws.r_frame.copy(),
                "frames_run": ws.r_frames_run.copy(),
                "soft_reset": np.zeros(n, dtype=bool),
                "sim_terminal": np.zeros(n, dtype=bool),
                "terminal_kind": ws.r_terminal.copy(),
                "episode_step": self._episode_step.copy(),
                "episode_seed": self._seeds.copy(),
            }
        autoreset = np.zeros(self.num_envs, dtype=bool)
        if reset_ids.size:
            autoreset[reset_ids] = True
        info["autoreset"] = autoreset
        obs_records = self._obs
        terminated = np.zeros(self.num_envs, dtype=bool)
        if self.termination_fn is not None:
            terminated |= np.asarray(self.termination_fn(obs_records, info), dtype=bool).reshape(self.num_envs)
        if self.terminate_on_soft_reset:
            terminated |= info["soft_reset"]
        truncated = info["sim_terminal"].copy()
        if self.max_episode_steps is not None:
            truncated |= self._episode_step >= self.max_episode_steps
        if self.reward_fn is not None:
            reward = np.asarray(self.reward_fn(self._prev_obs, obs_records, info), dtype=np.float32).reshape(self.num_envs)
            self._prev_obs.view(np.uint8).reshape(self.num_envs, -1)[:] = self._obs_b
        else:
            reward = np.zeros(self.num_envs, dtype=np.float32)
        if reset_ids.size:
            reward[reset_ids] = 0
            terminated[reset_ids] = False
            truncated[reset_ids] = False
        if self.autoreset:
            self._needs_reset = terminated | truncated
        return self._obs_out(), reward, terminated, truncated, info

    def step(self, actions, n_frames=None, hold_frames=None, auto_advance=None, max_frames=None,
             advance_frames=None):
        """One step of every env. Returns (obs, reward, terminated, truncated, info)."""
        self.step_async(actions, n_frames, hold_frames, auto_advance, max_frames, advance_frames)
        return self.step_wait()

    def _finish_async(self) -> None:
        if self._pending_step is not None:
            self.step_wait()
        self._check_idle()

    # ---- asynchronous stepping (send / recv) ------------------------------------------------------

    def send(self, actions, env_ids=None, n_frames=None, hold_frames=None, auto_advance=None, max_frames=None,
             advance_frames=None) -> None:
        """Starts a step of ``env_ids`` (default: every env not already in flight) without
        waiting; ``actions`` has one entry per sent env. Envs that ended are reset instead
        (next-step autoreset). Results come back with :meth:`recv`, possibly in a different
        order and before the other envs: no batch barrier, so one slow worker does not stall
        the others. Envs of a worker with an op in flight cannot be sent."""
        if self._pending_step is not None:
            self.step_wait()
        ws = self._ws
        if self._async is None:
            ws.enable_async()
            self._async = {"inflight": np.zeros(self.num_envs, dtype=bool),
                           "wbusy": np.zeros(self.num_workers, dtype=bool),
                           "wseq": np.zeros(self.num_workers, dtype=np.uint32),
                           "reset": np.zeros(self.num_envs, dtype=bool)}
        st = self._async
        if env_ids is None:
            ids = np.flatnonzero(~st["inflight"])
        else:
            ids = self._ids(env_ids)
        if ids.size == 0:
            return
        if not self._all_started:
            if not self._started[ids].all():
                raise RuntimeError("call reset() before send()")
            self._all_started = bool(self._started.all())
        if self.num_workers == self.num_envs and (ids.size == 1 or bool((ids[1:] > ids[:-1]).all())):
            workers = ids  # one env per worker, ids increasing: the workers are the envs
        else:
            workers = ws.workers_of(ids)
        if st["wbusy"][workers].any():
            raise RuntimeError("an env of a worker that is still running was sent (recv first)")
        flags = self._aa_flags(ids, auto_advance, max_frames, advance_frames)
        n = self.frameskip if n_frames is None else n_frames
        h = (self.hold_frames if self.hold_frames is not None else n) if hold_frames is None else hold_frames
        words = self._discrete_words(actions, ids.size, n, h)
        if words is not None:
            ws.c_w0[ids] = _ipc.OP_STEP | (flags << 32)
            ws.c_w8[ids] = words
        else:
            a = np.asarray(actions)
            if self.action_type == "discrete":
                a = a.reshape(ids.size)
                if a.size and (a.min() < 0 or a.max() >= len(self.actions)):
                    raise ValueError(f"discrete actions are 0..{len(self.actions) - 1}")
                buttons = self.actions[a.astype(np.int64)]
            elif self.action_type == "multibinary":
                buttons = multibinary_to_buttons(a.reshape(ids.size, 10))
            else:
                buttons = a.reshape(ids.size).astype(np.uint16)
            ws.c_op[ids] = _ipc.OP_STEP
            ws.c_flags[ids] = flags
            ws.c_buttons[ids] = buttons
            ws.c_n[ids] = n
            ws.c_hold[ids] = np.minimum(h, 0xFFFF)
        resets = ids[self._needs_reset[ids]] if self.autoreset and self._needs_reset.any() else _NO_IDS
        st["reset"][ids] = False
        if resets.size:
            for e in resets:
                self._seeds[e] = next_episode_seed(int(self._seeds[e]))
            self._set_reset_cmds(resets)
            st["reset"][resets] = True
        st["wseq"][workers] = ws.post_async(workers)
        st["wbusy"][workers] = True
        st["inflight"][ids] = True

    def recv(self, min_envs: Optional[int] = None, timeout: Optional[float] = None):
        """Results of sent envs: waits until at least ``min_envs`` (default: all in flight) are
        done and returns those (every finished one). Returns (obs, reward, terminated,
        truncated, info) for the finished envs, ``info["env_id"]`` their ids."""
        st = self._async
        if st is None or not st["inflight"].any():
            raise RuntimeError("recv() without send()")
        ws = self._ws
        busy = np.flatnonzero(st["wbusy"])
        one = self.num_workers == self.num_envs
        # one env per worker: the envs in flight are exactly the busy workers (send/recv set and
        # clear both together)
        inflight_envs = busy if one else np.flatnonzero(st["inflight"])
        want = inflight_envs.size if min_envs is None else max(1, min(int(min_envs), inflight_envs.size))
        # envs in flight per busy worker (one env per worker: one each)
        per_worker = None if one else np.bincount(ws.env_worker[inflight_envs], minlength=self.num_workers)[busy]
        done = ws.wait_any(busy, st["wseq"][busy], want, spin_us=ws.client_spin_us, timeout=timeout,
                           weights=per_worker)
        done_workers = busy[done]
        if one:
            ids = done_workers
        else:
            ids = np.flatnonzero(np.isin(ws.env_worker, done_workers) & st["inflight"])
        st["wbusy"][done_workers] = False
        st["inflight"][ids] = False
        ws.c_op[ids] = _ipc.OP_NONE
        any_status = int(np.bitwise_or.reduce(ws.r_status[ids]))
        if any_status & _ipc.ST_ERROR:
            self._check_errors(ids)
        autoreset = st["reset"][ids]
        reset_pos = np.flatnonzero(autoreset)  # (taken before user hooks can see info["autoreset"])
        reset_ids = ids[reset_pos]
        self._episode_step[ids] += 1
        if reset_ids.size:
            self._episode_step[reset_ids] = 0
        if any_status:
            info = self._results_info(ids)
        else:
            info = {
                "frame": ws.r_frame[ids],
                "frames_run": ws.r_frames_run[ids],
                "soft_reset": np.zeros(ids.size, dtype=bool),
                "sim_terminal": np.zeros(ids.size, dtype=bool),
                "terminal_kind": ws.r_terminal[ids],
                "episode_step": self._episode_step[ids],
                "episode_seed": self._seeds[ids],
            }
        info["env_id"] = ids
        info["autoreset"] = autoreset
        records = self._records(ids)
        terminated = np.zeros(ids.size, dtype=bool)
        if self.termination_fn is not None:
            terminated |= np.asarray(self.termination_fn(records, info), dtype=bool).reshape(ids.size)
        if self.terminate_on_soft_reset:
            terminated |= info["soft_reset"]
        truncated = info["sim_terminal"].copy() if any_status else np.zeros(ids.size, dtype=bool)
        if self.max_episode_steps is not None:
            truncated |= self._episode_step[ids] >= self.max_episode_steps
        if self.reward_fn is not None:
            if self._prev_obs is None:
                self._prev_obs = self._records()
            pb = self._prev_obs.view(np.uint8).reshape(self.num_envs, -1)[ids]
            prev = np.ndarray((ids.size,), dtype=self.obs_dtype, buffer=pb)
            reward = np.asarray(self.reward_fn(prev, records, info), dtype=np.float32).reshape(ids.size)
            self._prev_obs.view(np.uint8).reshape(self.num_envs, -1)[ids] = self._obs_b[ids]
        else:
            reward = np.zeros(ids.size, dtype=np.float32)
        if reset_pos.size:
            reward[reset_pos] = 0
            terminated[reset_pos] = False
            truncated[reset_pos] = False
        if self.autoreset:
            self._needs_reset[ids] = terminated | truncated
        pix = self._pix[ids] if self.pixels_mode else None
        if self.pixels_mode == "only":
            obs = pix
        elif self.pixels_mode:
            obs = {"state": records, "pixels": pix}
        else:
            obs = records
        return obs, reward, terminated, truncated, info

    def _check_idle(self) -> None:
        if self._async is not None and self._async["inflight"].any():
            raise RuntimeError("envs are in flight (send() without recv()): call recv() first")

    # ---- observations, pixels, hashes -----------------------------------------------------------

    def observe(self, env_ids=None):
        """Current observation records (fresh) of envs (with ``pixels`` on, their pixel slots are rendered too)."""
        ids = self._ids(env_ids)
        self._finish_async()
        self._run_op(ids, _ipc.OP_OBSERVE, _ipc.F_OBS | self._render_flag)
        return self._records(ids)

    def render(self, env_ids=None, fmt: Optional[str] = None) -> np.ndarray:
        """The last frame of envs, rendered on demand: uint8 [k, *shape] of pixel format ``fmt`` (None: the
        configured ``pixel_format``; e.g. ``"rgb"`` [k, 160, 240, 3] for a video of an env observing a gray
        format, which needs ``render=True``, i.e. RGB-sized slots). A copy. The frame is rendered into the envs'
        pixel slots; with ``pixels`` on and another format, they are rendered again in ``pixel_format`` after
        (the ``pixels`` view stays valid)."""
        if self._ws is None or self._ws.pixel_slots is None:
            raise RuntimeError("rendering is disabled (render=False and pixels=False)")
        name = self.pixel_format if fmt is None else fmt
        f = _ipc.pixel_format(name)
        stride = self._ws.layout.pixel_stride
        if f.bytes > stride:
            raise ValueError(f"render(fmt={name!r}): a frame is {f.bytes} bytes, the pixel slots {stride} "
                             f"(construct the env with render=True for RGB-sized slots)")
        if not _ipc.supports_pixel_format(self.info, name):
            raise RuntimeError(f"render(fmt={name!r}): the worker {self.worker_path} renders RGB only")
        ids = self._ids(env_ids)
        self._finish_async()
        if f.code == self._ws.pixel_code:
            self._run_op(ids, _ipc.OP_OBSERVE, _ipc.F_RENDER)
            return self._pix[ids].copy()
        self._run_op(ids, _ipc.OP_OBSERVE, _ipc.F_RENDER | (f.code << _ipc.F_FORMAT_SHIFT))
        out = self._ws.pixel_slots[ids, :f.bytes].reshape((ids.size,) + f.shape)  # (a copy: fancy indexing)
        if self.pixels_mode:
            self._run_op(ids, _ipc.OP_OBSERVE, _ipc.F_RENDER)  # the configured format back into the slots
        return out

    def state_hash(self, env_ids=None) -> np.ndarray:
        """xxh64 of each env's state (determinism checks; per worker executable)."""
        ids = self._ids(env_ids)
        self._finish_async()
        self._run_op(ids, _ipc.OP_OBSERVE, _ipc.F_HASH)
        return self._ws.results["state_hash"][ids].copy()

    def frames(self, env_ids=None) -> np.ndarray:
        ids = self._ids(env_ids)
        return self._ws.results["frame"][ids].copy()

    def _run_op(self, ids: np.ndarray, op: int, flags: int, slots: Optional[np.ndarray] = None) -> None:
        ws = self._ws
        ws.c_op[ids] = op
        ws.c_flags[ids] = flags
        if slots is not None:
            ws.cmds["slot"][ids] = slots
        ws.run(ws.workers_of(ids))
        ws.c_op[ids] = _ipc.OP_NONE
        self._check_errors(ids)

    # ---- snapshots --------------------------------------------------------------------------------

    @property
    def snapshot_slots(self) -> int:
        """Scratch slots of ``snapshot()`` / ``restore()`` (the constructor's ``snapshot_slots``)."""
        return int(self._ws.layout.snapshot_slots) - self._reserved_slots

    @property
    def reserved_slots(self) -> int:
        """Slots kept for the caller (``reserved_slots``), from ``reserved_base`` on."""
        return self._reserved_slots

    @property
    def reserved_base(self) -> int:
        """Index of the first reserved slot (= ``snapshot_slots``)."""
        return self.snapshot_slots

    @property
    def slot_bytes(self) -> np.ndarray:
        """All snapshot slots as a writable uint8 [slots, stride] view of the shared memory (a slot's snapshot
        file is its first ``snapshot_size`` bytes). Valid until ``close()``; do not write a slot while an op
        that reads or writes it runs."""
        if self._ws.slots is None:
            raise RuntimeError("snapshot_slots=0")
        return self._ws.slots

    def _slot_ids(self, ids: np.ndarray, slots) -> np.ndarray:
        sl = np.atleast_1d(np.asarray(slots, dtype=np.int64))
        if sl.shape != ids.shape:
            raise ValueError("one slot per env")
        total = int(self._ws.layout.snapshot_slots)
        if sl.size and (sl.min() < 0 or sl.max() >= total):
            raise IndexError(f"snapshot slot out of range (0..{total - 1})")
        if sl.size > 1:
            if np.unique(ids).size != ids.size:
                raise ValueError("env_ids must be distinct")
            if np.unique(sl).size != sl.size:
                raise ValueError("slots must be distinct")
        return sl

    def snapshot_to_slots(self, env_ids, slots, fast: bool = False) -> None:
        """Snapshots envs ``env_ids`` into the given slots (one per env, distinct; any slot, reserved ones
        included) in one batch and leaves the files there (no copy into Python; ``restore_from_slots``,
        ``slot_bytes``). ``fast`` as in ``snapshot``."""
        if self._ws.slots is None:
            raise RuntimeError("snapshot_slots=0")
        ids = self._ids(env_ids)
        sl = self._slot_ids(ids, slots)
        self._finish_async()
        if ids.size:
            self._run_op(ids, _ipc.OP_SNAPSHOT, _ipc.F_FAST if fast else 0, sl)

    def restore_from_slots(self, env_ids, slots):
        """Restores the snapshot files in ``slots`` into envs ``env_ids`` (one per env, distinct) in one batch;
        like ``restore`` (a file whose state hash is 0, i.e. a fast snapshot, restores without verification)
        but the files are read where they are. Returns the envs' observations."""
        if self._ws.slots is None:
            raise RuntimeError("snapshot_slots=0")
        ids = self._ids(env_ids)
        sl = self._slot_ids(ids, slots)
        self._finish_async()
        if ids.size:
            ws = self._ws
            fast = ~ws.slots[sl, 32:40].any(axis=1)  # header bytes 32-39: the state hash (0 = fast)
            ws.cmds["flags"][ids] = np.where(fast, _ipc.F_FAST, 0) | _ipc.F_OBS | self._render_flag
            ws.c_op[ids] = _ipc.OP_RESTORE
            ws.cmds["slot"][ids] = sl
            ws.run(ws.workers_of(ids))
            ws.c_op[ids] = _ipc.OP_NONE
            self._check_errors(ids)
        self._needs_reset[ids] = False
        self._started[ids] = True
        if self.reward_fn is not None and self._prev_obs is not None:
            self._prev_obs.view(np.uint8).reshape(self.num_envs, -1)[ids] = self._obs_b[ids]
        return self._records(ids)

    def _chunks(self, ids: np.ndarray):
        if self.snapshot_slots == 0 and self._reserved_slots:
            raise RuntimeError("snapshot_slots=0: no scratch slots (the reserved ones are the caller's)")
        k = max(1, self.snapshot_slots)
        for i in range(0, ids.size, k):
            yield ids[i:i + k]

    def snapshot(self, env_ids=None, fast: bool = False) -> List[bytes]:
        """Snapshots (fireredsim snapshot files, format v1) of envs. ``fast`` skips the
        state hash (in-memory use only; such a snapshot restores without verification)."""
        if self.snapshot_slots == 0:
            raise RuntimeError("snapshot_slots=0")
        ids = self._ids(env_ids)
        self._finish_async()
        uniq, inverse = np.unique(ids, return_inverse=True)
        blobs: List[bytes] = []
        for chunk in self._chunks(uniq):
            slots = np.arange(chunk.size)
            self._run_op(chunk, _ipc.OP_SNAPSHOT, _ipc.F_FAST if fast else 0, slots)
            for s in slots:
                blobs.append(self._ws.slots[s, :self.snapshot_size].tobytes())
        return [blobs[i] for i in inverse.reshape(-1)]

    def restore(self, snapshots: Sequence[bytes], env_ids=None):
        """Restores snapshots into envs (one per env). Returns the envs' observations."""
        ids = self._ids(env_ids)
        if len(snapshots) != ids.size:
            raise ValueError("one snapshot per env")
        if np.unique(ids).size != ids.size:
            raise ValueError("env_ids must be distinct")
        self._finish_async()
        k = 0
        for chunk in self._chunks(ids):
            slots = np.arange(chunk.size)
            fast = np.zeros(chunk.size, dtype=bool)
            for j, s in enumerate(slots):
                blob = snapshots[k + j]
                if len(blob) != self.snapshot_size:
                    raise SimulatorError(f"snapshot of {len(blob)} bytes, this worker's are {self.snapshot_size}")
                self._ws.slots[s, :self.snapshot_size] = np.frombuffer(blob, dtype=np.uint8)
                fast[j] = int.from_bytes(blob[32:40], "little") == 0
            ws = self._ws
            ws.cmds["flags"][chunk] = np.where(fast, _ipc.F_FAST, 0) | _ipc.F_OBS | self._render_flag
            ws.c_op[chunk] = _ipc.OP_RESTORE
            ws.cmds["slot"][chunk] = slots
            ws.run(ws.workers_of(chunk))
            ws.c_op[chunk] = _ipc.OP_NONE
            self._check_errors(chunk)
            k += chunk.size
        self._needs_reset[ids] = False
        self._started[ids] = True
        if self.reward_fn is not None and self._prev_obs is not None:
            self._prev_obs.view(np.uint8).reshape(self.num_envs, -1)[ids] = self._obs_b[ids]
        return self._records(ids)

    def clone(self, src_ids, dst_ids):
        """Copies the state of envs ``src_ids`` into envs ``dst_ids`` (any workers)."""
        src, dst = self._ids(src_ids), self._ids(dst_ids)
        if src.size != dst.size:
            raise ValueError("src_ids and dst_ids differ in length")
        snaps = self.snapshot(src, fast=True)
        obs = self.restore(snaps, dst)
        self._seeds[dst] = self._seeds[src]
        self._episode_step[dst] = self._episode_step[src]
        starts = [self._env_start[e] for e in src.tolist()]
        for e, st in zip(dst.tolist(), starts):
            self._env_start[e] = st  # the clone's later resets use the source's start
        return obs

    def save_snapshot(self, env_id: int, path: Union[str, os.PathLike]) -> None:
        """Writes env ``env_id``'s snapshot to a file (loadable by restore, start=PATH, fireredsim-cli
        built from the same sources only if it is the same executable: snapshots carry a build id)."""
        blob = self.snapshot([env_id])[0]
        with open(path, "wb") as f:
            f.write(blob)

    def load_snapshot(self, env_id: int, path: Union[str, os.PathLike]):
        with open(path, "rb") as f:
            blob = f.read()
        return self.restore([blob], [env_id])[0]

    # ---- misc ---------------------------------------------------------------------------------------

    @property
    def worker_pids(self) -> List[int]:
        return self._ws.pids

    def worker_stats(self) -> np.ndarray:
        """Per-worker counters (busy_ns, swap_ns, frames, batches)."""
        return self._ws.worker_stats()

    def close(self) -> None:
        ws = getattr(self, "_ws", None)
        if ws is not None:
            self._obs = None
            self._obs_b = None
            self._pix = None
            self.pixels = None
            self._prev_obs = None
            ws.close()
            self._ws = None
        if getattr(self, "_tmpdir", None):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
