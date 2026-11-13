"""The native (C, ctypes) reward, action masks and trail window of the training env.

    make -C rl/pokerl/native            # or: PYTHONPATH=rl:python python3 -m pokerl.native.build

(train.py builds it when missing or stale.)

* ``NativeReward(num_envs)``: ``reset(obs, ids=None)`` at every life start or rewind restore, ``step(obs)`` ->
  ``(reward, terms, progress, champion)`` as fresh arrays, ``trail_window(obs, out)``, ``state_dict`` /
  ``load_state_dict`` (the run-long tile counts), ``carry_capture`` / ``carry_restore`` (what a rewind capture keeps);
  the state arrays (``seen, tile_bm, tile_count, tile_bonus, map_seen, cap, opp_size, best, ...``) are numpy arrays
  owned by this object and updated in place by the C code (assigning a state attribute re-points the C state at the
  new array, after a dtype / shape check).
* ``native_action_mask(obs)``, ``native_fallback_mask(obs)``: the masks of mask.py.

``obs`` is a structured record array of ``fireredsim.obs_dtype.OBS_DTYPE`` (the C code gets the field offsets of that
dtype). The library is loaded on first use; :func:`load` raises ``NativeUnavailable`` when it is missing or was built
against a different record layout / tables (spec hash, :mod:`pokerl.native.spec`).
"""
from __future__ import annotations

import ctypes
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np

from ..mask import NUM_ACTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.environ.get("POKERL_NATIVE_LIB", os.path.join(HERE, "libpokerl_native.so"))

_vp, _i64 = ctypes.c_void_p, ctypes.c_int64


class NativeUnavailable(ImportError):
    pass


# struct pn_tables of pokerl_native.c (spec.tables(): the same names)
TABLE_FIELDS = ("map_index", "flag_matrix", "hm_learn", "scene_cols", "map_w", "map_h", "map_off", "action_buttons",
                "skill_weights", "obj_t", "obj_kind", "obj_end_src", "obj_end_bit", "obj_can_mask", "map_obj_flags",
                "obj_weights", "var_weights", "interact_keys", "interact_weights", "cap_bytes", "opp_table", "opp_size",
                "hm_moves")


class _Tables(ctypes.Structure):
    _fields_ = [(k, _vp) for k in TABLE_FIELDS]


# struct pn_reward_state of pokerl_native.c (after num_envs, every field a pointer to the array of the same name)
STATE_FIELDS = ("seen", "tile_bm", "tile_count", "map_seen", "cap", "opp_size", "best", "cover", "known", "fainted",
                "var_max", "var_last", "key_seen", "skill_step", "skill_next", "maps_visited", "flags_earned", "_scratch",
                "obj_bm", "prev_obj", "prev_t", "trash_key", "interact_bm", "trail_t", "trail_now", "trail_last",
                "trail_epoch", "trail_view", "trail_vkey", "hm_seen", "lv_pid")


class _RewardState(ctypes.Structure):
    _fields_ = [("num_envs", _i64)] + [(k.lstrip("_"), _vp) for k in STATE_FIELDS]


_LIB = None
_TABLES = None   # keeps the table arrays alive
_LAYOUTS: Dict[np.dtype, "_Layout"] = {}


def load():
    """The loaded library (loads, checks the spec hash and installs the tables on first call)."""
    global _LIB, _TABLES
    if _LIB is not None:
        return _LIB
    from . import spec
    if not os.path.exists(LIBRARY):
        raise NativeUnavailable(f"{LIBRARY} not found: build it with `make -C rl/pokerl/native` "
                                f"(or PYTHONPATH=rl:python python3 -m pokerl.native.build)")
    tabs = spec.tables()
    want = spec.spec_hash(tabs)
    # A stale library must be detected BEFORE dlopen: dlopen of the same path returns the already-mapped (old)
    # library for the rest of the process, so a rebuild after a failed load would never be picked up. gen_spec.h is
    # written together with the library and records its hash.
    header = os.path.join(os.path.dirname(LIBRARY), "gen_spec.h")
    if os.path.exists(header):
        with open(header) as f:
            m = re.search(r"#define SPEC_HASH (-?\d+)LL", f.read())
        if m and int(m.group(1)) != want:
            raise NativeUnavailable(f"{LIBRARY} was built for spec hash {int(m.group(1)) & 0xFFFFFFFFFFFFFFFF:#x} "
                                    f"(gen_spec.h), the installed fireredsim.obs_dtype / pokerl tables and reward "
                                    f"weights give {want & 0xFFFFFFFFFFFFFFFF:#x}: rebuild it with "
                                    f"`make -C rl/pokerl/native`")
    lib = ctypes.CDLL(LIBRARY)
    lib.pn_spec_hash.restype = _i64
    lib.pn_spec_hash.argtypes = []
    lib.pn_num_offsets.restype = ctypes.c_int
    have = int(lib.pn_spec_hash())
    if want != have:
        raise NativeUnavailable(f"{LIBRARY} was built for spec hash {have:#x}, the installed fireredsim.obs_dtype / "
                                f"pokerl tables give {want:#x}: rebuild it with `make -C rl/pokerl/native`")
    if lib.pn_num_offsets() != spec.offsets(spec.OBS_DTYPE).size:
        raise NativeUnavailable(f"{LIBRARY}: offset table size mismatch, rebuild it")
    t = _Tables()
    for name in TABLE_FIELDS:
        setattr(t, name, tabs[name].ctypes.data)
    lib.pn_set_tables.argtypes = [ctypes.POINTER(_Tables)]
    lib.pn_set_tables.restype = None
    lib.pn_set_tables(ctypes.byref(t))
    for f in ("pn_action_mask", "pn_fallback_mask"):
        getattr(lib, f).argtypes = [_vp, _i64, _i64, _vp, _vp]
    lib.pn_reward_state_size.restype = _i64
    lib.pn_reward_state_size.argtypes = []
    if lib.pn_reward_state_size() != ctypes.sizeof(_RewardState):
        raise NativeUnavailable(f"{LIBRARY}: reward state layout mismatch, rebuild it")
    lib.pn_reward_reset.argtypes = [ctypes.POINTER(_RewardState), _vp, _i64, _vp, _i64, _vp]
    lib.pn_reward_step.argtypes = [ctypes.POINTER(_RewardState), _vp, _i64, _vp, _vp, _i64, _vp, _vp, _vp, _vp, _vp]
    lib.pn_trail_window.argtypes = [ctypes.POINTER(_RewardState), _vp, _i64, _vp, _vp]
    for f in ("pn_action_mask", "pn_fallback_mask", "pn_reward_reset", "pn_reward_step", "pn_trail_window"):
        getattr(lib, f).restype = ctypes.c_int
    _TABLES = (tabs, t)
    _LIB = lib
    return lib


def available() -> bool:
    try:
        load()
        return True
    except NativeUnavailable:
        return False


class _Layout:
    """Field offsets (``spec.offsets``) and cache-line prefetch offsets of one record dtype."""

    def __init__(self, dtype: np.dtype):
        from . import spec
        from ..reward import FLAG_BYTES, SCENE_BLOCK, SEEN_COLS, SEEN_ROWS
        self.off = spec.offsets(dtype)
        tiles, flags, scene, vars_ = (spec.block_start(self.off, k) for k in ("TILES", "FLAGS", "SCENE", "VARS"))
        self.pf_rew = spec.prefetch_offsets(dtype, spec.REWARD_GROUPS, [(flags, FLAG_BYTES), (scene, SCENE_BLOCK),
                                                                       (tiles, SEEN_ROWS * SEEN_COLS), (vars_, 4)])
        self.p_off = self.off.ctypes.data
        self.p_rew, self.n_rew = self.pf_rew.ctypes.data, self.pf_rew.size


def _layout(dtype: np.dtype) -> _Layout:
    lay = _LAYOUTS.get(dtype)
    if lay is None:
        lay = _LAYOUTS[dtype] = _Layout(dtype)
    return lay


def _records(obs) -> Tuple[np.ndarray, int, int, int]:
    """(keep-alive array, address, n, record size) of structured records (copied only if not contiguous)."""
    if not obs.flags.c_contiguous:
        obs = np.ascontiguousarray(obs)
    return obs, obs.ctypes.data, obs.shape[0], obs.dtype.itemsize


def _addr(buf: np.ndarray) -> int:
    """Address of a fresh contiguous uint8/bool array (faster than ``.ctypes.data``)."""
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _check(rc: int) -> None:
    if rc != 0:
        raise RuntimeError(f"pokerl native call failed ({rc})")


# ---- masks -------------------------------------------------------------------------------------------------

def _mask(fn_name: str, obs) -> np.ndarray:
    lib = load()
    obs, p, n, rs = _records(obs)
    out = np.empty((n, NUM_ACTIONS), dtype=bool)
    _check(getattr(lib, fn_name)(p, n, rs, _layout(obs.dtype).p_off, out.ctypes.data))
    return out


def native_action_mask(obs) -> np.ndarray:
    """bool [N, 8]: the action mask (mask.py)."""
    return _mask("pn_action_mask", obs)


def native_fallback_mask(obs) -> np.ndarray:
    """bool [N, 8]: the mask fallback's mask (mask.py)."""
    return _mask("pn_fallback_mask", obs)


# ---- reward ------------------------------------------------------------------------------------------------

class NativeReward:
    """The reward of ``num_envs`` envs (reward.py has the terms), with the trail of the policy's trail plane.
    ``tile_count``: an existing uint32 [TILE_COUNT] table to share with other instances (e.g. the other env groups of
    the pipelined collector), used in place."""

    from ..reward import TERMS as terms

    def __init__(self, num_envs: int, tile_count: Optional[np.ndarray] = None):
        from .. import reward as R
        self._R = R
        object.__setattr__(self, "_st", _RewardState())
        self._lib = load()
        n = self.num_envs = int(num_envs)
        self._st.num_envs = n
        self._specs = {
            "seen": ((n, R.FLAG_BYTES), np.uint8),
            "tile_bm": ((n, R.TILE_BYTES), np.uint8),
            "tile_count": ((R.TILE_COUNT,), np.uint32),
            "map_seen": ((n, R.MAP_COUNT + 1), np.bool_),
            "cap": ((n,), np.int32),
            "opp_size": ((n,), np.int32),
            "best": ((n,), np.int64),
            "cover": ((n,), np.int32),
            "known": ((n,), np.int32),
            "fainted": ((n,), np.bool_),
            "var_max": ((n, R.SCENE_VAR_IDX.size), np.uint16),
            "var_last": ((n, R.SCENE_BLOCK), np.uint8),
            "key_seen": ((n, R.KEY_BYTES), np.uint8),
            "skill_step": ((1,), np.int64),
            "skill_next": ((n, R.N_SKILL), np.int64),
            "maps_visited": ((n,), np.int64),
            "flags_earned": ((n,), np.int64),
            "_scratch": ((9, n), np.int64),
            "obj_bm": ((n, R.OBJ_BYTES), np.uint8),
            "prev_obj": ((n,), np.int64),
            "prev_t": ((n,), np.int64),
            "trash_key": ((n,), np.uint32),
            "interact_bm": ((n, R.INTERACT_BYTES), np.uint8),
            # the trail (reward.py "Trail"; trail_view / trail_vkey: the C code's incremental trail window)
            "trail_t": ((n, R.TILE_COUNT), np.uint16),
            "trail_now": ((n,), np.uint32),
            "trail_last": ((n,), np.int64),
            "trail_epoch": ((n,), np.uint32),
            "trail_view": ((n, R.SEEN_ROWS * 16), np.uint8),
            "trail_vkey": ((n, 4), np.int64),
            "hm_seen": ((n, 2), np.uint8),  # the HM bits seen this life
            "lv_pid": ((n, 6), np.uint32),  # the levels term's "a mon entered the party" rule
        }
        assert tuple(self._specs) == STATE_FIELDS
        # the move_learn keys (numpy: reward.move_learn_pass)
        object.__setattr__(self, "ml_keys", np.zeros((n, R.ML_KEYS), dtype=np.uint64))
        object.__setattr__(self, "ml_n", np.zeros(n, dtype=np.int32))
        if tile_count is not None and (tile_count.shape != (R.TILE_COUNT,) or tile_count.dtype != np.uint32):
            raise ValueError(f"tile_count must be uint32 [{R.TILE_COUNT}]")
        for k, (shape, dt) in self._specs.items():
            setattr(self, k, tile_count if k == "tile_count" and tile_count is not None else np.zeros(shape, dtype=dt))
        self.trail_now[:] = R.TRAIL_RESET
        self.trail_last[:] = -1
        self.trail_vkey[:] = -1
        self.prev_obj[:] = -1  # (no facing state / tile before the first reset)
        self.prev_t[:] = -1
        self.tile_bonus = np.zeros(n, dtype=np.float32)
        self._win_addr: Dict[str, Tuple[np.ndarray, int]] = {}  # trail window buffer -> its address
        self._pst = ctypes.byref(self._st)
        self._step = self._lib.pn_reward_step
        self._nterms = len(self.terms)
        o1 = 4 * n * self._nterms
        self._cuts = (o1, o1 + 4 * n, o1 + 8 * n, o1 + 9 * n)
        self._nbytes = o1 + 10 * n

    def __setattr__(self, name, value):
        specs = self.__dict__.get("_specs")
        if specs is not None and name in specs:
            shape, dt = specs[name]
            a = np.asarray(value)
            if a.dtype != dt or a.shape != shape or not a.flags.c_contiguous or not a.flags.writeable:
                if a.shape != shape:
                    raise ValueError(f"{name}: shape {a.shape}, need {shape}")
                if name == "tile_count":  # possibly shared: never silently copied
                    raise ValueError(f"{name} must be a C-contiguous writable {np.dtype(dt)} array")
                a = np.array(a, dtype=dt, order="C")  # a private copy the C code may write
            object.__setattr__(self, name, a)
            setattr(self._st, name.lstrip("_"), a.ctypes.data)
            return
        object.__setattr__(self, name, value)

    def reset(self, obs, ids: Optional[np.ndarray] = None) -> None:
        """Every per-life state of envs ``ids`` (default all) from their first records ``obs`` (one per id)."""
        n = self.num_envs
        ids = np.arange(n) if ids is None else np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return
        ids = ids.reshape(-1)
        if ids.min() < -n or ids.max() >= n:
            raise IndexError(f"env ids out of range for {n} envs")
        ids = np.ascontiguousarray(np.where(ids < 0, ids + n, ids))
        obs, p, k, rs = _records(obs)
        if k != ids.size:
            raise ValueError(f"{k} records for {ids.size} env ids")
        _check(self._lib.pn_reward_reset(self._pst, ids.ctypes.data, ids.size, p, rs, _layout(obs.dtype).p_off))
        self.ml_n[ids] = 0

    def trail_window(self, obs, out: np.ndarray) -> np.ndarray:
        """The trail window (uint8 [N, 11, 15], C-contiguous writable ``out``) of the records ``obs`` (one per env, after
        this step's ``step`` and every ``reset`` of the returned rows); returns ``out``."""
        n = self.num_envs
        if obs.shape[0] != n:
            raise ValueError(f"{obs.shape[0]} records for {n} envs")
        a = self._win_addr.get("trail")
        if a is None or a[0] is not out:  # (the env passes the same staging array every step: its address cached)
            R = self._R
            shape = (n, R.SEEN_ROWS, R.SEEN_COLS)
            if out.shape != shape or out.dtype != np.uint8 or not out.flags.c_contiguous or not out.flags.writeable:
                raise ValueError(f"trail window buffer {out.dtype} {out.shape}, need C-contiguous writable uint8 {shape}")
            a = self._win_addr["trail"] = (out, out.ctypes.data)
        obs, p, _, rs = _records(obs)
        _check(self._lib.pn_trail_window(self._pst, p, rs, _layout(obs.dtype).p_off, a[1]))
        return out

    # ---- run-long state (checkpoints) ------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, np.ndarray]:
        """The run-long tile counts as their nonzero entries (index, count), plus the table size and layout."""
        R = self._R
        idx = np.flatnonzero(self.tile_count).astype(np.uint32)
        return {"tile_count_size": np.int64(R.TILE_COUNT), "tile_layout": np.int64(R.TILE_LAYOUT),
                "tile_count_idx": idx, "tile_count_val": self.tile_count[idx].copy()}

    def load_state_dict(self, d: Dict[str, np.ndarray]) -> None:
        R = self._R
        size, layout = int(d["tile_count_size"]), int(d["tile_layout"])
        if size != R.TILE_COUNT or layout != R.TILE_LAYOUT:
            raise ValueError(f"tile count table of {size} tiles (layout {layout:#x}), this build has {R.TILE_COUNT} "
                             f"({R.TILE_LAYOUT:#x}): tables.MAPS changed")
        tc = self.tile_count  # in place: the table may be shared with other instances
        tc[:] = 0
        tc[np.asarray(d["tile_count_idx"], dtype=np.int64)] = np.asarray(d["tile_count_val"], dtype=np.uint32)

    # ---- carry (env.py "Rewind") ------------------------------------------------------------------------------

    def carry_capture(self, ids, paid: bool = True) -> Dict[str, np.ndarray]:
        """Copies of what a rewind capture keeps of envs ``ids`` (after the capture step's ``step``: the state the next
        step of the captured life would start from): the HM bits seen and the move keys paid, with ``paid`` also
        reward.PAID_KEYS (the skill cooldowns as the steps left)."""
        R = self._R
        ids = np.asarray(ids, dtype=np.int64)
        d = {"hm_seen": self.hm_seen[ids].copy(), "ml_keys": self.ml_keys[ids].copy(), "ml_n": self.ml_n[ids].copy()}
        if paid:
            d.update({k: getattr(self, k)[ids].copy() for k in R.PAID_CARRY})
            d[R.PAID_COOLDOWN] = np.maximum(self.skill_next[ids] - self.skill_step[0], 0)
        return d

    def carry_restore(self, ids, d: Dict[str, np.ndarray]) -> None:
        """Puts a capture's state (``carry_capture``) back into envs ``ids``, after their ``reset`` from the restored
        records (in place, so the C state keeps pointing at the arrays)."""
        R = self._R
        ids = np.asarray(ids, dtype=np.int64)
        self.hm_seen[ids] = d["hm_seen"]
        self.ml_keys[ids] = d["ml_keys"]
        self.ml_n[ids] = d["ml_n"]
        for k in R.PAID_CARRY:
            if k in d:
                getattr(self, k)[ids] = d[k]
        if R.PAID_COOLDOWN in d:  # (the remaining steps from this instance's step counter on)
            self.skill_next[ids] = self.skill_step[0] + np.asarray(d[R.PAID_COOLDOWN], dtype=np.int64)

    # ---- step --------------------------------------------------------------------------------------------------

    def step(self, obs):
        """(reward float32 [N], {term: float32 [N]}, progress bool [N], Champion newly beaten bool [N])."""
        n = self.num_envs
        if not obs.flags.c_contiguous:
            obs = np.ascontiguousarray(obs)
        if obs.shape[0] != n:
            raise ValueError(f"{obs.shape[0]} records for {n} envs")
        lay = _layout(obs.dtype)
        # every output in one fresh allocation: out [n, terms] f32 | reward f32 | tile_bonus f32 | progress | champion
        o1, o2, o3, o4 = self._cuts
        buf = np.empty(self._nbytes, dtype=np.uint8)
        b = _addr(buf)
        _check(self._step(self._pst, obs.ctypes.data, obs.dtype.itemsize, lay.p_off, lay.p_rew, lay.n_rew, b, b + o1,
                          b + o3, b + o4, b + o2))
        out = buf[:o1].view(np.float32).reshape(n, self._nterms)
        reward = buf[o1:o2].view(np.float32)
        self.tile_bonus = buf[o2:o3].view(np.float32)
        progress = buf[o3:o4].view(np.bool_)
        champion = buf[o4:].view(np.bool_)
        # 10. move_learn: the numpy pass (reward.move_learn_pass), its rows summed again as the C code sums every row
        rows = self._R.move_learn_pass(self, obs, out)
        if rows is not None and rows.size:
            reward[rows] = self._R.reward_sum(out[rows])
        terms = {t: out[:, j] for j, t in enumerate(self.terms)}
        return reward, terms, progress, champion


__all__ = ["NativeReward", "NativeUnavailable", "available", "load", "native_action_mask", "native_fallback_mask"]
