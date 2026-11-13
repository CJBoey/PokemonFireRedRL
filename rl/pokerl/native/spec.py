"""What the native library is compiled against: record field groups, constants, tables and their hash.

Both the build (``build.py`` writes ``gen_spec.h`` from :func:`header`) and the loader (``__init__.py``)
use this module, so the C code never hardcodes a record offset or a table: the offsets of every field
group are computed at run time from the record dtype (:func:`offsets`), the tables are the numpy arrays
of reward.py / mask.py passed by pointer (:func:`tables`), and only sizes and scalar constants are compiled
in. :func:`spec_hash` covers the dtype layout, the group list, the compiled constants and the source tables;
the library returns the hash it was built with and the loader refuses a mismatch.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from fireredsim.obs_dtype import OBS_DTYPE

from .. import mask as M
from .. import reward as R
from .. import tables as T
from ..records import byte_range, leaf_offsets

U1, U2, I2, U4 = "u1", "u2", "i2", "u4"

# (group name, scalar type, [record paths]); a group's byte offsets are the concatenated leaf offsets
GROUPS: List[tuple] = [
    # reward
    ("VALID", U1, [("valid",)]),
    ("MAP_GROUP", U1, [("map_group",)]),
    ("MAP_NUM", U1, [("map_num",)]),
    ("FIELD_STATE", U1, [("field_state",)]),
    ("XY", I2, [("x",), ("y",)]),
    ("KEY_ID", U2, [("key_items", "id")]),  # the key-items pocket's item ids
    ("SKILL_EVENTS", U4, [("skill_events_priv",)]),  # the worker's skill event bits
    ("P_SPECIES", U2, [("party", "species")]),
    ("P_IS_EGG", U1, [("party", "is_egg")]),
    ("P_LEVEL", U1, [("party", "level")]),
    ("P_HP", U2, [("party", "hp")]),
    ("P_PID", U4, [("party", "pid_priv")]),  # the levels term's "a mon entered the party" rule
    # the puzzle terms' facing / interaction state
    ("FACING", U1, [("facing",)]),
    ("IN_DIALOG", U1, [("in_dialog",)]),
    ("SCREEN", U1, [("screen",)]),
    # the step's overworld interaction
    ("INTERACT_KIND", U1, [("interact_kind_priv",)]),
    ("INTERACT_MAP", U1, [("interact_map_group_priv",), ("interact_map_num_priv",)]),
    ("INTERACT_LOCAL_ID", U1, [("interact_local_id_priv",)]),
    ("INTERACT_XY", I2, [("interact_x_priv",), ("interact_y_priv",)]),
    # mask (and the puzzle terms: waiting, wait_ctx)
    ("WAITING", U1, [("waiting_for_input",)]),
    ("INPUT_KEYS", U2, [("input_keys",)]),
    ("WAIT_CTX", U1, [("wait_ctx",)]),
    ("START_CURSOR", U1, [("start_menu_cursor",)]),
    ("MENU_ACTION", U1, [("menu_action",)]),
]
# single byte offsets of top-level byte blocks; VARS = the start of vars_priv (the trash cans read VAR_TEMP_0 |
# VAR_TEMP_1 << 16 there), SCENE = the first scene variable's
BLOCKS = ["TILES", "FLAGS", "SCENE", "VARS"]
# Per-mon leaves as rows (base, stride over the 6 party slots); packed inner arrays (moves[4]: one 8-byte load)
ROWS = [("P_MOVES", ("party", "moves"))]


def _strided(dtype: np.dtype, path) -> Tuple[int, int, int, int, int]:
    """(base, stride_out, stride_in, n_out, n_in) of a 1-D or 2-D leaf, verified."""
    offs, _ = leaf_offsets(dtype, path)
    offs = np.asarray(offs)
    if offs.ndim == 0:
        offs = offs.reshape(1, 1)
    if offs.ndim == 1:
        offs = offs[:, None]
    if offs.ndim != 2:
        raise ValueError(f"{path}: {offs.ndim}-D leaf")
    n_out, n_in = offs.shape
    base = int(offs[0, 0])
    so = int(offs[1, 0] - offs[0, 0]) if n_out > 1 else 0
    si = int(offs[0, 1] - offs[0, 0]) if n_in > 1 else 0
    want = base + np.arange(n_out)[:, None] * so + np.arange(n_in)[None, :] * si
    if not np.array_equal(offs, want):
        raise ValueError(f"{path}: offsets are not strided")
    return base, so, si, n_out, n_in


def _rows(dtype: np.dtype) -> List[int]:
    out: List[int] = []
    for name, path in ROWS:
        base, so, si, n_out, n_in = _strided(dtype, path)
        _, dt = leaf_offsets(dtype, path)
        if si != dt.itemsize:
            raise ValueError(f"{path}: inner array not packed")
        out += [base, so]
    return out


def _group_offsets(dtype: np.dtype, scalar: str, paths: Sequence[Tuple[str, ...]]) -> np.ndarray:
    """Byte offsets of a group's leaves."""
    out = []
    for p in paths:
        offs, dt = leaf_offsets(dtype, p)
        if dt != np.dtype(scalar).newbyteorder("<") and dt != np.dtype(scalar):
            raise TypeError(f"field {p}: {dt}, expected {scalar}")
        out.append(np.asarray(offs).reshape(-1))
    return np.concatenate(out).astype(np.int64)


_COUNTS = None


def group_counts() -> "OrderedDict[str, int]":
    global _COUNTS
    if _COUNTS is None:
        _COUNTS = OrderedDict((g[0], _group_offsets(OBS_DTYPE, g[1], g[2]).size) for g in GROUPS)
    return _COUNTS


def offsets(dtype: np.dtype) -> np.ndarray:
    """int32 byte offsets of every group (``GROUPS`` order), then the ``BLOCKS`` starts, then the ``ROWS``, for records
    of ``dtype`` (fireredsim.obs_dtype.OBS_DTYPE)."""
    parts = [_group_offsets(dtype, g[1], g[2]) for g in GROUPS]
    t0, t1 = byte_range(dtype, "tiles")
    if t1 - t0 != R.SEEN_ROWS * R.SEEN_COLS:
        raise ValueError("tiles field size")
    f0, f1 = byte_range(dtype, "flags_priv")
    if f1 - f0 != R.FLAG_BYTES:
        raise ValueError("flags_priv size")
    v0, v1 = byte_range(dtype, "vars_priv")
    s0 = v0 + 2 * int(R.SCENE_VAR_IDX[0])
    assert s0 + R.SCENE_BLOCK <= v1
    blocks = [t0, f0, s0, v0]
    out = np.concatenate(parts + [np.asarray(blocks, dtype=np.int64), np.asarray(_rows(dtype), dtype=np.int64)])
    if out.max() >= dtype.itemsize:
        raise ValueError("offset beyond the record")
    return np.ascontiguousarray(out, dtype=np.int32)


# the fields the reward step reads from every record (prefetched; the interaction key fields only on an interaction)
REWARD_GROUPS = ["MAP_GROUP", "MAP_NUM", "XY", "P_SPECIES", "P_IS_EGG", "P_LEVEL", "P_HP", "VALID", "FIELD_STATE",
                 "KEY_ID", "SKILL_EVENTS", "WAITING", "WAIT_CTX", "SCREEN", "FACING", "IN_DIALOG", "INTERACT_KIND",
                 "P_PID"]


def prefetch_offsets(dtype: np.dtype, groups: Sequence[str], blocks: Sequence[Tuple[int, int]] = ()) -> np.ndarray:
    """int32 offsets whose cache lines cover every byte the given groups (and byte blocks: (start, size)) touch in a
    record of ``dtype``, whatever the record's alignment: each run of touched bytes (gaps < 64 merged) contributes its
    start, every 64th byte after it and its last byte. Only a prefetch hint for the C loops (no effect on results)."""
    spans = []
    for g in GROUPS:
        name, s, paths = g[0], g[1], g[2]
        if name in groups:
            size = np.dtype(s).itemsize
            for o in _group_offsets(dtype, s, paths):
                spans.append((int(o), int(o) + size - 1))
    for o, size in blocks:
        spans.append((int(o), int(o) + size - 1))
    spans.sort()
    out, cur = [], None
    for a, b in spans:
        if cur is not None and a - cur[1] < 64:
            cur[1] = max(cur[1], b)
        else:
            if cur is not None:
                out.append(cur)
            cur = [a, b]
    if cur is not None:
        out.append(cur)
    pts = []
    for a, b in out:
        pts.extend(range(a, b + 1, 64))
        if pts[-1] != b:
            pts.append(b)
    return np.asarray(pts, dtype=np.int32)


def block_start(off: np.ndarray, name: str) -> int:
    """The ``BLOCKS`` entry ``name`` of an :func:`offsets` array."""
    return int(off[sum(group_counts().values()) + BLOCKS.index(name)])


def constants() -> "OrderedDict[str, object]":
    """Compiled-in sizes and scalars (``#define`` in gen_spec.h; a tuple: an initializer list ``{a, b, ...}``)."""
    c = OrderedDict()
    c["SEEN_ROWS"] = R.SEEN_ROWS        # the trail window (and the observer's tile window): shape, the player's cell
    c["SEEN_COLS"] = R.SEEN_COLS
    c["SEEN_CELLS"] = R.SEEN_ROWS * R.SEEN_COLS
    c["SEEN_ROW0"] = R.SEEN_ROW0
    c["SEEN_COL0"] = R.SEEN_COL0
    c["TRAIL_K"] = R.TRAIL_K            # the moves shown; a reset adds TRAIL_RESET to the move counter
    c["TRAIL_RESET"] = R.TRAIL_RESET
    c["TRAIL_EPOCH_SHIFT"] = R.TRAIL_EPOCH_SHIFT
    c["TRAIL_REFRESH_AGE"] = R.TRAIL_REFRESH_AGE
    c["FLAG_BYTES"] = R.FLAG_BYTES
    c["FLAGS_COUNT"] = T.FLAGS_COUNT
    c["SCENE_BLOCK"] = R.SCENE_BLOCK
    c["N_SCENE"] = int(R.SCENE_VAR_IDX.size)
    c["KEY_BYTES"] = R.KEY_BYTES        # per-env bitmap of the key item ids seen in the life
    c["ITEMS_COUNT"] = T.ITEMS_COUNT
    c["MAP_COUNT"] = R.MAP_COUNT
    c["TILE_COUNT"] = R.TILE_COUNT
    c["TILE_BYTES"] = R.TILE_BYTES
    c["OPP_SHIFT"] = R.OPP_SHIFT        # next-opponent key = ace << OPP_SHIFT | ladder index
    c["OPP_IDX"] = R.OPP_IDX
    c["HM_PAY_MASK"] = int(R.HM_PAY_MASK)  # the HMs that pay hm_cover / hm_taught (hm_reward_moves)
    c["CHAMP_BYTE"] = int(R.CHAMP_BYTE)
    c["CHAMP_MASK"] = int(R.CHAMP_MASK)
    c["SPECIES_EGG"] = T.SPECIES_EGG
    c["N_TERMS"] = len(R.TERMS)
    c["SUM_COLS"] = R.SUM_COLS          # the reward's summation order (reward.TERM_SUM_COL; an initializer list)
    c["TERM_SUM_COL"] = R.TERM_SUM_COL
    c["VALID_INGAME"] = R.VALID_INGAME
    c["FIELD_WARP"] = R.FIELD_WARP
    c["NUM_ACTIONS"] = M.NUM_ACTIONS
    c["A_INDEX"] = M.A_INDEX
    c["WAIT_START_MENU"] = M.WAIT_START_MENU
    c["WAIT_SAVE_YESNO"] = M.WAIT_SAVE_YESNO
    c["WAIT_SAVE_TEXT"] = M.WAIT_SAVE_TEXT
    c["START_MENU_NO_A_BITS"] = M.START_MENU_NO_A_BITS
    c["MENU_ACTION_RELEASE"] = M.MENU_ACTION_RELEASE
    # the skill term: its events and their cooldowns (reward.SKILL_COOLDOWNS: an initializer list, skill_cooldown[])
    c["N_SKILL"] = R.N_SKILL
    c["SKILL_COOLDOWNS"] = tuple(int(v) for v in R.SKILL_COOLDOWNS)
    c["SKILL_EVENT_MASK"] = R.SKILL_EVENT_MASK
    # reward weights: exact in float32 (the C code multiplies them as double or float32 like the term definitions)
    for k in ("TILE_REWARD", "TILE_COUNT_REWARD", "TILE_COUNT_SCALE", "MAP_REWARD", "HM_REWARD", "HM_TAUGHT_REWARD",
              "WHITEOUT_REWARD", "KEY_ITEM_REWARD", "LEVEL_REWARD"):
        c[k] = float(getattr(R, k))
    # the puzzle terms
    c["OBJ_N"] = R.OBJ_N
    c["OBJ_BYTES"] = R.OBJ_BYTES
    c["VERMILION_MAP"] = R.VERMILION_MAP
    c["OBJ_TERM0"] = R.OBJ_TERM0
    for k in ("OBJ_TRASH_CAN", "OBJ_MANSION_SWITCH", "OBJ_SPINNER", "OBJ_WARP_PAD", "OBJ_END_NONE", "OBJ_END_FLAG",
              "OBJ_END_KEY_ITEM", "SCREEN_OVERWORLD", "WAIT_OW_FREE"):
        c[k] = int(getattr(R, k))
    # the overworld interactions
    c["INTERACT_N"] = R.INTERACT_N
    c["INTERACT_BYTES"] = R.INTERACT_BYTES
    c["INTERACT_TERM"] = R.INTERACT_TERM
    c["MOVE_LEARN_TERM"] = R.MOVE_LEARN_TERM
    for k in ("INTERACT_OBJECT", "INTERACT_SIGN", "INTERACT_HIDDEN_ITEM", "INTERACT_SIGN_WALK", "INTERACT_TAG_OBJECT",
              "INTERACT_TAG_BG"):
        c[k] = int(getattr(R, k))
    c["N_PARTY"] = 6
    c["N_MOVES"] = 4
    c["N_HM"] = int(np.count_nonzero(R.HM_MOVE_BIT))
    c["N_CAP_BYTES"] = int(np.unique(R.ACE_BYTE).size)
    return c


def tables() -> "OrderedDict[str, np.ndarray]":
    """The arrays passed to the library (``struct pn_tables`` field order), as contiguous arrays."""
    t = OrderedDict()
    t["map_index"] = np.ascontiguousarray(R.MAP_INDEX, dtype=np.int64)
    t["flag_matrix"] = np.ascontiguousarray(R.FLAG_MATRIX, dtype=np.float32)
    t["hm_learn"] = np.ascontiguousarray(R.HM_LEARN, dtype=np.uint8)
    t["scene_cols"] = np.ascontiguousarray(R.SCENE_COLS, dtype=np.int32)
    t["map_w"] = np.ascontiguousarray(R.MAP_W, dtype=np.int64)
    t["map_h"] = np.ascontiguousarray(R.MAP_H, dtype=np.int64)
    t["map_off"] = np.ascontiguousarray(R.MAP_OFF, dtype=np.int64)
    t["action_buttons"] = np.ascontiguousarray(M.ACTION_BUTTONS, dtype=np.uint16)
    t["skill_weights"] = np.ascontiguousarray(R.SKILL_WEIGHTS, dtype=np.float32)
    # the object tiles (hashed), the per-kind / per-variable weights (runtime tables)
    t["obj_t"] = np.ascontiguousarray(R.OBJ_T, dtype=np.int64)
    t["obj_kind"] = np.ascontiguousarray(R.OBJ_KIND, dtype=np.uint8)
    t["obj_end_src"] = np.ascontiguousarray(R.OBJ_END_SRC, dtype=np.uint8)
    t["obj_end_bit"] = np.ascontiguousarray(R.OBJ_END_BIT, dtype=np.int32)
    t["obj_can_mask"] = np.ascontiguousarray(R.OBJ_CAN_MASK, dtype=np.uint8)
    t["map_obj_flags"] = np.ascontiguousarray(R.MAP_OBJ_FLAGS, dtype=np.uint8)
    t["obj_weights"] = np.ascontiguousarray(R.OBJ_WEIGHTS, dtype=np.float32)
    t["var_weights"] = np.ascontiguousarray(R.VAR_WEIGHTS, dtype=np.float32)
    # the interaction keys (hashed), the weight (a runtime table)
    t["interact_keys"] = np.ascontiguousarray(R.INTERACT_KEYS, dtype=np.int64)
    t["interact_weights"] = np.ascontiguousarray(R.INTERACT_WEIGHTS, dtype=np.float32)
    # reward.OPP_KEY as per-byte lookups: for each distinct seen byte holding ladder flags, table[v] = the min over
    # that byte's ladder entries of (OPP_KEY_AFTER if beaten else the entry's OPP_KEY) for byte value v; the key is the
    # min over the bytes (min is exact in any order), cap = key >> OPP_SHIFT, K = opp_size[key & OPP_IDX]
    cap_bytes = np.unique(R.ACE_BYTE)
    if cap_bytes.size > 8:
        raise ValueError("ACE_LADDER spans more than 8 flag bytes")
    tab = np.zeros((8, 256), dtype=np.int32)
    v = np.arange(256)
    for b_i, byte in enumerate(cap_bytes):
        ks = np.flatnonzero(R.ACE_BYTE == byte)
        cand = np.where((v[:, None] & R.ACE_MASK[ks][None, :]) != 0, R.OPP_KEY_AFTER, R.OPP_KEY[ks][None, :])
        tab[b_i] = cand.min(axis=1)
    t["cap_bytes"] = np.concatenate([cap_bytes, np.zeros(8 - cap_bytes.size, np.int64)]).astype(np.int64)
    t["opp_table"] = tab
    t["opp_size"] = np.ascontiguousarray(R.OPP_SIZE, dtype=np.int32)
    # the HM move ids (bit i of reward.HM_MOVE_BIT): the C code compares the party's moves against them
    hm = np.zeros(8, dtype=np.uint16)
    for m in np.flatnonzero(R.HM_MOVE_BIT):
        b = int(R.HM_MOVE_BIT[m])
        if b & (b - 1) or b >= 256:
            raise ValueError("HM_MOVE_BIT: one bit per move expected")
        i = b.bit_length() - 1
        if hm[i]:
            raise ValueError("HM_MOVE_BIT: two moves on one bit")
        hm[i] = m
    k = int(np.count_nonzero(hm))
    if not (hm[:k] != 0).all():
        raise ValueError("HM_MOVE_BIT: bits not contiguous from 0")
    t["hm_moves"] = hm
    return t


HASHED_TABLES = ("map_index", "flag_matrix", "hm_learn", "scene_cols", "map_w", "map_h", "map_off", "action_buttons",
                 "obj_t", "obj_kind", "obj_end_src", "obj_end_bit", "obj_can_mask", "map_obj_flags", "interact_keys",
                 "cap_bytes", "opp_table", "opp_size", "hm_moves")


def spec_hash(tabs: Dict[str, np.ndarray] = None) -> int:
    """64-bit hash of the record layout, the groups, the compiled constants and the source tables."""
    h = hashlib.sha256()
    h.update(repr(OBS_DTYPE.descr).encode())
    h.update(str(OBS_DTYPE.itemsize).encode())
    for g in GROUPS:
        h.update(repr(tuple(g)).encode())
    h.update(offsets(OBS_DTYPE).tobytes())
    h.update(repr(list(constants().items())).encode())
    tabs = tabs if tabs is not None else tables()
    for k in HASHED_TABLES:
        a = tabs[k]
        h.update(f"{k}:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    return int.from_bytes(h.digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


def header() -> str:
    """Contents of gen_spec.h."""
    lines = ["/* GENERATED by rl/pokerl/native/build.py from pokerl.native.spec. Do not edit. */",
             "#pragma once", ""]
    pos = 0
    for name, n in group_counts().items():
        lines.append(f"#define G_{name} {pos}")
        lines.append(f"#define N_{name} {n}")
        pos += n
    for b in BLOCKS:
        lines.append(f"#define B_{b} {pos}")
        pos += 1
    for name, _ in ROWS:
        lines.append(f"#define R_{name} {pos}")
        pos += 2
    lines.append(f"#define N_OFFSETS {pos}")
    lines.append("")
    for k, v in constants().items():
        if isinstance(v, tuple):
            lines.append(f"#define {k} {{{', '.join(str(int(x)) for x in v)}}}")
        else:
            lines.append(f"#define {k} {v!r}" if isinstance(v, float) else f"#define {k} {int(v)}")
    lines.append("")
    lines.append(f"#define SPEC_HASH {spec_hash()}LL")
    return "\n".join(lines) + "\n"
