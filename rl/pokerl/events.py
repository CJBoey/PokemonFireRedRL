"""Story milestones ("events") for the training logs (train.py ``EventLog``) and the relay scopes' predicates.

``EVENTS`` is the ordered list of milestones in FireRed story order (some are known blockers: the Tea for
Saffron's gates, the Lift Key and Silph Scope, the Route 12 Snorlax, the Card Key, the Secret Key). Each
:class:`Event` has a ``tag`` ``f"{i + 1:02d}_{name}"`` (zero-padded so TensorBoard sorts them in story order), a
``kind`` and a ``key`` (the game constant's name) resolved to ``id``:

* ``flag``: an event flag (``FLAG_*`` of game/include/constants/flags.h, or ``TFLAG_TRAINER_*`` = the trainer
  flag ``TRAINER_FLAGS_START + TRAINER_*``; :func:`flag_names`) set in ``flags_priv``;
* ``map``: the player is on that map (``MAP_*`` of ``tables.MAPS`` -> dense index 1..MAP_COUNT, ``reward.MAP_INDEX``);
* ``move``: a party member knows that move (``MOVE_*``; the HM moves of ``tables.HM_MOVES``) in ``party.moves``;
* ``var``: a game variable (``vars_priv``, ``VAR_*``) is at least ``arg`` (Oak's Parcel: the Viridian Mart scene
  variable, which the parcel scene sets to 1; the game has no flag for it);
* ``item``: an item id is in the key-items pocket (``key_items.id``; the Silph Scope: its object's hide flag is set
  at the new game, cleared when Giovanni leaves and set again on pickup, so no flag tells it).

:func:`observe` reads them from observation records (vectorised, one gather of the record bytes per kind; rows
without the INGAME bit of ``valid`` are all False). The env ORs them into a per-life matrix (env.py
``life_events``) so a map visited once stays true until the life ends.

Flags, variables and key items describe the game state, so a rewound life has the ones of its restored state at
once; maps and moves are what the life itself saw.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import numpy as np

from . import gen_tables
from . import tables as T
from .records import Gather, as_bytes, byte_range
from .reward import MAP_COUNT, MAP_INDEX, VALID_INGAME

KINDS = ("flag", "map", "move", "var", "item")

# (name, kind, key[, arg]) in story order; the tag is the 1-based position, zero-padded
EVENT_DEFS = [
    ("starter", "flag", "FLAG_SYS_POKEMON_GET"),
    ("oaks_parcel", "var", "VAR_MAP_SCENE_VIRIDIAN_CITY_MART", 1),
    ("pokedex", "flag", "FLAG_SYS_POKEDEX_GET"),
    ("viridian_forest", "map", "MAP_VIRIDIAN_FOREST"),
    ("pewter_city", "map", "MAP_PEWTER_CITY"),
    ("badge1", "flag", "FLAG_BADGE01_GET"),
    ("mt_moon", "map", "MAP_MT_MOON_1F"),
    ("cerulean_city", "map", "MAP_CERULEAN_CITY"),
    ("ss_ticket", "flag", "FLAG_GOT_SS_TICKET"),
    ("badge2", "flag", "FLAG_BADGE02_GET"),
    ("vermilion_city", "map", "MAP_VERMILION_CITY"),
    ("ss_anne", "map", "MAP_SSANNE_1F_CORRIDOR"),
    ("ss_anne_captain", "map", "MAP_SSANNE_CAPTAINS_OFFICE"),
    ("hm01_cut", "flag", "FLAG_GOT_HM01"),
    ("cut_known", "move", "MOVE_CUT"),
    ("badge3", "flag", "FLAG_BADGE03_GET"),
    ("digletts_cave", "map", "MAP_DIGLETTS_CAVE_B1F"),
    ("hm05_flash", "flag", "FLAG_GOT_HM05"),
    ("flash_known", "move", "MOVE_FLASH"),
    ("rock_tunnel", "map", "MAP_ROCK_TUNNEL_1F"),
    ("lavender_town", "map", "MAP_LAVENDER_TOWN"),
    ("celadon_city", "map", "MAP_CELADON_CITY"),
    ("tea", "flag", "FLAG_GOT_TEA"),                                # blocker: Saffron's gate guards want it
    ("badge4", "flag", "FLAG_BADGE04_GET"),
    ("rocket_hideout", "map", "MAP_ROCKET_HIDEOUT_B1F"),
    ("lift_key", "flag", "FLAG_CAN_USE_ROCKET_HIDEOUT_LIFT"),       # set when the Lift Key is picked up
    ("silph_scope", "item", "ITEM_SILPH_SCOPE"),
    ("pokemon_tower_7f", "map", "MAP_POKEMON_TOWER_7F"),            # past the ghost (needs the Silph Scope)
    ("poke_flute", "flag", "FLAG_GOT_POKE_FLUTE"),
    ("snorlax_route12", "flag", "FLAG_HIDE_ROUTE_12_SNORLAX"),      # set when the Snorlax is woken
    ("fuchsia_city", "map", "MAP_FUCHSIA_CITY"),
    ("badge5", "flag", "FLAG_BADGE05_GET"),
    ("hm03_surf", "flag", "FLAG_GOT_HM03"),
    ("surf_known", "move", "MOVE_SURF"),
    ("hm04_strength", "flag", "FLAG_GOT_HM04"),
    ("strength_known", "move", "MOVE_STRENGTH"),
    ("saffron_city", "map", "MAP_SAFFRON_CITY"),
    ("silph_co", "map", "MAP_SILPH_CO_1F"),
    ("card_key", "flag", "FLAG_HIDE_SILPH_CO_5F_CARD_KEY"),         # the Card Key's item ball, set on pickup
    ("silph_giovanni", "flag", "TFLAG_TRAINER_BOSS_GIOVANNI_2"),
    ("badge6", "flag", "FLAG_BADGE06_GET"),
    ("cinnabar_island", "map", "MAP_CINNABAR_ISLAND"),
    ("pokemon_mansion", "map", "MAP_POKEMON_MANSION_1F"),
    ("secret_key", "flag", "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY"),  # the Secret Key's item ball
    ("badge7", "flag", "FLAG_BADGE07_GET"),
    ("badge8", "flag", "FLAG_BADGE08_GET"),
    ("victory_road", "map", "MAP_VICTORY_ROAD_1F"),
    ("elite4_lorelei", "flag", "FLAG_DEFEATED_LORELEI"),
    ("elite4_bruno", "flag", "FLAG_DEFEATED_BRUNO"),
    ("elite4_agatha", "flag", "FLAG_DEFEATED_AGATHA"),
    ("elite4_lance", "flag", "FLAG_DEFEATED_LANCE"),
    ("champion", "flag", "FLAG_DEFEATED_CHAMP"),
    ("hall_of_fame", "flag", "FLAG_SYS_GAME_CLEAR"),                # (the episode ends at the Champion: rare)
    # appended after the story (keeps the tags above): Bill's One Island offer after Blaine was answered,
    # YES or NO (CinnabarIsland_EventScript_{Agree,Decline}SailToOneIsland set the scene var to 2; 3 = back with
    # Bill, 4 = he has left). The Viridian relay scope waits for it (relay/scopes.py), so a life is never confined on
    # the step the YES would carry it to One Island.
    ("bill_answered", "var", "VAR_MAP_SCENE_CINNABAR_ISLAND", 2),
]


class Event(NamedTuple):
    tag: str      # "13_ss_anne_captain"
    name: str
    kind: str     # KINDS
    key: str      # the game constant's name
    id: int       # flag id, dense map index, move id, var id or item id
    arg: int = 0  # var: the minimum value


_CONSTANTS: Optional[gen_tables.Constants] = None


def _constants() -> gen_tables.Constants:
    global _CONSTANTS
    if _CONSTANTS is None:
        _CONSTANTS = gen_tables.Constants()
    return _CONSTANTS


def flag_names() -> Dict[int, str]:
    """flag id -> name from game/include/constants/flags.h (+ trainer flags from opponents.h as ``TFLAG_TRAINER_*``)."""
    c = _constants()
    out: Dict[int, str] = {}
    for n, v in c.names("FLAG_").items():
        if 0 <= v < T.FLAGS_COUNT and (v not in out or out[v].startswith("FLAG_0x")):
            out[v] = n
    start = c.value("TRAINER_FLAGS_START")
    for n, v in c.names("TRAINER_").items():
        fid = start + v
        if n.startswith("TRAINER_FLAGS") or not (start <= fid < T.FLAGS_COUNT):
            continue
        if fid not in out or out[fid].startswith("FLAG_0x"):
            out[fid] = "TFLAG_" + n
    return out


def flag_id(name: str) -> int:
    """Flag id of ``FLAG_*`` or ``TFLAG_TRAINER_*`` (KeyError if unknown or out of the flag range)."""
    c = _constants()
    if name.startswith("TFLAG_"):
        fid = c.value("TRAINER_FLAGS_START") + c.value(name[len("TFLAG_"):])
    else:
        fid = c.value(name)
    if not 0 <= fid < T.FLAGS_COUNT:
        raise KeyError(f"{name} = {fid}: outside the {T.FLAGS_COUNT} flags")
    return fid


MAP_IDS = {n: i + 1 for i, (_, n, _, _) in enumerate(T.MAPS)}  # MAP_* -> dense index (reward.MAP_INDEX)


def resolve(defs=EVENT_DEFS) -> List[Event]:
    """The events of ``defs`` with their ids (KeyError naming the event for an unknown constant)."""
    c = _constants()
    out = []
    for i, d in enumerate(defs):
        name, kind, key = d[:3]
        arg = int(d[3]) if len(d) > 3 else 0
        try:
            if kind == "flag":
                v = flag_id(key)
            elif kind == "map":
                v = MAP_IDS[key]
            elif kind in ("move", "var", "item"):
                v = c.value(key)
            else:
                raise KeyError(f"unknown kind {kind!r}")
        except KeyError as e:
            raise KeyError(f"event {name}: {key}: {e}") from None
        out.append(Event(f"{i + 1:02d}_{name}", name, kind, key, int(v), arg))
    return out


EVENTS: List[Event] = resolve()
NUM_EVENTS = len(EVENTS)
TAGS = [e.tag for e in EVENTS]


def _cols(kind: str) -> np.ndarray:
    return np.array([i for i, e in enumerate(EVENTS) if e.kind == kind], dtype=np.intp)


_FLAG_COL, _MAP_COL, _MOVE_COL, _VAR_COL, _ITEM_COL = (_cols(k) for k in KINDS)
_FLAG_ID = np.array([EVENTS[i].id for i in _FLAG_COL], dtype=np.int64)
_FLAG_MASK = (1 << (_FLAG_ID & 7)).astype(np.uint8)
_MAP_ID = np.array([EVENTS[i].id for i in _MAP_COL], dtype=np.int64)
_MOVE_ID = np.array([EVENTS[i].id for i in _MOVE_COL], dtype=np.int64)
_VAR_IDX = np.array([EVENTS[i].id - _constants().value("VARS_START") for i in _VAR_COL], dtype=np.int64)
_VAR_MIN = np.array([EVENTS[i].arg for i in _VAR_COL], dtype=np.int64)
_ITEM_ID = np.array([EVENTS[i].id for i in _ITEM_COL], dtype=np.int64)


def _bit_table(ids: np.ndarray):
    """u16 value -> uint8 bit j for ids[j] (at most 8 ids), and the bits: OR-reduce a row's values through it."""
    assert ids.size <= 8
    lut = np.zeros(65536, dtype=np.uint8)
    bits = (1 << np.arange(ids.size)).astype(np.uint8)
    for b, v in zip(bits, ids):
        lut[v] |= b
    return lut, bits


_MOVE_LUT, _MOVE_BITS = _bit_table(_MOVE_ID)
_ITEM_LUT, _ITEM_BITS = _bit_table(_ITEM_ID)

_PLANS: Dict[np.dtype, tuple] = {}


def _plan(dtype: np.dtype):
    p = _PLANS.get(dtype)
    if p is None:
        u8 = Gather(dtype, np.uint8, [("valid", ["valid"]), ("map_group", ["map_group"]), ("map_num", ["map_num"])])
        u16 = Gather(dtype, np.uint16, [("moves", ["party", "moves"]), ("key_ids", ["key_items", "id"])])
        f0, f1 = byte_range(dtype, "flags_priv")
        v0, v1 = byte_range(dtype, "vars_priv")
        assert f0 + int(_FLAG_ID.max(initial=0) >> 3) < f1 and v0 + 2 * int(_VAR_IDX.max(initial=0)) + 1 < v1
        flag_bytes = f0 + (_FLAG_ID >> 3)
        var_lo = v0 + 2 * _VAR_IDX
        p = _PLANS[dtype] = (u8, u16, flag_bytes, var_lo)
    return p


# observe() builds the columns kind by kind (flags, maps, moves, variables, items) and permutes them once
_ORDER = np.argsort(np.concatenate([_FLAG_COL, _MAP_COL, _MOVE_COL, _VAR_COL, _ITEM_COL]))
_MAP_LUT = np.zeros((MAP_COUNT + 1, _MAP_COL.size), dtype=bool)  # dense map index -> its map events (0: none)
_MAP_LUT[_MAP_ID, np.arange(_MAP_COL.size)] = True


def observe(obs) -> np.ndarray:
    """bool [N, NUM_EVENTS]: the events true in records ``obs`` (rows outside the game: all False)."""
    u8, u16, flag_bytes, var_lo = _plan(obs.dtype)
    rec = as_bytes(obs)
    n = rec.shape[0]
    a = u8(rec)
    ingame = (a[:, 0] & VALID_INGAME) != 0
    b = u16(rec)
    known = np.bitwise_or.reduce(_MOVE_LUT[u16.get(b, "moves").reshape(n, -1)], axis=1)
    held = np.bitwise_or.reduce(_ITEM_LUT[u16.get(b, "key_ids")], axis=1)
    v = rec[:, var_lo].astype(np.int64) | (rec[:, var_lo + 1].astype(np.int64) << 8)
    out = np.concatenate([(rec[:, flag_bytes] & _FLAG_MASK) != 0,
                          _MAP_LUT[MAP_INDEX[a[:, 1].astype(np.int64) * 256 + a[:, 2]]],
                          (known[:, None] & _MOVE_BITS) != 0, v >= _VAR_MIN,
                          (held[:, None] & _ITEM_BITS) != 0], axis=1)[:, _ORDER]
    out &= ingame[:, None]
    return out
