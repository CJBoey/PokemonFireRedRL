"""The reward: its terms, weights and lookup tables (the per-step computation is the C code of pokerl.native,
``NativeReward``). Every term reads the record at the step boundary and per-life state that a life's first record
initialises (so nothing already true at a life's start pays); rows outside the game (no INGAME bit) hold their state
and pay nothing. The v76 weights are train.py's defaults.

1. ``flags``, ``trainers``, ``badges``, ``elite4``: an event flag first seen set in the life (privileged
   ``flags_priv``): story flags 1, item balls of non-key items 0.25, trainer flags ``trainer_reward`` (1), the 8
   badges ``badge_reward`` (10 each), the Elite Four ``elite4_reward`` (10), the Champion ``champion_reward`` (20),
   FLAG_TEMP_1..6 (cleared by the game on every map change) ``temp_flag_reward`` (1), the Vermilion Gym's
   FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES ``trash_unlock_reward`` (5); hidden items and the other system flags 0.
2. ``tiles`` per (map, x, y) first visited in the life: ``tile_reward`` (0.02) + the count bonus W / sqrt(max(N, S) /
   S), N = 1 + the number of earlier lives of the whole run (all envs: ``tile_count``) that first-visited it, W =
   ``tile_count_reward`` (0.05), S = ``tile_count_scale`` (512: full pay for a tile's first 512 first visits);
   ``maps``: ``map_reward`` (0.05) per map first visited in the life.
3. ``levels``: ``level_reward`` (1) x the rise of the life's maximum of S = sum of min(level, cap) over the K
   highest-level non-egg party mons, cap = the next opponent's ace level and K = its party size (the opponents: the
   8 gym leaders, then the Champion); nothing on a step where cap or K changes or a mon entered the party (a catch, a
   withdrawal, a gift: the maximum re-based), so a mon's levels pay only when it gains them in the party.
   ``hm_cover``: 2 per HM the party can learn and ``hm_taught``: ``hm_taught_reward`` (10) per HM move known by a
   party mon, each bit first seen in the life (``hm_seen``), only the HMs of ``hm_reward_moves`` (Cut, Surf,
   Strength).
4. ``whiteout``: ``whiteout_reward`` (-0.2) on the step every non-egg party mon reaches HP 0.
5. ``vars``: 2 each time a story variable (the trigger variables of map scripts, ``tables.SCENE_VAR_IDX``) exceeds
   its maximum in the life.
6. ``key_items``: 5 per item id first seen in the life in the key-items pocket.
7. ``skill``: the worker's skill events (observer ``skill_events_priv``, obs/src/skill_events.c), each on its own
   cooldown (``skill_event_cooldown``: 250 steps, party_order 2000) with its weight ``SKILL_WEIGHTS``
   (``skill_event_reward`` after ``skill_reward`` / ``skill_menu_reward``: v76 pays the party / bag menus and the
   player's PC 0.02, Bill's PC 0.1, a party reorder 1.0, the rest 0).
8. Puzzles on the object tiles of ``tables.OBJECT_TILES`` (each with an end condition, a flag or a key item, after
   which it pays nothing; once per tile and life): ``trash`` (``trash_reward``, 0.1) A on a Vermilion Gym trash can
   (once per can and switch draw: a new draw clears the cans), ``mansion`` (0.1) a Pokemon Mansion switch statue,
   ``spinner`` / ``warp_pad`` (0.1 each) a spinner or warp pad used.
9. ``interact`` (``interact_reward``, 0.1): an overworld interaction (an object talked to, a sign or hidden item
   checked; observer ``interact_*_priv``) with a key first started in the life.
10. ``move_learn``: ``move_replace_reward`` (0.5) per move a party mon learnt in place of a forgotten one (observer
    ``learned_move_priv`` / ``learned_replaced_priv``), once per (mon, move) in the life (``ML_KEYS`` keys); not for
    an HM move outside ``hm_reward_moves`` (:func:`move_learn_pass`).

``progress`` (the stagnation rule, env.py) = a flag of weight > 0 newly seen, a capped-level gain, an HM gain, a new
map, a scene-variable increase or a new key item.

The life's tiles are a per-env bitmap over every map's (width x height) grid (``tables.MAPS``); positions outside the
map grid or during the warp window (``field_state`` 2: the map is already the destination, x/y are not) earn no tile.
The run-long tile counts are exact (one uint32 per tile, ``TILE_COUNT`` ~ 234k, shared by every env and env group;
envs that first-visit a tile on the same step all see the count from before that step). The life's start tile is
marked without being counted. ``state_dict`` / ``load_state_dict`` (pokerl.native) save the counts for checkpoints.

Trail (the policy's trail plane, env.py): per env a move counter ``trail_now``, the tile of the last move
``trail_last`` and the low 16 bits of the move number of each tile's last entry ``trail_t`` [N, TILE_COUNT]. A move =
a step whose record marks a tile other than ``trail_last``: the counter goes up by 1 and stamps the tile. A life's
start (and a rewind restore) adds TRAIL_K + 2 to the counter (every earlier stamp is then older than the window
shows) and stamps the start tile. The window: per cell 255 - age for age = trail_now - stamp (uint16) <= TRAIL_K - 1
= 254, else 0, over the 11 x 15 tiles around the player (0 outside the map and on OUTSIDE tile classes). Moves, not
steps: battles, menus and dialogue do not age the trail. 16-bit stamps would alias, so whenever an env's counter
enters a new epoch (trail_now >> TRAIL_EPOCH_SHIFT, 32,768 moves) every stamp of that env whose age is >= TRAIL_K is
rewritten to age TRAIL_REFRESH_AGE (16,384).

Carry (env.py "Rewind"): ``PAID_KEYS``, the per-life "already paid" state a rewind capture keeps, so that the reward
after a rewind is the captured life's: the flags, key items and scene-variable maxima seen, the level-sum maximum
with its cap and K, the object tiles used, the interaction keys and the skill events' remaining cooldowns (with the
HM bits and the move keys, which are always carried).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import zlib

import numpy as np

from . import tables as T

TERMS = ("flags", "trainers", "badges", "elite4", "tiles", "maps", "levels", "hm_cover", "hm_taught", "whiteout",
         "vars", "key_items", "skill", "trash", "mansion", "spinner", "warp_pad", "interact", "move_learn")
MOVE_LEARN_TERM = TERMS.index("move_learn")
# The step reward is the float32 sum of its terms in numpy's pairwise order over the 26 term columns of the v76 run
# (whose 7 other terms, off in v76, were always 0): the order of the float additions decides the last bit of a few
# rewards, so this keeps the step rewards bit-identical to v76's. TERM_SUM_COL: each term's column there.
SUM_COLS = 26
TERM_SUM_COL = tuple(range(17)) + (18, 24)

TILE_REWARD = 0.02  # the floor of a tile's first visit in a life (configure's tile_reward)
TILE_COUNT_REWARD = 0.0  # W: + W / sqrt(max(N, S) / S) on a tile's first visit (configure's tile_count_reward)
TILE_COUNT_SCALE = 1.0  # S (configure's tile_count_scale)
MAP_REWARD_DEFAULT = 0.5  # per map first visited in a life (configure's map_reward)
MAP_REWARD = MAP_REWARD_DEFAULT
HM_REWARD = 2.0  # hm_cover
HM_TAUGHT_REWARD_DEFAULT = 20.0  # hm_taught (configure's hm_taught_reward)
HM_TAUGHT_REWARD = HM_TAUGHT_REWARD_DEFAULT
WHITEOUT_REWARD_DEFAULT = -0.1  # configure's whiteout_reward
WHITEOUT_REWARD = WHITEOUT_REWARD_DEFAULT
LEVEL_REWARD_DEFAULT = 1.0  # the levels term's weight (configure's level_reward)
LEVEL_REWARD = LEVEL_REWARD_DEFAULT
# move_learn (term 10): MOVE_REPLACE_REWARD per move learnt in place of a forgotten one, once per (mon personality,
# move) and life: ML_KEYS keys per env, then nothing more; the keys are carried by rewinds
MOVE_REPLACE_REWARD = 0.0
ML_KEYS = 128
VAR_REWARD = 2.0
KEY_ITEM_REWARD = 5.0
SKILL_REWARD = 1.0  # per skill event off cooldown (configure's skill_reward)
SKILL_COOLDOWN = 250  # steps per env and event (the default of every event)
SKILL_EVENTS = ("party_menu", "bag_menu", "party_order", "use_item", "use_key_item", "use_tm", "pc_storage", "pc_player",
                "pc_party", "tree", "boulder", "rock", "mart", "use_consumable", "use_battle_item", "mart_buy")
N_SKILL = len(SKILL_EVENTS)
SKILL_COOLDOWNS = np.full(N_SKILL, SKILL_COOLDOWN, dtype=np.int64)  # (configure's skill_event_cooldown; compiled)
# events whose default weight is 0 (only skill_event_reward turns them on)
SKILL_DEFAULT_OFF = ("mart", "use_consumable", "use_battle_item", "mart_buy")
SKILL_DEFAULT_OFF_IDX = np.array([SKILL_EVENTS.index(k) for k in SKILL_DEFAULT_OFF], dtype=np.int64)
# "opening a menu" events: party menu, bag, Bill's PC, the player's PC (configure's skill_menu_reward)
SKILL_MENU_EVENTS = ("party_menu", "bag_menu", "pc_storage", "pc_player")
SKILL_MENU_IDX = np.array([SKILL_EVENTS.index(k) for k in SKILL_MENU_EVENTS], dtype=np.int64)
SKILL_WEIGHTS = np.full(N_SKILL, SKILL_REWARD, dtype=np.float32)  # per-event reward (a native table; configure sets it)
SKILL_WEIGHTS[SKILL_DEFAULT_OFF_IDX] = 0
SKILL_EVENT_MASK = (1 << N_SKILL) - 1  # bits 0-15 of skill_events_priv
# enum values of obs/observation.h (checked against fireredsim.obs_dtype by rl/tests/test_reward.py)
SCREEN_OVERWORLD = 7
WAIT_OW_FREE = 11
VALID_INGAME = 1
FIELD_WARP = 2
# the puzzle terms' weights (configure; runtime table OBJ_WEIGHTS of the native library)
TRASH_REWARD = 0.0            # per Vermilion Gym trash can interaction (once per can and switch assignment)
TRASH_UNLOCK_REWARD = None    # weight of FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES (None: its story weight)
MANSION_SWITCH_REWARD = 0.0   # per Pokemon Mansion switch statue first pressed in a life
SPINNER_REWARD = 0.0          # per spinner tile first used in a life
WARP_PAD_REWARD = 0.0         # per warp pad first used in a life
OBJ_TRASH_CAN, OBJ_MANSION_SWITCH, OBJ_SPINNER, OBJ_WARP_PAD = 0, 1, 2, 3  # OBJ_WEIGHTS / term order
OBJ_WEIGHTS = np.zeros(4, dtype=np.float32)  # per kind (OBJ_* order)
OBJ_TERM0 = TERMS.index("trash")             # term columns of the kinds: OBJ_TERM0 + kind
assert TERMS[OBJ_TERM0:OBJ_TERM0 + 4] == ("trash", "mansion", "spinner", "warp_pad")
# the overworld interaction term (configure; the runtime table INTERACT_WEIGHTS of the native library)
INTERACT_REWARD = 0.0         # per interaction key (object / background event) first started in a life
INTERACT_WEIGHTS = np.zeros(1, dtype=np.float32)  # [INTERACT_REWARD]
INTERACT_TERM = TERMS.index("interact")
# enum PnInteractKind of obs/observation.h (checked against fireredsim.obs_dtype by the tests)
INTERACT_NONE, INTERACT_OBJECT, INTERACT_SIGN, INTERACT_HIDDEN_ITEM, INTERACT_SIGN_WALK = 0, 1, 2, 3, 4

# ---- lookup tables -----------------------------------------------------------------------------------

FLAG_WEIGHTS = np.asarray(T.FLAG_WEIGHTS, dtype=np.float32)
_cats = np.asarray(T.FLAG_CATS)
_BASE_WEIGHTS = FLAG_WEIGHTS.copy()  # tables.py's weights: configure rescales them in place
_ITEM_BALL = (_cats == T.FLAG_CAT_STORY) & (_BASE_WEIGHTS == np.float32(0.25))  # non-key item balls (gen_tables)
_STORY = (_cats == T.FLAG_CAT_STORY) & ~_ITEM_BALL & (_BASE_WEIGHTS > 0)
STORY_WEIGHT, ITEM_BALL_WEIGHT, TRAINER_WEIGHT = 1.0, 0.25, 0.5  # (configure's trainer_weight overrides the last)
# [2304, 5]: column = flags, trainers, badges, elite4, earned-count (weight > 0)
FLAG_MATRIX = np.zeros((T.FLAGS_COUNT, 5), dtype=np.float32)
FLAG_BYTES = T.FLAGS_COUNT // 8
TEMP_FLAGS = np.arange(1, 7, dtype=np.int64)  # FLAG_TEMP_1..6 (game constants/flags.h: TEMP_FLAGS_START (0) + 1..6)
TEMP_FLAG_REWARD = 0.0  # their weight (category NONE in tables.py)
BADGE_FLAGS = np.asarray(T.BADGE_FLAGS, dtype=np.int64)  # FLAG_BADGE01_GET..FLAG_BADGE08_GET
ELITE4_FLAGS = np.flatnonzero(_cats == T.FLAG_CAT_ELITE4)  # Lorelei, Bruno, Agatha, Lance, the Champion
assert np.all(_cats[TEMP_FLAGS] == T.FLAG_CAT_NONE) and np.all(_BASE_WEIGHTS[TEMP_FLAGS] == 0)
assert BADGE_FLAGS.size == 8 and np.all(_cats[BADGE_FLAGS] == T.FLAG_CAT_BADGE) and ELITE4_FLAGS.size == 5

# species / move ids are u16: tables padded to 65536 so any value indexes safely
HM_LEARN = np.zeros(65536, dtype=np.uint8)
HM_LEARN[:T.SPECIES_COUNT] = T.HM_LEARN
HM_MOVE_BIT = np.zeros(65536, dtype=np.uint8)
for _i, _m in enumerate(T.HM_MOVES):
    HM_MOVE_BIT[_m] = 1 << _i
HM_ALL = (1 << len(T.HM_MOVES)) - 1  # coverage of every obtainable HM (HM01-HM07; HM08 Dive is not)
assert int(HM_LEARN.max()) <= HM_ALL
# configure's hm_reward_moves: bit i = HM_NAMES[i] = T.HM_MOVES[i] pays hm_cover / hm_taught; HM_UNPAID_MOVE[move] =
# an HM move outside it (no move_learn pay)
HM_NAMES = ("cut", "fly", "surf", "strength", "flash", "rock_smash", "waterfall")
assert [T.MOVE_NAMES[m].lower() for m in T.HM_MOVES] == list(HM_NAMES)
HM_PAY_MASK = HM_ALL
HM_UNPAID_MOVE = np.zeros(65536, dtype=bool)
SCENE_VAR_IDX = np.asarray(T.SCENE_VAR_IDX, dtype=np.intp)
SCENE_COLS = SCENE_VAR_IDX - SCENE_VAR_IDX[0]  # u16 columns of the scene variables in their byte block
SCENE_BLOCK = 8 * ((2 * int(SCENE_COLS[-1] + 1) + 7) // 8)  # bytes from the first scene variable, x 8
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
VAR_WEIGHTS = np.full(SCENE_VAR_IDX.size, VAR_REWARD, dtype=np.float32)  # per scene variable weight of ``vars``
TRASH_UNLOCK_FLAG = int(T.FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES)
assert _cats[TRASH_UNLOCK_FLAG] == T.FLAG_CAT_STORY
KEY_BYTES = 8 * ((T.ITEMS_COUNT + 63) // 64)  # per-env bitmap of the key item ids seen in the life

MAP_COUNT = len(T.MAPS)
MAP_INDEX = np.zeros(65536, dtype=np.int64)  # map_group * 256 + map_num -> 1..MAP_COUNT, unknown 0
MAP_W = np.zeros(MAP_COUNT + 1, dtype=np.int64)
MAP_H = np.zeros(MAP_COUNT + 1, dtype=np.int64)
for _i, (_k, _n, _w, _h) in enumerate(T.MAPS):
    MAP_INDEX[_k] = _i + 1
    MAP_W[_i + 1], MAP_H[_i + 1] = _w, _h
MAP_OFF = np.concatenate([[0], np.cumsum(MAP_W * MAP_H)[:-1]]).astype(np.int64)
TILE_COUNT = int((MAP_W * MAP_H).sum())
# identifies the tile layout (map order and sizes) a saved count table was indexed with
TILE_LAYOUT = zlib.crc32(np.array([(k, w, h) for k, _, w, h in T.MAPS], dtype=np.int64).tobytes())
TILE_BYTES = (TILE_COUNT + 7) // 8
# the trail window: the observer's 11 x 15 tile window around the player (row SEEN_ROW0, column SEEN_COL0), cell (i,
# j) = map tile (x - SEEN_COL0 + j, y - SEEN_ROW0 + i)
SEEN_ROWS, SEEN_COLS = 11, 15
SEEN_ROW0, SEEN_COL0 = SEEN_ROWS // 2, SEEN_COLS // 2
TRAIL_K = 255  # the moves the trail shows (the tile under the player 255, then 254, ..., 1; the policy scales 1 / 255)
TRAIL_RESET = TRAIL_K + 2  # added to an env's move counter by ``reset`` (its earlier stamps then read 0; +1: the start)
TRAIL_EPOCH_SHIFT = 15  # a refresh whenever trail_now >> 15 changes (every 32,768 moves)
TRAIL_REFRESH_AGE = 16384  # the age a refresh gives the stale stamps (module docstring "Trail")

# the object tiles (tables.OBJECT_TILES; term 8): sorted by global tile index (the tile bitmap's index), one entry
# per tile; per entry the kind, the end condition (OBJ_END_* source: 0 none, 1 a flag bit of the seen flags, 2 an item
# bit of the key item bitmap) and its bit
OBJ_END_NONE, OBJ_END_FLAG, OBJ_END_KEY_ITEM = T.OBJ_END_NONE, T.OBJ_END_FLAG, T.OBJ_END_KEY_ITEM
_objs = sorted((int(MAP_OFF[MAP_INDEX[k]] + y * MAP_W[MAP_INDEX[k]] + x), kd, i, ek, ev) for k, x, y, kd, i, ek, ev in T.OBJECT_TILES)
OBJ_T = np.asarray([o[0] for o in _objs], dtype=np.int64)
OBJ_KIND = np.asarray([o[1] for o in _objs], dtype=np.uint8)
OBJ_END_SRC = np.asarray([o[3] for o in _objs], dtype=np.uint8)
OBJ_END_BIT = np.asarray([o[4] for o in _objs], dtype=np.int32)
OBJ_N = int(OBJ_T.size)
OBJ_BYTES = (OBJ_N + 7) // 8
assert OBJ_N > 0 and np.all(np.diff(OBJ_T) > 0) and OBJ_KIND.max() <= OBJ_WARP_PAD
assert np.all(OBJ_END_BIT[OBJ_END_SRC == OBJ_END_FLAG] < T.FLAGS_COUNT) and np.all(OBJ_END_BIT[OBJ_END_SRC == OBJ_END_KEY_ITEM] < KEY_BYTES * 8)
OBJ_CAN_MASK = np.packbits(OBJ_KIND == OBJ_TRASH_CAN, bitorder="little")  # [OBJ_BYTES]: the trash cans' bits
assert OBJ_CAN_MASK.size == OBJ_BYTES and POPCOUNT8[OBJ_CAN_MASK].sum() == 15
VERMILION_MAP = int(MAP_INDEX[T.VERMILION_GYM_MAP])  # dense map id of the gym (the trash cans' attempt key)
assert VERMILION_MAP > 0
# per dense map id: 1 = the map has object tiles (the native fast path: a record on any other map with no object
# faced at the previous step skips the lookups; the results are the same)
MAP_OBJ_FLAGS = np.zeros(MAP_COUNT + 1, dtype=np.uint8)
MAP_OBJ_FLAGS[[int(MAP_INDEX[k]) for k, *_ in T.OBJECT_TILES]] = 1
MAP_OBJ_FLAGS[0] = 0

# the interaction keys (term 9) as int64: tag << 48 | map key << 32 | the rest, the tag 1 for an object (rest = local
# id), 2 for a background event (rest = x << 16 | y, both as u16)
INTERACT_TAG_OBJECT, INTERACT_TAG_BG = 1, 2
INTERACT_KIND_TAG = np.zeros(256, dtype=np.int64)  # record kind -> tag
INTERACT_KIND_TAG[INTERACT_OBJECT] = INTERACT_TAG_OBJECT
INTERACT_KIND_TAG[[INTERACT_SIGN, INTERACT_HIDDEN_ITEM, INTERACT_SIGN_WALK]] = INTERACT_TAG_BG


def interact_key(kind, group, num, local_id, x, y) -> np.ndarray:
    """int64 keys of reported interactions (arrays of the record fields, broadcast): an object's (map group, map num,
    local id), a background event's (map group, map num, x, y); kind NONE / unknown: a key of tag 0 (no entry)."""
    tag = INTERACT_KIND_TAG[np.asarray(kind, dtype=np.int64)]
    mk = (np.asarray(group, dtype=np.int64) << 8) | np.asarray(num, dtype=np.int64)
    rest = np.where(tag == INTERACT_TAG_OBJECT, np.asarray(local_id, dtype=np.int64),
                    ((np.asarray(x, dtype=np.int64) & 0xFFFF) << 16) | (np.asarray(y, dtype=np.int64) & 0xFFFF))
    return (tag << 48) | (mk << 32) | np.where(tag != 0, rest, 0)


INTERACT_KEYS = np.sort(np.concatenate([
    interact_key(INTERACT_OBJECT, [k >> 8 for k, _ in T.INTERACT_OBJECTS], [k & 0xFF for k, _ in T.INTERACT_OBJECTS],
                 [i for _, i in T.INTERACT_OBJECTS], 0, 0),
    interact_key(INTERACT_SIGN, [k >> 8 for k, _, _ in T.INTERACT_BG], [k & 0xFF for k, _, _ in T.INTERACT_BG], 0,
                 [x for _, x, _ in T.INTERACT_BG], [y for _, _, y in T.INTERACT_BG])]))
INTERACT_N = int(INTERACT_KEYS.size)
INTERACT_BYTES = (INTERACT_N + 7) // 8
assert INTERACT_N == len(T.INTERACT_OBJECTS) + len(T.INTERACT_BG) and np.all(np.diff(INTERACT_KEYS) > 0)

ACE = np.array([a for _, a, _ in T.ACE_LADDER], dtype=np.int32)
_ace_flags = np.array([f for _, _, f in T.ACE_LADDER], dtype=np.int64)
ACE_BYTE = _ace_flags >> 3
ACE_MASK = (1 << (_ace_flags & 7)).astype(np.uint8)
CAP_AFTER = int(T.ACE_AFTER_CHAMPION)
# the next opponent as one ordered key, ace * 16 + ladder index: the min over the unbeaten opponents is the lowest ace
# (the first in ladder order among equal aces); beaten -> the Champion's key, the largest (after the Champion: the
# Champion's ace and party size). cap = key >> OPP_SHIFT, K = OPP_SIZE[key & OPP_IDX]
OPP_SHIFT, OPP_IDX = 4, 15
OPP_SIZE = np.zeros(OPP_IDX + 1, dtype=np.int32)
OPP_SIZE[:ACE.size] = T.ACE_PARTY_SIZES
OPP_KEY = (ACE << OPP_SHIFT) + np.arange(ACE.size, dtype=np.int32)
OPP_KEY_AFTER = int(OPP_KEY[-1])
assert ACE.size <= OPP_IDX + 1 and OPP_KEY_AFTER == OPP_KEY.max() and ACE[-1] == CAP_AFTER
assert int(T.PARTY_SIZE_AFTER_CHAMPION) == OPP_SIZE[ACE.size - 1] and 1 <= OPP_SIZE[:ACE.size].min()
assert OPP_SIZE.max() <= 6
CHAMP_BYTE, CHAMP_MASK = T.FLAG_DEFEATED_CHAMP >> 3, np.uint8(1 << (T.FLAG_DEFEATED_CHAMP & 7))

# The per-life "already paid" state a rewind capture also carries (module docstring "Carry"): the reward state's
# arrays of these names, and the skill events' remaining cooldown steps (int64 [k, N_SKILL], relative: env groups
# count their own steps)
PAID_CARRY = ("seen", "key_seen", "var_max", "best", "cap", "opp_size", "obj_bm", "interact_bm")
PAID_COOLDOWN = "skill_cd"
PAID_KEYS = PAID_CARRY + (PAID_COOLDOWN,)


# ---- configuration ---------------------------------------------------------------------------------------------

def parse_skill_event_reward(spec) -> Dict[str, float]:
    """``--skill-event-reward`` ("name=W,name=W", a mapping, or None/"" for none) -> {event: weight}; the names are
    those of SKILL_EVENTS (unknown name, malformed item or negative weight: ValueError)."""
    out = {}
    for k, v in _items(spec, "skill event reward", "name=weight"):
        if k not in SKILL_EVENTS:
            raise ValueError(f"skill event reward: unknown event {k!r} (events: {', '.join(SKILL_EVENTS)})")
        try:
            w = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"skill event reward {k}={v!r}: not a number") from None
        if not w >= 0:
            raise ValueError(f"skill event reward {k}={w}: must be >= 0")
        out[k] = w
    return out


def parse_skill_event_cooldown(spec) -> Dict[str, int]:
    """``--skill-event-cooldown`` ("name=STEPS,name=STEPS", a mapping, or None/"" for none) -> {event: steps}; the
    names are those of SKILL_EVENTS, the steps whole numbers >= 0 (0: no cooldown); unknown name, malformed item or a
    bad value: ValueError."""
    out = {}
    for k, v in _items(spec, "skill event cooldown", "name=steps"):
        if k not in SKILL_EVENTS:
            raise ValueError(f"skill event cooldown: unknown event {k!r} (events: {', '.join(SKILL_EVENTS)})")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"skill event cooldown {k}={v!r}: not a number") from None
        if not (0 <= f < 2 ** 31 and f == int(f)):
            raise ValueError(f"skill event cooldown {k}={v!r}: must be a whole number of steps >= 0")
        out[k] = int(f)
    return out


def _items(spec, what: str, form: str):
    """The (name, value) items of a "name=value,..." string or a mapping (None: none)."""
    if spec is None:
        return []
    if not isinstance(spec, str):
        return list(dict(spec).items())
    items = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, eq, v = part.partition("=")
        if not eq:
            raise ValueError(f"{what} {part!r}: expected {form}")
        items.append((k.strip(), v.strip()))
    return items


def parse_badge_reward(spec) -> Optional[np.ndarray]:
    """``--badge-reward`` ("w1,...,w8" or 8 numbers, None for the tables' weights) -> float32 [8] (ValueError unless 8
    numbers >= 0)."""
    if spec is None:
        return None
    vals = [v for v in spec.split(",") if v.strip()] if isinstance(spec, str) else list(spec)
    try:
        w = np.array([float(v) for v in vals], dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"badge reward {spec!r}: expected 8 numbers") from None
    if w.size != BADGE_FLAGS.size or not np.all(w >= 0):
        raise ValueError(f"badge reward {spec!r}: expected 8 weights >= 0 (FLAG_BADGE01_GET..FLAG_BADGE08_GET)")
    return w.astype(np.float32)


def parse_hm_reward_moves(spec) -> Optional[Tuple[str, ...]]:
    """``--hm-reward-moves`` ("cut,surf,strength", a list of HM_NAMES, or None / "all" for every HM) -> the names in
    HM order, or None for all (ValueError naming an unknown HM)."""
    if spec is None:
        return None
    vals = [v.strip().lower() for v in spec.split(",") if v.strip()] if isinstance(spec, str) else \
        [str(v).strip().lower() for v in spec]
    if vals == ["all"]:
        return None
    bad = [v for v in vals if v not in HM_NAMES]
    if bad:
        raise ValueError(f"hm reward moves {spec!r}: unknown HM(s) {', '.join(bad)} (names: {', '.join(HM_NAMES)})")
    return tuple(h for h in HM_NAMES if h in vals)


def _weight(name: str, v) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    if not v >= 0:
        raise ValueError(f"{name} {v}: must be >= 0")
    return v


def configure(skill_reward: float = None, skill_menu_reward: float = None, trainer_weight: float = None,
              temp_flag_reward: float = None, skill_event_reward=None, badge_reward=None, elite4_reward: float = None,
              champion_reward: float = None, tile_reward: float = None, map_reward: float = None,
              whiteout_reward: float = None, trash_reward: float = None, trash_unlock_reward: float = None,
              mansion_switch_reward: float = None, spinner_reward: float = None, warp_pad_reward: float = None,
              interact_reward: float = None, hm_taught_reward: float = None, tile_count_reward: float = None,
              tile_count_scale: float = None, level_reward: float = None, skill_event_cooldown=None,
              move_replace_reward: float = None, hm_reward_moves=None) -> None:
    """Sets the reward weights (idempotent; None = the module default; the module docstring has the terms). Invalid
    values raise ValueError before anything changes. The flag weights change FLAG_MATRIX (hashed by the native spec: a
    library built for other weights is rebuilt); TILE_REWARD, TILE_COUNT_REWARD, TILE_COUNT_SCALE, MAP_REWARD,
    WHITEOUT_REWARD, HM_TAUGHT_REWARD, LEVEL_REWARD, HM_PAY_MASK and SKILL_COOLDOWNS are compiled native constants;
    SKILL_WEIGHTS, OBJ_WEIGHTS and INTERACT_WEIGHTS are runtime tables."""
    global TILE_REWARD, TILE_COUNT_REWARD, TILE_COUNT_SCALE, MAP_REWARD, SKILL_REWARD, TEMP_FLAG_REWARD
    global WHITEOUT_REWARD, HM_TAUGHT_REWARD, LEVEL_REWARD, MOVE_REPLACE_REWARD, HM_PAY_MASK
    global TRASH_REWARD, TRASH_UNLOCK_REWARD, MANSION_SWITCH_REWARD, SPINNER_REWARD, WARP_PAD_REWARD, INTERACT_REWARD
    hm_pay = parse_hm_reward_moves(hm_reward_moves)
    events = parse_skill_event_reward(skill_event_reward)
    cooldowns = parse_skill_event_cooldown(skill_event_cooldown)
    badges = parse_badge_reward(badge_reward)
    temp = _weight("temp flag reward", temp_flag_reward) or 0.0
    elite4 = _weight("elite4 reward", elite4_reward)
    champion = _weight("champion reward", champion_reward)
    tile = _weight("tile reward", tile_reward)
    tile_count_w = _weight("tile count reward", tile_count_reward)
    tile_scale = None if tile_count_scale is None else float(tile_count_scale)
    if tile_scale is not None and not 1.0 <= tile_scale < float("inf"):
        raise ValueError(f"tile count scale {tile_scale}: must be >= 1 and finite")
    map_w = _weight("map reward", map_reward)
    whiteout = None if whiteout_reward is None else float(whiteout_reward)
    if whiteout is not None and not whiteout <= 0:
        raise ValueError(f"whiteout reward {whiteout}: must be <= 0 (a penalty)")
    obj = [_weight(k, v) or 0.0 for k, v in (("trash reward", trash_reward), ("mansion switch reward", mansion_switch_reward),
                                             ("spinner reward", spinner_reward), ("warp pad reward", warp_pad_reward))]
    unlock = _weight("trash unlock reward", trash_unlock_reward)
    interact = _weight("interact reward", interact_reward) or 0.0
    hm_taught = _weight("hm taught reward", hm_taught_reward)
    level_w = _weight("level reward", level_reward)
    move_replace_w = _weight("move replace reward", move_replace_reward)
    SKILL_REWARD = float(1.0 if skill_reward is None else skill_reward)
    SKILL_WEIGHTS[:] = np.float32(SKILL_REWARD)
    SKILL_WEIGHTS[SKILL_DEFAULT_OFF_IDX] = 0
    if skill_menu_reward is not None:
        SKILL_WEIGHTS[SKILL_MENU_IDX] = np.float32(skill_menu_reward)
    for k, v in events.items():
        SKILL_WEIGHTS[SKILL_EVENTS.index(k)] = np.float32(v)
    SKILL_COOLDOWNS[:] = SKILL_COOLDOWN
    for k, v in cooldowns.items():
        SKILL_COOLDOWNS[SKILL_EVENTS.index(k)] = v
    TILE_REWARD = float(0.02 if tile is None else tile)
    TILE_COUNT_REWARD = float(0.0 if tile_count_w is None else tile_count_w)
    TILE_COUNT_SCALE = float(1.0 if tile_scale is None else tile_scale)
    MAP_REWARD = float(MAP_REWARD_DEFAULT if map_w is None else map_w)
    WHITEOUT_REWARD = float(WHITEOUT_REWARD_DEFAULT if whiteout is None else whiteout)
    HM_TAUGHT_REWARD = float(HM_TAUGHT_REWARD_DEFAULT if hm_taught is None else hm_taught)
    LEVEL_REWARD = float(LEVEL_REWARD_DEFAULT if level_w is None else level_w)
    MOVE_REPLACE_REWARD = float(move_replace_w or 0.0)
    HM_PAY_MASK = HM_ALL if hm_pay is None else sum(1 << HM_NAMES.index(h) for h in hm_pay)
    HM_UNPAID_MOVE[:] = False
    for _i, _m in enumerate(T.HM_MOVES):
        HM_UNPAID_MOVE[_m] = not (HM_PAY_MASK >> _i) & 1
    w = _BASE_WEIGHTS.copy()
    w[_STORY] = STORY_WEIGHT
    w[_ITEM_BALL] = ITEM_BALL_WEIGHT
    w[_cats == T.FLAG_CAT_TRAINER] = TRAINER_WEIGHT if trainer_weight is None else float(trainer_weight)
    if badges is not None:
        w[BADGE_FLAGS] = badges
    if elite4 is not None:
        w[ELITE4_FLAGS] = elite4
    if champion is not None:
        w[T.FLAG_DEFEATED_CHAMP] = champion
    TEMP_FLAG_REWARD = temp
    w[TEMP_FLAGS] = temp
    TRASH_REWARD, MANSION_SWITCH_REWARD, SPINNER_REWARD, WARP_PAD_REWARD = obj
    OBJ_WEIGHTS[:] = np.asarray(obj, dtype=np.float32)
    TRASH_UNLOCK_REWARD = unlock
    if unlock is not None:
        w[TRASH_UNLOCK_FLAG] = unlock
    VAR_WEIGHTS[:] = np.float32(VAR_REWARD)
    INTERACT_REWARD = interact
    INTERACT_WEIGHTS[0] = np.float32(interact)
    FLAG_WEIGHTS[:] = w
    _flag_matrix()


def _flag_matrix() -> None:
    """FLAG_MATRIX from FLAG_WEIGHTS: the category columns, the temp flags (category NONE) in the ``flags`` column,
    column 4 = weight > 0."""
    for _j, _c in enumerate((T.FLAG_CAT_STORY, T.FLAG_CAT_TRAINER, T.FLAG_CAT_BADGE, T.FLAG_CAT_ELITE4)):
        FLAG_MATRIX[:, _j] = np.where(_cats == _c, FLAG_WEIGHTS, 0)
    FLAG_MATRIX[TEMP_FLAGS, 0] = FLAG_WEIGHTS[TEMP_FLAGS]  # temp flags count in ``flags``
    FLAG_MATRIX[:, 4] = FLAG_WEIGHTS > 0


configure()


def summary() -> str:
    """The weights now applied, for the trainer's startup line."""
    badges = ",".join(f"{v:g}" for v in FLAG_WEIGHTS[BADGE_FLAGS])
    e4 = sorted(set(float(v) for v in FLAG_WEIGHTS[ELITE4_FLAGS[:4]]))
    champ = float(FLAG_WEIGHTS[T.FLAG_DEFEATED_CHAMP])
    ev = ", ".join(f"{k}={float(SKILL_WEIGHTS[i]):g}" for i, k in enumerate(SKILL_EVENTS) if SKILL_WEIGHTS[i] > 0)
    cds = [f"{k}={int(SKILL_COOLDOWNS[i])}" for i, k in enumerate(SKILL_EVENTS) if SKILL_COOLDOWNS[i] != SKILL_COOLDOWN]
    if cds:
        ev += f" (cooldown {SKILL_COOLDOWN} steps; {', '.join(cds)})"
    return (f"temp flags {TEMP_FLAG_REWARD:g}; badges {badges}; elite4 {','.join(f'{v:g}' for v in e4)}; champion "
            f"{champ:g}; tile floor {TILE_REWARD:g}, count bonus {TILE_COUNT_REWARD:g} / sqrt(max(N, "
            f"{TILE_COUNT_SCALE:g}) / {TILE_COUNT_SCALE:g}); map {MAP_REWARD:g}; whiteout {WHITEOUT_REWARD:g}; levels "
            f"{LEVEL_REWARD:g} (capped); hm taught {HM_TAUGHT_REWARD:g} ({','.join(_hm_paid())}); move replace "
            f"{MOVE_REPLACE_REWARD:g} (once per mon and move); skill events {ev}; trash {TRASH_REWARD:g} (unlock "
            f"{float(FLAG_WEIGHTS[TRASH_UNLOCK_FLAG]):g}); mansion switches {MANSION_SWITCH_REWARD:g}; spinners "
            f"{SPINNER_REWARD:g}; warp pads {WARP_PAD_REWARD:g}; interactions {INTERACT_REWARD:g}")


def _hm_paid() -> Tuple[str, ...]:
    """The HMs that pay hm_cover / hm_taught now (HM_PAY_MASK)."""
    return tuple(h for i, h in enumerate(HM_NAMES) if (HM_PAY_MASK >> i) & 1)


def reward_sum(terms: np.ndarray) -> np.ndarray:
    """float32 [k]: the rewards of term rows ``terms`` (float32 [k, N_TERMS]) summed as the native code does
    (``TERM_SUM_COL``: numpy's pairwise order over v76's 26 term columns)."""
    a = np.zeros((terms.shape[0], SUM_COLS), dtype=np.float32)
    a[:, TERM_SUM_COL] = terms
    return a.sum(axis=1)


def move_learn_pass(rw, obs, out: np.ndarray) -> Optional[np.ndarray]:
    """Term 10, move_learn, into ``out[:, MOVE_LEARN_TERM]`` (float32, set only on the returned rows: the in-game
    rows with a move learnt in place of a forgotten one; None while its weight is 0). ``rw``: the NativeReward (its
    ``ml_keys`` / ``ml_n``: the (mon, move) keys paid in the life)."""
    if MOVE_REPLACE_REWARD <= 0:
        return None
    ro = obs.dtype.fields["learned_replaced_priv"][1]
    if obs.flags.c_contiguous and obs.shape[0]:  # (almost every step: the one replaced byte per record)
        b = np.ndarray((obs.shape[0],), dtype=np.uint8, buffer=obs, offset=ro, strides=(obs.dtype.itemsize,))
        rows = np.flatnonzero(b)
    else:
        rows = np.flatnonzero(obs["learned_replaced_priv"])
    if not rows.size:
        return rows
    # vectorised over the learnt (row, slot) events, in row-major order (np.nonzero)
    learned = obs["learned_move_priv"][rows]
    hit = (learned != 0) & (learned < T.MOVES_COUNT) & ((obs["valid"][rows] & VALID_INGAME) != 0)[:, None]
    hit_any = hit.any(axis=1)
    rows, learned, hit = rows[hit_any], learned[hit_any], hit[hit_any]
    if not rows.size:
        return rows
    ri, si = np.nonzero(hit)
    mv = learned[ri, si].astype(np.uint64)
    rep = (obs["learned_replaced_priv"][rows][ri].astype(np.int64) >> si) & 1
    w = np.where(rep != 0, MOVE_REPLACE_REWARD, 0.0)
    w = np.where(HM_UNPAID_MOVE[mv], 0.0, w)  # (an HM move outside hm_reward_moves pays nothing)
    keep = w > 0
    if keep.any():  # once per (mon, move) and life
        er = rows[ri]  # env row of each event
        key = (obs["party"]["pid_priv"][rows][ri, si].astype(np.uint64) << np.uint64(16)) | mv
        n_ = rw.ml_n[er]
        seen = ((rw.ml_keys[er] == key[:, None]) & (np.arange(ML_KEYS)[None, :] < n_[:, None])).any(axis=1)
        idx = np.flatnonzero(keep & ~seen)
        if idx.size:  # (a key twice in one row and step pays once: the first event)
            _, first = np.unique(np.stack([er[idx].astype(np.uint64), key[idx]], 1), axis=0, return_index=True)
            idx = idx[np.sort(first)]
            rr = er[idx]
            rank = np.arange(idx.size) - np.searchsorted(rr, rr, side="left")  # (idx is row-major: rr sorted)
            pos = rw.ml_n[rr] + rank
            ok = pos < ML_KEYS
            rw.ml_keys[rr[ok], pos[ok]] = key[idx[ok]]
            np.add.at(rw.ml_n, rr[ok], 1)
            idx = idx[ok]
        keep = np.zeros(ri.size, dtype=bool)
        keep[idx] = True
    vals = np.zeros(rows.size, dtype=np.float64)
    np.add.at(vals, ri[keep], w[keep])
    out[rows, MOVE_LEARN_TERM] = vals
    return rows
