#!/usr/bin/env python3
"""Generates ``rl/pokerl/tables.py`` from the game source: the tables of the reward and the milestone events.

    PYTHONPATH=rl python3 -m pokerl.gen_tables            # rewrite tables.py
    PYTHONPATH=rl python3 -m pokerl.gen_tables --check    # exit 1 if tables.py is out of date

Reads (constants: ``#define`` lines evaluated recursively):
game/include/constants/{species,moves,items,flags,vars,map_groups,opponents,battle}.h,
game/src/data/pokemon/tmhm_learnsets.h (HM01-HM07 per species), game/src/data/trainers.h +
game/src/data/trainer_parties.h (ace level = highest level in the party; party size = number of mons, reward
v0.5), game/src/data/items.json (item pockets) + game/data/maps/*/scripts.inc + game/data/scripts/*.inc (item-ball
items and trigger variables, reward v0.7), game/data/maps/*/map.json +
game/data/layouts/layouts.json (map sizes, for the per-env visited-tile bitmap); for the puzzle terms:
map.json bg_events (the Vermilion Gym trash cans, the Pokémon Mansion switch statues) and object_events (the
Strength boulders' start positions), the metatile behaviours of the layouts through atlas.py (spinner
tiles, warp pads); for the interaction term: map.json object_events (every non-clone object's local id) and
bg_events (every sign / hidden item position), the overworld interaction term's keys. Every size and the ace ladder
are asserted against known values.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GAME = os.path.join(ROOT, "game")
OUT = os.path.join(HERE, "tables.py")

HEADERS = ["species.h", "moves.h", "items.h", "flags.h", "vars.h", "map_groups.h", "opponents.h", "battle.h"]
DEFINE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?://.*)?$")


class Constants:
    """``#define NAME expr`` of the constant headers, evaluated on demand (digits/operators only)."""

    def __init__(self):
        self.raw = {}
        for h in HEADERS:
            with open(os.path.join(GAME, "include", "constants", h), encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = DEFINE.match(line)
                    if m and "(" not in m.group(1):
                        self.raw.setdefault(m.group(1), m.group(2).split("/*")[0].strip())
        self.cache = {}

    def value(self, name, depth=0):
        if name in self.cache:
            return self.cache[name]
        if depth > 40:
            raise ValueError(f"constant {name}: recursion")
        expr = self.raw.get(name)
        if expr is None:
            raise KeyError(f"unknown game constant {name}")
        v = self.eval(expr, depth)
        self.cache[name] = v
        return v

    def eval(self, expr, depth=0):
        def sub(m):
            tok = m.group(0)
            if re.match(r"^(0x[0-9a-fA-F]+|\d+)[uUlL]*$", tok):
                return str(int(re.sub(r"[uUlL]+$", "", tok), 0))
            return str(self.value(tok, depth + 1))
        py = re.sub(r"0x[0-9a-fA-F]+[uUlL]*|\b\d+[uUlL]*\b|\b[A-Za-z_]\w*\b", sub, expr)
        if not re.match(r"^[\d\s()+\-*/|&<>~^]+$", py):
            raise ValueError(f"cannot evaluate {expr!r}")
        return int(eval(py.replace("/", "//")))  # noqa: S307 (digits and operators only)

    def names(self, prefix):
        out = {}
        for n in self.raw:
            if n.startswith(prefix):
                try:
                    out[n] = self.value(n)
                except (KeyError, ValueError):
                    pass
        return out


# Gym leaders in badge order (FLAG_BADGE01_GET .. FLAG_BADGE08_GET: Boulder, Cascade, Thunder, Rainbow,
# Soul, Marsh, Volcano, Earth), then the Elite Four in battle order, then the Champion.
LEADERS = ["BROCK", "MISTY", "LT_SURGE", "ERIKA", "KOGA", "SABRINA", "BLAINE", "GIOVANNI"]
ELITE_FOUR = ["LORELEI", "BRUNO", "AGATHA", "LANCE"]
CHAMPION_TRAINERS = ["TRAINER_CHAMPION_FIRST_SQUIRTLE", "TRAINER_CHAMPION_FIRST_BULBASAUR",
                     "TRAINER_CHAMPION_FIRST_CHARMANDER"]
EXPECTED_ACES = [14, 21, 24, 29, 43, 43, 47, 50, 63]  # RL_BASELINE § 3, in LADDER order (v0.7: no Elite Four)
EXPECTED_PARTY_SIZES = [2, 2, 3, 3, 4, 4, 4, 5, 6]  # RL_BASELINE § 3 (reward v0.5), in LADDER order
HM_NAMES = ["HM01_CUT", "HM02_FLY", "HM03_SURF", "HM04_STRENGTH", "HM05_FLASH", "HM06_ROCK_SMASH", "HM07_WATERFALL"]
FAMILY_COUNT_EXPECTED = 202  # evolutionary family roots among the real species (behaviour novelty)
HM_MOVE_NAMES = ["MOVE_CUT", "MOVE_FLY", "MOVE_SURF", "MOVE_STRENGTH", "MOVE_FLASH", "MOVE_ROCK_SMASH", "MOVE_WATERFALL"]
EXPECTED_HM_MOVES = [15, 19, 57, 70, 148, 249, 127]

# system flags that keep weight 1 (reward v0.1); every other FLAG_SYS_* flag, FLAG_OPENED_START_MENU,
# every FLAG_WORLD_MAP_* and every FLAG_VISITED_* flag has weight 0 (state toggles, menu use, map-visit
# duplicates of the map reward; RL_BASELINE § Terms "Reward v0.1")
SYS_FLAGS_KEPT = ["FLAG_SYS_POKEMON_GET", "FLAG_SYS_POKEDEX_GET", "FLAG_SYS_B_DASH", "FLAG_SYS_NATIONAL_DEX",
                  "FLAG_SYS_GAME_CLEAR", "FLAG_SYS_GOT_BERRY_POUCH", "FLAG_SYS_SEVII_MAP_123",
                  "FLAG_SYS_SEVII_MAP_4567"]
# FLAG_PN_BOULDERS_SOLVED (= FLAG_UNUSED_0x4A7): fireredsim's new-game preset flag (Route 23 keeps the Victory Road
# switch variables); never set by the game, so it is no progress (weight 0, category none: not in the flags term or
# flags_earned)
ZERO_FLAGS = ["FLAG_OPENED_START_MENU", "FLAG_PN_BOULDERS_SOLVED"]
ZERO_FLAG_PREFIXES = ["FLAG_SYS_", "FLAG_WORLD_MAP_", "FLAG_VISITED_"]
VARS_START = 0x4000

# the base flag weights (reward.configure sets the trainer, badge and Elite Four weights)
STORY_WEIGHT = 1
TRAINER_WEIGHT = 0.5
ITEM_BALL_WEIGHT = 0.25
BADGE_WEIGHT = 20
ELITE4_WEIGHT = 20  # each Elite Four member and the Champion

# an item ball (map object with OBJ_EVENT_GFX_ITEM_BALL) whose script gives a non-key item has weight
# ITEM_BALL_WEIGHT (v0.10; 0 in v0.7-v0.9) and counts as a story flag; one that gives a key item (POCKET_KEY_ITEMS:
# Lift Key, Silph Scope, Card Key, Secret Key, Gold Teeth) or no item (a gift or static Pokémon shown as a ball: the
# starters, Eevee, the Dojo Hitmons, the Power Plant Electrodes) keeps the story weight
ITEM_BALLS_EXPECTED = (178, 165, 5)  # (item balls, non-key item balls, key items)

# reward v0.7: the story variables are the trigger variables (map_script_2 conditions of map scripts and
# coord_events "var" of map.json), minus the counters / link-state variables excluded below (was: every
# VAR_MAP_SCENE_*). Rule-based so it carries over to other Gen 3 decomps.
STORY_VARS_EXCLUDED = ["VAR_CABLE_CLUB_STATE"]  # link-cable state (Two Island minigame room), not story
STORY_VARS_EXPECTED = 49

# flag categories (index into the per-term split of the flag reward)
FLAG_CAT_NONE, FLAG_CAT_STORY, FLAG_CAT_TRAINER, FLAG_CAT_BADGE, FLAG_CAT_ELITE4 = 0, 1, 2, 3, 4

# object tiles of the puzzle terms and the boulder puzzles' events. Kinds and end conditions as written to tables.OBJECT_TILES.
OBJ_TRASH_CAN, OBJ_MANSION_SWITCH, OBJ_SPINNER, OBJ_WARP_PAD = 0, 1, 2, 3
OBJ_END_NONE, OBJ_END_FLAG, OBJ_END_KEY_ITEM = 0, 1, 2
# spinner tiles (MB_SPIN_*) per map and what ends their pay: the Rocket Hideout's once the Silph Scope is held, the
# Viridian Gym's once its badge is won (the Five Island Rocket Warehouse's 116 spinners are post-game: not listed)
SPINNER_MAPS = {"MAP_ROCKET_HIDEOUT_B1F": ("item", "ITEM_SILPH_SCOPE"), "MAP_ROCKET_HIDEOUT_B2F": ("item", "ITEM_SILPH_SCOPE"),
                "MAP_ROCKET_HIDEOUT_B3F": ("item", "ITEM_SILPH_SCOPE"), "MAP_ROCKET_HIDEOUT_B4F": ("item", "ITEM_SILPH_SCOPE"),
                "MAP_VIRIDIAN_CITY_GYM": ("flag", "FLAG_BADGE08_GET")}
# warp pads (a warp event on an MB_REGULAR_WARP tile): Silph Co.'s until its Giovanni is beaten (his trainer flag),
# the Saffron Gym's until its badge
WARP_PAD_MAPS = {**{f"MAP_SILPH_CO_{f}F": ("trainer", "TRAINER_BOSS_GIOVANNI_2") for f in range(2, 12)},
                 "MAP_SAFFRON_CITY_GYM": ("flag", "FLAG_BADGE06_GET")}
TRASH_CAN_MAP, TRASH_CAN_END = "MAP_VERMILION_CITY_GYM", "FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES"
MANSION_MAPS = ["MAP_POKEMON_MANSION_1F", "MAP_POKEMON_MANSION_2F", "MAP_POKEMON_MANSION_3F", "MAP_POKEMON_MANSION_B1F"]
OBJECT_TILES_EXPECTED = (15, 5, 70, 64)  # trash cans, switch statues, spinners (B2F 42 + B3F 16 + gym 12), pads (34 + 30)
BOULDER_GFX = "OBJ_EVENT_GFX_PUSHABLE_BOULDER"
BOULDERS_EXPECTED = 58  # Seafoam, Victory Road, Sevii caves (every PUSHABLE_BOULDER object)
# the boulder puzzles' completion events: the Seafoam currents stopped (flags), the Victory Road floor switches
# (story variables set to 100 by the FloorSwitch scripts)
BOULDER_FLAGS = ["FLAG_STOPPED_SEAFOAM_B3F_CURRENT", "FLAG_STOPPED_SEAFOAM_B4F_CURRENT"]
BOULDER_VARS = ["VAR_MAP_SCENE_VICTORY_ROAD_1F", "VAR_MAP_SCENE_VICTORY_ROAD_2F_BOULDER1",
                "VAR_MAP_SCENE_VICTORY_ROAD_2F_BOULDER2", "VAR_MAP_SCENE_VICTORY_ROAD_3F"]

# the overworld interaction term's keys. An
# object's local id is its 1-based position in its map's object_events, clones counted (tools/mapjson emits
# ``object_event <i + 1>`` / ``clone_event <i + 1>``: the template's localId); a clone spawns with its original's map
# and local id (event_object_movement.c InitObjectEventStateFromTemplate), so clones get no key of their own.
# Background events: every bg_events type of map.json, both of which GetInteractedBackgroundEventScript selects (sign
# = BG_EVENT_PLAYER_FACING_*, hidden_item = BG_EVENT_HIDDEN_ITEM; no two share a position). The reward v0.13 trash
# cans and switch statues are signs like any other here (they may also pay through the puzzle terms).
INTERACT_BG_TYPES = ("sign", "hidden_item")
INTERACT_OBJECTS_EXPECTED = (1639, 9)  # object events (local ids), clones (no key)
INTERACT_BG_EXPECTED = (519, 183)  # signs (the 15 cans and 5 statues among them), hidden items (6 of them underfoot)


def party_levels():
    """{sParty_X: [levels]} of game/src/data/trainer_parties.h (one level per mon of the party, in order)."""
    text = open(os.path.join(GAME, "src", "data", "trainer_parties.h"), encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"static const struct \w+ (sParty_\w+)\[\]\s*=\s*\{(.*?)\n\};", text, re.S):
        levels = [int(x) for x in re.findall(r"\.lvl\s*=\s*(\d+)", m.group(2))]
        n_species = len(re.findall(r"\.species\s*=", m.group(2)))
        assert len(levels) == n_species, (m.group(1), len(levels), n_species)  # party size = entries of the array
        out[m.group(1)] = levels
    return out


def trainer_parties():
    """{TRAINER_X: sParty_Y} of game/src/data/trainers.h."""
    text = open(os.path.join(GAME, "src", "data", "trainers.h"), encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"\[(TRAINER_\w+)\]\s*=\s*\{(.*?)\n    \},", text, re.S):
        p = re.search(r"\.party\s*=\s*\w+\((sParty_\w+)\)", m.group(2))
        if p:
            out[m.group(1)] = p.group(1)
    return out


def hm_learnsets(c: Constants, species_count: int):
    """species id -> 7-bit mask (bit i = HM0{i+1} learnable) from tmhm_learnsets.h."""
    text = open(os.path.join(GAME, "src", "data", "pokemon", "tmhm_learnsets.h"), encoding="utf-8").read()
    masks = [0] * species_count
    seen = 0
    for m in re.finditer(r"\[(SPECIES_\w+)\]\s*=\s*TMHM_LEARNSET\((.*?)\),\s*\n", text, re.S):
        sid = c.value(m.group(1))
        body = m.group(2)
        bits = 0
        for i, hm in enumerate(HM_NAMES):
            if re.search(r"TMHM\(%s\)" % hm, body):
                bits |= 1 << i
        masks[sid] = bits
        seen += 1
    return masks, seen


def tmhm_learnsets(c: Constants, species_count: int):
    """species id -> 58-bit mask of the TM/HM items it can learn (bit i = item TM_FIRST_ITEM + i, i.e. TM01..TM50,
    HM01..HM08) from tmhm_learnsets.h (TMHM(x) = bit
    ITEM_x - ITEM_TM01_FOCUS_PUNCH); species without an entry: 0."""
    text = open(os.path.join(GAME, "src", "data", "pokemon", "tmhm_learnsets.h"), encoding="utf-8").read()
    first = c.value("ITEM_TM01_FOCUS_PUNCH")
    masks = [0] * species_count
    for m in re.finditer(r"\[(SPECIES_\w+)\]\s*=\s*TMHM_LEARNSET\((.*?)\),\s*\n", text, re.S):
        bits = 0
        for name in re.findall(r"TMHM\((\w+)\)", m.group(2)):
            b = c.value("ITEM_" + name) - first
            assert 0 <= b < 58, name
            bits |= 1 << b
        masks[c.value(m.group(1))] = bits
    return masks


def evolution_parents(c: Constants):
    """{species id: the species it evolves from} of game/src/data/pokemon/evolution.h (every method: level, item,
    friendship, trade, ...); a species with two parents raises."""
    text = open(os.path.join(GAME, "src", "data", "pokemon", "evolution.h"), encoding="utf-8").read()
    parent = {}
    for m in re.finditer(r"\[(SPECIES_\w+)\]\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text):
        src = c.value(m.group(1))
        for t in re.findall(r"\{\s*EVO_\w+\s*,\s*[^,{}]+?\s*,\s*(SPECIES_\w+)\s*\}", m.group(2)):
            dst = c.value(t)
            assert parent.get(dst, src) == src, (t, parent.get(dst), src)
            parent[dst] = src
    return parent


def families(c: Constants, species_count: int, species_egg: int):
    """FAMILY (species id -> species id of its evolutionary family root: pre-evolutions followed to the root,
    species without one are their own root) for the real species 1 .. NUM_SPECIES - 1; 0 for SPECIES_NONE, the
    Egg and the unused SPECIES_OLD_UNOWN_* placeholders (never in a record: no family); the Unown form ids above
    the Egg (graphics only) -> SPECIES_UNOWN. Returns (FAMILY, the sorted roots)."""
    parent = evolution_parents(c)
    old_unown = set(c.names("SPECIES_OLD_UNOWN_").values())
    unown = c.value("SPECIES_UNOWN")
    fam = [0] * species_count
    for s in range(1, species_count):
        if s == species_egg or s in old_unown:
            continue
        r, hops = (unown if s > species_egg else s), 0
        while r in parent:
            r, hops = parent[r], hops + 1
            assert hops < 4, s
        fam[s] = r
    roots = sorted(set(fam) - {0})
    assert all(fam[r] == r for r in roots)
    return fam, roots


def id_names(c: Constants, prefix: str, count: int, skip=()):
    """[name without the prefix] per id 0 .. count - 1: the first ``#define prefix*`` of the headers with that value
    (header order; aliases after it are ignored), "" for ids without one."""
    out = [""] * count
    for n in c.raw:
        if not n.startswith(prefix) or n in skip:
            continue
        try:
            v = c.value(n)
        except (KeyError, ValueError):
            continue
        if 0 <= v < count and not out[v]:
            out[v] = n[len(prefix):]
    return out


def map_table(c: Constants):
    """Sorted [(key = group*256 + num, name, width, height)] of every MAP_* constant."""
    maps = {n: v for n, v in c.names("MAP_").items() if n != "MAP_GROUPS_COUNT"}
    layouts = {}
    for lay in json.load(open(os.path.join(GAME, "data", "layouts", "layouts.json")))["layouts"]:
        if lay and "id" in lay:
            layouts[lay["id"]] = (int(lay["width"]), int(lay["height"]))
    dims = {}
    for path in glob.glob(os.path.join(GAME, "data", "maps", "*", "map.json")):
        m = json.load(open(path))
        dims[m["id"]] = layouts[m["layout"]]
    out = []
    for name, v in maps.items():
        num, group = v & 0xFF, v >> 8
        w, h = dims[name]
        out.append((group * 256 + num, name, w, h))
    out.sort()
    return out


def _script_labels():
    """{label: body} of every script label in game/data/maps/*/scripts.inc and game/data/scripts/*.inc."""
    out = {}
    paths = sorted(glob.glob(os.path.join(GAME, "data", "maps", "*", "scripts.inc")) +
                   glob.glob(os.path.join(GAME, "data", "scripts", "*.inc")))
    for path in paths:
        text = open(path, encoding="utf-8").read()
        for m in re.finditer(r"^(\w+)::?[ \t]*\n(.*?)(?=^\w+::?[ \t]*\n|\Z)", text, re.S | re.M):
            out.setdefault(m.group(1), m.group(2))
    return out


POCKET_IDX = {"POCKET_ITEMS": 1, "POCKET_KEY_ITEMS": 2, "POCKET_POKE_BALLS": 3, "POCKET_TM_CASE": 4,
              "POCKET_BERRY_POUCH": 5}
TM_FIRST_ITEM_NAME, TM_LAST_ITEM_NAME = "ITEM_TM01", "ITEM_HM08"


def item_pockets(c, items_count):
    """item id -> pocket index (POCKET_IDX; 0 for ITEM_NONE / unlisted) from items.json."""
    out = [0] * items_count
    for it in json.load(open(os.path.join(GAME, "src", "data", "items.json")))["items"]:
        if it["itemId"] != "ITEM_NONE":  # the unused slots of the table are ITEM_NONE entries
            out[c.value(it["itemId"])] = POCKET_IDX[it["pocket"]]
    return out


def move_stats(c, moves_count):
    """per move id (type, power, accuracy, pp, priority) from game/src/data/battle_moves.h: static move
    numbers."""
    with open(os.path.join(GAME, "include", "constants", "pokemon.h"), encoding="utf-8") as f:
        types = dict(re.findall(r"#define (TYPE_\w+)\s+(\d+)", f.read()))
    with open(os.path.join(GAME, "src", "data", "battle_moves.h"), encoding="utf-8") as f:
        text = f.read()
    out = [(0, 0, 0, 0, 0)] * moves_count
    seen = 0
    for name, body in re.findall(r"\[(MOVE_\w+)\] =\s*\{(.*?)\n    \}", text, re.S):
        field = lambda k: re.search(r"\." + k + r" = (-?\w+)", body).group(1)  # noqa: E731
        out[c.value(name)] = (int(types[field("type")]), int(field("power")), int(field("accuracy")), int(field("pp")),
                              int(field("priority")))
        seen += 1
    assert seen == moves_count, seen
    assert out[c.value("MOVE_SURF")] == (11, 95, 100, 15, 0), out[c.value("MOVE_SURF")]
    assert out[c.value("MOVE_QUICK_ATTACK")] == (0, 40, 100, 30, 1), out[c.value("MOVE_QUICK_ATTACK")]
    return out


def species_base_stats(c, species_count):
    """per species id (base HP, Attack, Defense, Speed, Sp. Attack, Sp. Defense) from species_info.h: what the entity
    network knows of an opponent, whose stats are hidden. All 0 for the entries without their own block (the unused ids)."""
    with open(os.path.join(GAME, "src", "data", "pokemon", "species_info.h"), encoding="utf-8") as f:
        text = f.read()
    out = [(0, 0, 0, 0, 0, 0)] * species_count
    seen = 0
    for name, body in re.findall(r"\[(SPECIES_\w+)\] =\s*\n\s*\{(.*?)\n    \}", text, re.S):
        out[c.value(name)] = tuple(int(re.search(r"\.base" + k + r" = (\d+)", body).group(1))
                                   for k in ("HP", "Attack", "Defense", "Speed", "SpAttack", "SpDefense"))
        seen += 1
    assert seen >= 386, seen
    assert out[c.value("SPECIES_BULBASAUR")] == (45, 49, 49, 45, 65, 65), out[c.value("SPECIES_BULBASAUR")]
    return out


def tm_moves(c):
    """move id of TM01..TM50, HM01..HM08 (party_menu.c ItemIdToBattleMoveId: sTMHMMoves of
    game/src/data/party_menu.h, indexed by item - ITEM_TM01)."""
    text = open(os.path.join(GAME, "src", "data", "party_menu.h"), encoding="utf-8").read()
    body = re.search(r"static const u16 sTMHMMoves\[\] =\s*\{(.*?)\};", text, re.S).group(1)
    names = re.findall(r"MOVE_\w+", body)
    assert len(names) == 58, len(names)
    return [c.value(n) for n in names]


def _map_jsons():
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(GAME, "data", "maps", "*", "map.json")))]


def item_balls(labels):
    """(non_key, key): the flag names of the item balls whose script gives a non-key item (``ITEM_BALL_WEIGHT``;
    v0.7-v0.9: weight 0) and of those giving a key item; balls giving no item (a Pokémon) are in neither. Also the ball count."""
    pockets = {it["itemId"]: it["pocket"]
               for it in json.load(open(os.path.join(GAME, "src", "data", "items.json")))["items"]}
    zero, key, n = set(), set(), 0
    for m in _map_jsons():
        for o in m.get("object_events") or []:
            if o.get("graphics_id") != "OBJ_EVENT_GFX_ITEM_BALL":
                continue
            n += 1
            items = re.findall(r"^\s*(?:finditem|giveitem)\s+(ITEM_\w+)", labels[o["script"]], re.M)
            if any(pockets[i] == "POCKET_KEY_ITEMS" for i in items):
                key.add(o["flag"])
            elif items:
                zero.add(o["flag"])
    return sorted(zero), sorted(key), n


def trigger_vars(labels):
    """Names of the variables map scripts trigger on: map_script_2 conditions and coord_events vars."""
    out = set()
    for m in _map_jsons():
        for e in m.get("coord_events") or []:
            if str(e.get("var", "")).startswith("VAR_"):
                out.add(e["var"])
    for body in labels.values():
        out.update(re.findall(r"^\s*map_script_2\s+(VAR_\w+)", body, re.M))
    return out


def generate():
    c = Constants()
    species = c.names("SPECIES_")
    species_max = max(v for n, v in species.items() if n != "NUM_SPECIES")
    species_egg = c.value("SPECIES_EGG")
    moves_count = c.value("MOVES_COUNT")
    items_count = c.value("ITEMS_COUNT")
    flags_count = c.value("FLAGS_COUNT")
    assert species_max == 439, species_max
    assert species_egg == 412, species_egg
    assert moves_count == 355, moves_count
    assert items_count == 375, items_count
    assert flags_count == 0x900, flags_count

    hm_moves = [c.value(n) for n in HM_MOVE_NAMES]
    assert hm_moves == EXPECTED_HM_MOVES, hm_moves
    item_pocket = item_pockets(c, items_count)
    assert item_pocket[c.value("ITEM_POTION")] == 1 and item_pocket[c.value("ITEM_BICYCLE")] == 2
    assert item_pocket[c.value("ITEM_POKE_BALL")] == 3 and item_pocket[c.value("ITEM_TM01")] == 4
    assert item_pocket[c.value("ITEM_CHERI_BERRY")] == 5 and item_pocket[0] == 0
    tm_first, tm_last = c.value(TM_FIRST_ITEM_NAME), c.value(TM_LAST_ITEM_NAME)
    assert (tm_first, tm_last) == (289, 346)
    tmm = tm_moves(c)
    assert tmm[50 + 0] == c.value("MOVE_CUT") and tmm[50 + 1] == c.value("MOVE_FLY")  # HM01 Cut, HM02 Fly
    hm_learn, n_learn = hm_learnsets(c, species_max + 1)
    assert n_learn == species_egg, n_learn  # SPECIES_NONE .. SPECIES_CHIMECHO (411); Egg and Unown forms: 0
    for name, want in (("SPECIES_BULBASAUR", 1 << 0), ("SPECIES_PIDGEY", 1 << 1), ("SPECIES_SQUIRTLE", 1 << 2)):
        assert hm_learn[c.value(name)] & want, name
    assert hm_learn[c.value("SPECIES_MAGIKARP")] == 0
    # families, TM/HM learnsets, names
    tm_learn = tmhm_learnsets(c, species_max + 1)
    for i, name in enumerate(HM_NAMES):  # (the 7-bit HM_LEARN is the HM01-HM07 bits 50-56 of TM_LEARN)
        assert all(((tm_learn[s] >> (50 + i)) & 1) == ((hm_learn[s] >> i) & 1) for s in range(species_max + 1)), name
    venusaur = c.value("SPECIES_VENUSAUR")
    assert (tm_learn[venusaur] >> 18) & 1 and not (tm_learn[venusaur] >> 38) & 1  # TM19 Giga Drain, no TM39 Rock Tomb
    family, roots = families(c, species_max + 1, species_egg)
    assert len(roots) == FAMILY_COUNT_EXPECTED, len(roots)
    species_names = id_names(c, "SPECIES_", species_max + 1, skip=("SPECIES_SHINY_TAG",))
    move_names = id_names(c, "MOVE_", moves_count, skip=("MOVES_COUNT",))
    assert species_names[c.value("SPECIES_VENUSAUR")] == "VENUSAUR" and species_names[species_egg] == "EGG"
    assert move_names[c.value("MOVE_GIGA_DRAIN")] == "GIGA_DRAIN" and move_names[0] == "NONE"

    # flags
    temp_start, temp_end = c.value("TEMP_FLAGS_START"), c.value("TEMP_FLAGS_END")
    hidden_start = c.value("FLAG_HIDDEN_ITEMS_START")
    hidden_end = c.value("FLAG_HIDDEN_ITEM_SSANNE_EXTERIOR_LAVA_COOKIE")
    hidden_names = [n for n in c.names("FLAG_HIDDEN_ITEM_") if n != "FLAG_HIDDEN_ITEMS_START"]
    assert hidden_end == hidden_start + 190 == max(c.value(n) for n in hidden_names), hidden_end
    trainer_start, trainer_end = c.value("TRAINER_FLAGS_START"), c.value("TRAINER_FLAGS_END")
    assert (temp_start, temp_end, hidden_start, trainer_start, trainer_end) == (0, 0x1F, 0x3E8, 0x500, 0x7FF)
    badge_flags = [c.value(f"FLAG_BADGE0{i}_GET") for i in range(1, 9)]
    assert badge_flags == list(range(0x820, 0x828)), badge_flags
    e4_flags = [c.value(f"FLAG_DEFEATED_{n}") for n in ELITE_FOUR] + [c.value("FLAG_DEFEATED_CHAMP")]
    assert e4_flags == list(range(0x4B8, 0x4BD)), e4_flags
    weights = [STORY_WEIGHT] * flags_count
    cats = [FLAG_CAT_STORY] * flags_count
    for f in range(temp_start, temp_end + 1):
        weights[f], cats[f] = 0, FLAG_CAT_NONE
    for f in range(hidden_start, hidden_end + 1):
        weights[f], cats[f] = 0, FLAG_CAT_NONE
    for f in range(trainer_start, trainer_end + 1):
        weights[f], cats[f] = TRAINER_WEIGHT, FLAG_CAT_TRAINER
    for f in badge_flags:
        weights[f], cats[f] = BADGE_WEIGHT, FLAG_CAT_BADGE
    for f in e4_flags:
        weights[f], cats[f] = ELITE4_WEIGHT, FLAG_CAT_ELITE4
    assert c.value("SYS_FLAGS") == 0x800
    zero = {n: c.value(n) for n in ZERO_FLAGS}
    for pre in ZERO_FLAG_PREFIXES:
        zero.update(c.names(pre))
    for n in SYS_FLAGS_KEPT:
        del zero[n]
    assert c.value("FLAG_VISITED_OAKS_LAB") == 0x2CF and c.value("FLAG_OPENED_START_MENU") == 0x83E
    assert len(c.names("FLAG_WORLD_MAP_")) == 51 and len(c.names("FLAG_SYS_")) == 29
    for n, f in zero.items():
        assert cats[f] == FLAG_CAT_STORY, (n, f)  # never a badge / trainer / Elite Four / temp flag
        weights[f], cats[f] = 0, FLAG_CAT_NONE
    labels = _script_labels()
    ball_zero, ball_key, n_balls = item_balls(labels)
    assert (n_balls, len(ball_zero), len(ball_key)) == ITEM_BALLS_EXPECTED, (n_balls, len(ball_zero), len(ball_key))
    assert "FLAG_HIDE_SILPH_SCOPE" in ball_key and "FLAG_HIDE_ROUTE2_ETHER" in ball_zero
    for n in ball_zero:
        f = c.value(n)
        assert cats[f] == FLAG_CAT_STORY and weights[f] == STORY_WEIGHT, (n, f)
        weights[f] = ITEM_BALL_WEIGHT  # still a story flag (the ``flags`` term)

    # story variables (reward v0.7): the trigger variables of map scripts (vars_priv[i] = var VARS_START + i)
    assert c.value("VARS_START") == VARS_START
    excluded = {"VAR_REPEL_STEP_COUNT", "VAR_ICE_STEP_COUNT", *STORY_VARS_EXCLUDED}
    excluded |= {n for n in c.names("VAR_TEMP_")} | set(c.names("VAR_OBJ_GFX_ID_")) | set(c.names("VAR_QLBAK_"))
    temp_vars = set(range(c.value("TEMP_VARS_START"), c.value("TEMP_VARS_END") + 1))
    story = {n: c.value(n) for n in trigger_vars(labels) - excluded}
    story = {n: v for n, v in story.items() if v not in temp_vars}
    scene_idx = sorted(v - VARS_START for v in story.values())
    assert len(story) == len(set(scene_idx)) == STORY_VARS_EXPECTED, sorted(story)
    assert "VAR_MAP_SCENE_SILPH_CO_11F" in story and "VAR_VERMILION_CITY_TICKET_CHECK_TRIGGER" in story
    assert scene_idx[0] == c.value("VAR_MAP_SCENE_PALLET_TOWN_OAK") - VARS_START == 0x50, scene_idx[0]
    assert scene_idx[-1] < 0x100  # vars_priv holds vars 0x4000..0x40FF

    # ace ladder (and, reward v0.5, the party size of each opponent: the levels term counts that many mons)
    lv = party_levels()
    tp = trainer_parties()
    ace = lambda t: max(lv[tp[t]])  # noqa: E731
    size = lambda t: len(lv[tp[t]])  # noqa: E731
    ladder, sizes = [], []
    for i, n in enumerate(LEADERS):
        ladder.append((n, ace(f"TRAINER_LEADER_{n}"), badge_flags[i]))
        sizes.append(size(f"TRAINER_LEADER_{n}"))
    # reward v0.7: the Elite Four are not on the ladder (after the 8th badge the next opponent is the Champion)
    champ_aces = [ace(t) for t in CHAMPION_TRAINERS]
    champ_sizes = [size(t) for t in CHAMPION_TRAINERS]
    assert len(set(champ_sizes)) == 1, champ_sizes  # the three starter variants: same party size
    ladder.append(("CHAMPION", max(champ_aces), e4_flags[4]))
    sizes.append(champ_sizes[0])
    assert [a for _, a, _ in ladder] == EXPECTED_ACES, ladder
    assert sizes == EXPECTED_PARTY_SIZES, sizes

    maps = map_table(c)
    assert len(maps) == 425, len(maps)
    assert len({k for k, *_ in maps}) == len(maps)

    battle_type_trainer = c.value("BATTLE_TYPE_TRAINER")
    assert battle_type_trainer == 8

    # reward v0.13: object tiles, boulders, the boulder puzzles' events
    keys = {n: k for k, n, _, _ in maps}
    objs = object_tiles(c, keys)
    counts = tuple(sum(1 for o in objs if o[3] == k) for k in range(4))
    assert counts == OBJECT_TILES_EXPECTED, counts
    assert sorted(o[4] for o in objs if o[3] == OBJ_TRASH_CAN) == list(range(1, 16))
    dims = {k: (w, h) for k, _, w, h in maps}
    assert all(0 <= x < dims[k][0] and 0 <= y < dims[k][1] for k, x, y, *_ in objs)
    boulder_pos = boulders(keys)
    assert len(boulder_pos) == BOULDERS_EXPECTED, len(boulder_pos)
    boulder_flags = [c.value(n) for n in BOULDER_FLAGS]
    boulder_vars = [c.value(n) - VARS_START for n in BOULDER_VARS]
    assert all(cats[f] == FLAG_CAT_STORY and weights[f] == STORY_WEIGHT for f in boulder_flags)
    assert all(v in scene_idx for v in boulder_vars), boulder_vars
    trash_flag = c.value(TRASH_CAN_END)
    assert cats[trash_flag] == FLAG_CAT_STORY and weights[trash_flag] == STORY_WEIGHT
    assert c.value("ITEM_SILPH_SCOPE") == 359 and keys[TRASH_CAN_MAP] == 0x0906

    # reward v0.14: the overworld interaction keys
    i_objs, i_bg = interact_keys(keys)
    assert all(1 <= i <= 255 for _, i in i_objs) and all(0 <= x < dims[k][0] and 0 <= y < dims[k][1]
                                                           for k, x, y in i_bg)
    assert {(k, x, y) for k, x, y, kd, *_ in objs if kd in (OBJ_TRASH_CAN, OBJ_MANSION_SWITCH)} <= set(i_bg)

    return dict(item_pocket=item_pocket, tm_first=tm_first, tm_moves=tmm, move_stats=move_stats(c, moves_count),
                species_base=species_base_stats(c, species_max + 1),
                species_count=species_max + 1, species_egg=species_egg, moves_count=moves_count,
                tm_learn=tm_learn, family=family, family_roots=roots, species_names=species_names,
                move_names=move_names,
                items_count=items_count, flags_count=flags_count, hm_moves=hm_moves, hm_learn=hm_learn,
                weights=weights, cats=cats, ladder=ladder, sizes=sizes, maps=maps, badge_flags=badge_flags,
                e4_flags=e4_flags, battle_type_trainer=battle_type_trainer, scene_idx=scene_idx,
                objs=objs, boulders=boulder_pos, boulder_flags=boulder_flags, boulder_vars=boulder_vars,
                trash_flag=trash_flag, trash_map=keys[TRASH_CAN_MAP], interact_objects=i_objs, interact_bg=i_bg)


def _atlas():
    """pokerl.atlas (the layouts' metatile behaviours)."""
    from . import atlas
    return atlas


def object_tiles(c: Constants, keys):
    """[(map key, x, y, kind, id, end kind, end value)] sorted: the Vermilion Gym trash cans (id = can number), the
    Pokémon Mansion switch statues, the spinner tiles of SPINNER_MAPS and the warp pads of WARP_PAD_MAPS, with the
    per-map end condition (OBJ_END_*: a flag of flags_priv or a key item id)."""
    def end(kind_name):
        if kind_name is None:
            return OBJ_END_NONE, 0
        kind, name = kind_name
        if kind == "flag":
            return OBJ_END_FLAG, c.value(name)
        if kind == "trainer":
            return OBJ_END_FLAG, c.value("TRAINER_FLAGS_START") + c.value(name)
        return OBJ_END_KEY_ITEM, c.value(name)

    out = []
    for m in _map_jsons():
        mid = m["id"]
        if mid == TRASH_CAN_MAP:
            e = end(("flag", TRASH_CAN_END))
            for b in m.get("bg_events") or []:
                mt = re.search(r"_EventScript_TrashCan(\d+)$", str(b.get("script", "")))
                if mt:
                    out.append((keys[mid], b["x"], b["y"], OBJ_TRASH_CAN, int(mt.group(1))) + e)
        if mid in MANSION_MAPS:
            for b in m.get("bg_events") or []:
                if str(b.get("script", "")).endswith("_EventScript_Statue"):
                    out.append((keys[mid], b["x"], b["y"], OBJ_MANSION_SWITCH, 0) + end(None))
    atlas = _atlas()
    B = atlas.behaviors()
    spin = {B["MB_SPIN_RIGHT"], B["MB_SPIN_LEFT"], B["MB_SPIN_UP"], B["MB_SPIN_DOWN"]}
    maps = atlas.maps()
    for mid, cond in SPINNER_MAPS.items():
        m, e = maps[mid], end(cond)
        for y in range(m.h):
            for x in range(m.w):
                if m.behavior(x, y) in spin:
                    out.append((keys[mid], x, y, OBJ_SPINNER, 0) + e)
    for mid, cond in WARP_PAD_MAPS.items():
        m, e = maps[mid], end(cond)
        for w in m.warps:
            if m.behavior(w["x"], w["y"]) == B["MB_REGULAR_WARP"]:
                out.append((keys[mid], w["x"], w["y"], OBJ_WARP_PAD, 0) + e)
    out.sort()
    assert len(set((k, x, y) for k, x, y, *_ in out)) == len(out)  # one entry per tile
    return out


def boulders(keys):
    """Sorted [(map key, x, y)] of every Strength boulder object's template position."""
    out = []
    for m in _map_jsons():
        for o in m.get("object_events") or []:
            if o.get("graphics_id") == BOULDER_GFX:
                out.append((keys[m["id"]], o["x"], o["y"]))
    out.sort()
    return out


def interact_keys(keys):
    """(objects, bg): the sorted (map key, local id) of every object event that is not a clone (local id = its
    1-based position in the map's object_events, INTERACT_* comment above), and the sorted (map key, x, y) of every
    background event of INTERACT_BG_TYPES (one per position)."""
    out_o, out_b, clones, types = [], set(), 0, {}
    for m in _map_jsons():
        key = keys[m["id"]]
        for i, o in enumerate(m.get("object_events") or []):
            if o["type"] == "clone":
                clones += 1
            else:
                assert o["type"] == "object", o["type"]
                out_o.append((key, i + 1))
        for b in m.get("bg_events") or []:
            assert b["type"] in INTERACT_BG_TYPES, b["type"]  # (a new type: decide whether the game selects it)
            types[b["type"]] = types.get(b["type"], 0) + 1
            out_b.add((key, b["x"], b["y"]))
    assert len(out_b) == sum(types.values())  # (no two background events share a position)
    assert (len(out_o), clones) == INTERACT_OBJECTS_EXPECTED, (len(out_o), clones)
    assert (types["sign"], types["hidden_item"]) == INTERACT_BG_EXPECTED, types
    return sorted(out_o), sorted(out_b)


def _rows(values, per=32):
    return "\n".join("    " + ", ".join(str(v) for v in values[i:i + per]) + "," for i in range(0, len(values), per))


def render(t) -> str:
    lad = "\n".join(f"    ({n!r}, {a}, 0x{f:03X})," for n, a, f in t["ladder"])
    maps = "\n".join(f"    (0x{k:04X}, {n!r}, {w}, {h})," for k, n, w, h in t["maps"])
    objs = "\n".join("    " + ", ".join(f"(0x{k:04X}, {x}, {y}, {kd}, {i}, {ek}, 0x{ev:03X})" for k, x, y, kd, i, ek, ev in t["objs"][j:j + 3]) + ","
                     for j in range(0, len(t["objs"]), 3))
    boulders = "\n".join("    " + ", ".join(f"(0x{k:04X}, {x}, {y})" for k, x, y in t["boulders"][j:j + 6]) + ","
                         for j in range(0, len(t["boulders"]), 6))
    io, ib = t["interact_objects"], t["interact_bg"]
    i_objs = "\n".join("    " + ", ".join(f"(0x{k:04X}, {i})" for k, i in io[j:j + 8]) + ","
                       for j in range(0, len(io), 8))
    i_bg = "\n".join("    " + ", ".join(f"(0x{k:04X}, {x}, {y})" for k, x, y in ib[j:j + 6]) + ","
                     for j in range(0, len(ib), 6))
    return f'''"""GENERATED by rl/pokerl/gen_tables.py from the game source. Do not edit.

Regenerate: PYTHONPATH=rl python3 -m pokerl.gen_tables
"""

SPECIES_COUNT = {t["species_count"]}  # ids 0..{t["species_count"] - 1} (SPECIES_EGG {t["species_egg"]}, Unown forms above)
SPECIES_EGG = {t["species_egg"]}
MOVES_COUNT = {t["moves_count"]}
ITEMS_COUNT = {t["items_count"]}
FLAGS_COUNT = {t["flags_count"]}  # flags 0x000..0x{t["flags_count"] - 1:03X} = flags_priv (288 bytes)
BATTLE_TYPE_TRAINER = {t["battle_type_trainer"]}

# HM01..HM07 move ids (Cut, Fly, Surf, Strength, Flash, Rock Smash, Waterfall)
HM_MOVES = {t["hm_moves"]}

# per species id: bit i = HM0(i+1) learnable (tmhm_learnsets.h)
HM_LEARN = [
{_rows(t["hm_learn"])}
]

# item id -> pocket: 0 none, 1 ITEMS, 2 KEY_ITEMS, 3 POKE_BALLS, 4 TM_CASE, 5 BERRY_POUCH (src/data/items.json)
ITEM_POCKET = [
{_rows(t["item_pocket"], 64)}
]

# TM01..TM50, HM01..HM08 (item ids TM_FIRST_ITEM .. TM_FIRST_ITEM + 57) -> move id (party_menu.c sTMHMMoves)
TM_FIRST_ITEM = {t["tm_first"]}
TM_MOVES = [
{_rows(t["tm_moves"])}
]

# per species id the species id of its
# evolutionary family root (evolution.h: pre-evolutions followed to the root; Egg, SPECIES_NONE and the unused
# SPECIES_OLD_UNOWN_* 0 = no family; the Unown form ids above the Egg -> SPECIES_UNOWN)
FAMILY = [
{_rows(t["family"])}
]
FAMILY_ROOTS = [  # dense family index -> root species id (sorted)
{_rows(t["family_roots"])}
]
FAMILY_COUNT = {len(t["family_roots"])}
FAMILY_INDEX = [  # per species id: dense family index (position of FAMILY[s] in FAMILY_ROOTS), -1 = no family
{_rows([t["family_roots"].index(f) if f else -1 for f in t["family"]])}
]
# per move id: type, power, accuracy, PP, priority (battle_moves.h)
MOVE_STATS = [
{_rows(t["move_stats"], 8)}
]

# per species id: base HP, Attack, Defense, Speed, Sp. Attack, Sp. Defense (species_info.h; 0 for ids without a block)
SPECIES_BASE_STATS = [
{_rows(t["species_base"], 8)}
]

# per species id: 58-bit mask of the learnable TM/HM items, bit i = item TM_FIRST_ITEM + i, move TM_MOVES[i]
# (tmhm_learnsets.h; bits 50-56 = HM01-HM07 = HM_LEARN)
TM_LEARN = [
{_rows(["0x%015X" % v for v in t["tm_learn"]], 8)}
]
# names (the first #define of each id in the game's constant headers, without the prefix)
SPECIES_NAMES = [
{_rows([repr(n) for n in t["species_names"]], 10)}
]
MOVE_NAMES = [
{_rows([repr(n) for n in t["move_names"]], 10)}
]

# flag categories: 0 no reward, 1 story/system, 2 trainer, 3 badge, 4 Elite Four / Champion; weights (reward
# v0.10): story {STORY_WEIGHT}, item ball of a non-key item {ITEM_BALL_WEIGHT} (a story flag), trainer {TRAINER_WEIGHT}, badge {BADGE_WEIGHT},
# Elite Four / Champion {ELITE4_WEIGHT}; 0: temp flags, hidden items, most system flags, world-map / visited flags
FLAG_CAT_NONE, FLAG_CAT_STORY, FLAG_CAT_TRAINER, FLAG_CAT_BADGE, FLAG_CAT_ELITE4 = 0, 1, 2, 3, 4
FLAG_WEIGHTS = [
{_rows(t["weights"], 64)}
]
FLAG_CATS = [
{_rows(t["cats"], 64)}
]

# story variables (reward v0.7: the trigger variables of map scripts, gen_tables.trigger_vars, minus counters and
# link state; v0.6 and before: every VAR_MAP_SCENE_*) as indices into vars_priv (var - 0x4000), sorted
SCENE_VAR_IDX = [
{_rows(["0x%02X" % i for i in t["scene_idx"]], 16)}
]

BADGE_FLAGS = [{", ".join(f"0x{f:03X}" for f in t["badge_flags"])}]  # FLAG_BADGE01_GET..08
ELITE4_FLAGS = [{", ".join(f"0x{f:03X}" for f in t["e4_flags"])}]  # LORELEI, BRUNO, AGATHA, LANCE, CHAMP
FLAG_DEFEATED_CHAMP = 0x{t["e4_flags"][4]:03X}

# (opponent, ace level = highest level in the party, flag that marks it beaten): the 8 gym leaders in
# badge order, then the Champion (reward v0.7: the Elite Four are not on the ladder). The level cap is the
# lowest ace whose flag is unseen.
ACE_LADDER = [
{lad}
]
ACE_AFTER_CHAMPION = {t["ladder"][-1][1]}

# party size (number of mons in trainer_parties.h) of each ACE_LADDER opponent, same order (reward v0.5: the
# levels term sums the next opponent's party size of the party's highest levels); after the Champion: the Champion's
ACE_PARTY_SIZES = {t["sizes"]}
PARTY_SIZE_AFTER_CHAMPION = {t["sizes"][-1]}

# every MAP_* constant: (map_group * 256 + map_num, name, width, height); dense index = position + 1
MAPS = [
{maps}
]

# object tiles of the puzzle terms (reward.py term 8),
# (map key, x, y, kind, id, end kind, end value) sorted; kind 0 Vermilion Gym trash can (id = can number 1..15), 1
# Pokémon Mansion secret-switch statue, 2 spinner tile (Rocket Hideout B2F/B3F, Viridian Gym), 3 warp pad (Silph
# Co. 2F-11F, Saffron Gym); end kind 0 none, 1 a flag (nothing paid once it is set: the Vermilion switches found,
# badge 8 / badge 6, Silph Co.'s Giovanni beaten = his trainer flag), 2 a key item id (once held: the Silph Scope)
OBJ_TRASH_CAN, OBJ_MANSION_SWITCH, OBJ_SPINNER, OBJ_WARP_PAD = 0, 1, 2, 3
OBJ_END_NONE, OBJ_END_FLAG, OBJ_END_KEY_ITEM = 0, 1, 2
OBJECT_TILES = [
{objs}
]

# Strength boulders' template positions (map key, x, y), sorted: a boulder seen on any other tile has been pushed
BOULDERS = [
{boulders}
]
# the boulder puzzles' completion events (reward v0.13 boulder_event_reward): flags FLAG_STOPPED_SEAFOAM_B3F_CURRENT,
# FLAG_STOPPED_SEAFOAM_B4F_CURRENT and the vars_priv indices of VAR_MAP_SCENE_VICTORY_ROAD_1F, _2F_BOULDER1,
# _2F_BOULDER2, _3F (the floor switches set them to 100; all four are in SCENE_VAR_IDX)
BOULDER_FLAGS = [{", ".join(f"0x{f:03X}" for f in t["boulder_flags"])}]
BOULDER_VAR_IDX = [{", ".join(f"0x{v:02X}" for v in t["boulder_vars"])}]
FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES = 0x{t["trash_flag"]:03X}  # (reward v0.13 trash_unlock_reward)
VERMILION_GYM_MAP = 0x{t["trash_map"]:04X}  # map key (the trash can term's attempt key is read there)
ITEM_SILPH_SCOPE = 359

# the overworld interaction term's keys (reward.py term 9).
# (map key, local id) of every object event that is not a clone, sorted: local id = the event's 1-based position in
# its map's object_events, clones counted (the template's localId; a clone spawns with its original's map and local
# id, so clones have no key of their own) -- NPCs, trainers, item balls, Cut trees, rocks, boulders, ...
INTERACT_OBJECTS = [
{i_objs}
]
# (map key, x, y) of every sign / hidden item background event, sorted, one per position (the Vermilion Gym trash
# cans and Pokémon Mansion switch statues included: they may also pay through reward v0.13's terms)
INTERACT_BG = [
{i_bg}
]
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if tables.py differs from the generated text")
    args = ap.parse_args(argv)
    text = render(generate())
    if args.check:
        cur = open(OUT).read() if os.path.exists(OUT) else ""
        if cur != text:
            print("tables.py is out of date", file=sys.stderr)
            return 1
        print("tables.py up to date")
        return 0
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
