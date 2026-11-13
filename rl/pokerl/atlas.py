#!/usr/bin/env python3
"""Static world atlas of the decompilation's map data (gen_tables.py reads the layouts' metatile behaviours here).

Reads the decompilation's own data: game/data/maps/*/map.json (warps, connections, objects,
coord/bg events), game/data/layouts/layouts.json + map.bin (metatile id, collision, elevation),
the tilesets' metatile_attributes.bin (behaviour) and include/constants/metatile_behaviors.h.
Tile semantics mirror the game's movement code (GetCollisionAtCoords, GetLedgeJumpDirection,
gDirectionBlockedMetatileFuncs, IsElevationMismatchAt).

    PYTHONPATH=rl:python python3 -m pokerl.atlas MAP_PALLET_TOWN_PLAYERS_HOUSE_1F    # print a map (behaviours)
"""
import functools
import glob
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GAME = os.path.join(ROOT, "game")

DIRS = {"S": (0, 1), "N": (0, -1), "W": (-1, 0), "E": (1, 0)}
OPP = {"S": "N", "N": "S", "W": "E", "E": "W"}


@functools.lru_cache(None)
def behaviors():
    out = {}
    with open(os.path.join(GAME, "include", "constants", "metatile_behaviors.h")) as f:
        for line in f:
            m = re.match(r"#define\s+(MB_\w+)\s+(0x[0-9A-Fa-f]+|\d+)", line)
            if m:
                out[m.group(1)] = int(m.group(2), 0)
    return out


def mb(name):
    return behaviors()[name]


class Behavior:
    """Predicates of game/src/metatile_behavior.c used by navigation."""

    def __init__(self):
        B = behaviors()
        self.door = {B["MB_WARP_DOOR"]}
        self.nonanim_door = {B["MB_CAVE_DOOR"]}
        self.ladder = {B["MB_LADDER"]}
        self.escalator = set(range(B["MB_UP_ESCALATOR"], B["MB_DOWN_ESCALATOR"] + 1))
        self.arrow = {B["MB_NORTH_ARROW_WARP"]: "N", B["MB_SOUTH_ARROW_WARP"]: "S",
                      B["MB_WEST_ARROW_WARP"]: "W", B["MB_EAST_ARROW_WARP"]: "E"}
        self.stair = {B["MB_UP_LEFT_STAIR_WARP"]: "W", B["MB_DOWN_LEFT_STAIR_WARP"]: "W",
                      B["MB_UP_RIGHT_STAIR_WARP"]: "E", B["MB_DOWN_RIGHT_STAIR_WARP"]: "E"}
        self.surfable = {B[n] for n in ("MB_POND_WATER", "MB_FAST_WATER", "MB_DEEP_WATER", "MB_WATERFALL",
                                        "MB_OCEAN_WATER", "MB_UNUSED_WATER", "MB_CYCLING_ROAD_WATER",
                                        "MB_EASTWARD_CURRENT", "MB_WESTWARD_CURRENT", "MB_NORTHWARD_CURRENT",
                                        "MB_SOUTHWARD_CURRENT")}
        self.waterfall = {B["MB_WATERFALL"]}
        self.ice = {B["MB_ICE"]}  # MetatileBehavior_IsIce_2: ForcedMovement_Slip
        self.warp_pad = {B[n] for n in ("MB_REGULAR_WARP", "MB_UNION_ROOM_WARP") if n in B}
        self.fall_warp = {B["MB_FALL_WARP"]} if "MB_FALL_WARP" in B else set()
        self.jump = {B["MB_JUMP_SOUTH"]: "S", B["MB_JUMP_NORTH"]: "N", B["MB_JUMP_WEST"]: "W",
                     B["MB_JUMP_EAST"]: "E"}
        self.block = {"N": set(), "S": set(), "W": set(), "E": set()}
        for d, names in (("E", ("EAST", "NORTHEAST", "SOUTHEAST")), ("W", ("WEST", "NORTHWEST", "SOUTHWEST")),
                         ("N", ("NORTH", "NORTHEAST", "NORTHWEST")), ("S", ("SOUTH", "SOUTHEAST", "SOUTHWEST"))):
            for n in names:
                if "MB_IMPASSABLE_" + n in B:
                    self.block[d].add(B["MB_IMPASSABLE_" + n])
        self.spin = {B["MB_SPIN_RIGHT"]: "E", B["MB_SPIN_LEFT"]: "W", B["MB_SPIN_UP"]: "N", B["MB_SPIN_DOWN"]: "S"}
        self.stop_spin = B["MB_STOP_SPINNING"]
        self.tall_grass = {B["MB_TALL_GRASS"]}
        self.counter = {B["MB_COUNTER"]} if "MB_COUNTER" in B else set()
        forced = set(range(B["MB_WALK_EAST"], B["MB_TRICK_HOUSE_PUZZLE_8_FLOOR"] + 1)) if \
            "MB_TRICK_HOUSE_PUZZLE_8_FLOOR" in B else set()
        forced |= set(range(B["MB_EASTWARD_CURRENT"], B["MB_SOUTHWARD_CURRENT"] + 1))
        forced |= {B["MB_WATERFALL"], B["MB_ICE"]}
        if "MB_SPIN_RIGHT" in B:
            forced |= set(range(B["MB_SPIN_RIGHT"], B["MB_SPIN_DOWN"] + 1))
        self.forced = forced

    def step_warp(self, b):
        """IsWarpMetatileBehavior-like: stepping onto the tile takes its warp event."""
        return b in self.door or b in self.nonanim_door or b in self.ladder or b in self.escalator or \
            b in self.warp_pad or b in self.fall_warp


BEH = None


def beh():
    global BEH
    if BEH is None:
        BEH = Behavior()
    return BEH


@functools.lru_cache(None)
def tileset_attrs():
    """{tileset symbol: [u32 attributes per metatile]}"""
    hdr = open(os.path.join(GAME, "src", "data", "tilesets", "headers.h")).read()
    mt = open(os.path.join(GAME, "src", "data", "tilesets", "metatiles.h")).read()
    attr_file = {}
    for m in re.finditer(r"(gMetatileAttributes_\w+)\[\]\s*=\s*INCBIN_U32\(\"([^\"]+)\"\)", mt):
        attr_file[m.group(1)] = m.group(2)
    out = {}
    for m in re.finditer(r"const struct Tileset (gTileset_\w+)\s*=\s*\{(.*?)\};", hdr, re.S):
        a = re.search(r"\.metatileAttributes\s*=\s*(\w+)", m.group(2))
        if not a or a.group(1) not in attr_file:
            continue
        p = os.path.join(GAME, attr_file[a.group(1)])
        data = open(p, "rb").read()
        out[m.group(1)] = list(struct.unpack("<%dI" % (len(data) // 4), data))
    return out


@functools.lru_cache(None)
def layouts():
    j = json.load(open(os.path.join(GAME, "data", "layouts", "layouts.json")))
    return {l["id"]: l for l in j["layouts"] if "id" in l}


NUM_METATILES_IN_PRIMARY = 640


class Map:
    def __init__(self, j, dirname):
        self.j = j
        self.id = j["id"]
        self.name = j["name"]
        self.dir = dirname
        lay = layouts()[j["layout"]]
        self.w, self.h = lay["width"], lay["height"]
        raw = open(os.path.join(GAME, lay["blockdata_filepath"]), "rb").read()
        self.block = struct.unpack("<%dH" % (self.w * self.h), raw[:2 * self.w * self.h])
        ta = tileset_attrs()
        self.prim = ta.get(lay["primary_tileset"], [])
        self.sec = ta.get(lay["secondary_tileset"], [])
        self.warps = j.get("warp_events") or []
        self.connections = j.get("connections") or []
        self.objects = j.get("object_events") or []
        self.coords = j.get("coord_events") or []
        self.bgs = j.get("bg_events") or []

    def inside(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def metatile(self, x, y):
        return self.block[y * self.w + x] & 0x3FF

    def collision(self, x, y):
        return (self.block[y * self.w + x] >> 10) & 3

    def elevation(self, x, y):
        return self.block[y * self.w + x] >> 12

    def behavior(self, x, y):
        mt = self.metatile(x, y)
        if mt < NUM_METATILES_IN_PRIMARY:
            a = self.prim[mt] if mt < len(self.prim) else 0
        else:
            i = mt - NUM_METATILES_IN_PRIMARY
            a = self.sec[i] if i < len(self.sec) else 0
        return a & 0x1FF

    def encounter_type(self, x, y):
        mt = self.metatile(x, y)
        if mt < NUM_METATILES_IN_PRIMARY:
            a = self.prim[mt] if mt < len(self.prim) else 0
        else:
            i = mt - NUM_METATILES_IN_PRIMARY
            a = self.sec[i] if i < len(self.sec) else 0
        return (a >> 24) & 7

    def local_id(self, name_or_num):
        """Object local id (1-based index in object_events) from a LOCALID_ name or a number."""
        if isinstance(name_or_num, int):
            return name_or_num
        if re.match(r"^\d+$", str(name_or_num)):
            return int(name_or_num)
        for i, o in enumerate(self.objects):
            if o.get("local_id") == name_or_num:
                return i + 1
        raise KeyError("%s: no object %s" % (self.id, name_or_num))

    def object(self, lid):
        return self.objects[self.local_id(lid) - 1]

    def warp_kind(self, i):
        w = self.warps[i]
        b = self.behavior(w["x"], w["y"]) if self.inside(w["x"], w["y"]) else 0
        B = beh()
        if b in B.door:
            return "door"
        if b in B.arrow:
            return "arrow_" + B.arrow[b]
        if b in B.stair:
            return "stair_" + B.stair[b]
        if B.step_warp(b):
            return "step"
        return "none"

    def text(self):
        B = beh()
        rows = []
        for y in range(self.h):
            r = []
            for x in range(self.w):
                b = self.behavior(x, y)
                c = "#" if self.collision(x, y) else "."
                if b in B.surfable:
                    c = "~" if not self.collision(x, y) else "w"
                elif b in B.tall_grass:
                    c = '"'
                elif b in B.jump:
                    c = {"S": "v", "N": "^", "W": "<", "E": ">"}[B.jump[b]]
                for i, w in enumerate(self.warps):
                    if (w["x"], w["y"]) == (x, y):
                        c = str(i % 10)
                r.append(c + "%x" % self.elevation(x, y))
            rows.append("%3d " % y + "".join(r))
        return "\n".join(rows)


@functools.lru_cache(None)
def maps():
    out = {}
    for p in glob.glob(os.path.join(GAME, "data", "maps", "*", "map.json")):
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        if "layout" not in j or j["layout"] not in layouts():
            continue
        out[j["id"]] = Map(j, os.path.dirname(p))
    return out


def main():
    for mid in sys.argv[1:]:
        m = maps()[mid]
        print("== %s %dx%d" % (mid, m.w, m.h))
        print(m.text())
        for i, w in enumerate(m.warps):
            print("  warp %d (%d,%d) %s -> %s #%s" % (i, w["x"], w["y"], m.warp_kind(i), w["dest_map"],
                                                     w["dest_warp_id"]))
        for i, o in enumerate(m.objects):
            print("  obj %d %s (%d,%d) %s %s" % (i + 1, o.get("local_id"), o["x"], o["y"], o.get("graphics_id"),
                                               o.get("script")))


if __name__ == "__main__":
    main()
