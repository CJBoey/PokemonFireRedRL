"""A scope's area exits as D-pad presses: the table of the env's exit mask (env.py "Relay scopes").

The table of an area holds every (map, x, y, direction) where pressing that direction walks out of the area, read
from the data the field code acts on (game/src/field_control_avatar.c):

* arrow warp (``MB_{NORTH,SOUTH,EAST,WEST}_ARROW_WARP``, the mats at cave and building exits; ``TryArrowWarp``):
  standing on the warp tile, pressing the arrow's direction. Diglett's Cave north entrance (4, 6) DOWN -> Route 2;
* stair warp (``MB_{UP,DOWN}_{LEFT,RIGHT}_STAIR_WARP``; ``TryArrowWarp``): standing on the warp tile, pressing the
  stairs' direction;
* step warp (doors, cave doors, ladders, escalators, warp pads, holes; ``IsWarpMetatileBehavior``, ``TryDoorWarp``):
  stepping onto the warp tile, from each neighbouring tile of its map;
* connection: walking off the map's edge along the part that joins a map outside the area.

Warps to ``MAP_DYNAMIC`` (elevators, the Union Room) are not listed: as in ``scopes.closure``, what they serve is
inside the area. A warp event on a tile with no warp behaviour (the ``MB_NORMAL`` tiles beside a mat) never fires and
is not listed. Warp elevations are ignored (a press masked on a bridge over a warp would only cost that press; no
exit warp of the six scopes has one). Moves the player does not press (a ledge's landing, currents, spinners, ice) are
not listed either: :func:`uncovered` reports any that would carry a player out (none in the six scopes; a test holds
it). Everything else that leaves an area does not walk: Fly, Dig, Escape Rope, Teleport, a whiteout's respawn, Bill's
boat from Cinnabar Island.

    PYTHONPATH=rl:python python3 -m pokerl.relay.exits      # prints every scope's presses
"""
from __future__ import annotations

import functools
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np

from ..mask import ACTIONS, NUM_ACTIONS
from .scopes import MAP_DENSE, ScopeSet, scope_set

NORTH, SOUTH, WEST, EAST = 1, 2, 4, 8  # the direction bits of a table entry (in the order of ALLOWED's actions)
DIRECTIONS: Tuple[Tuple[int, int, int], ...] = ((NORTH, 0, -1), (SOUTH, 0, 1), (WEST, -1, 0), (EAST, 1, 0))
DIR_NAMES = {NORTH: "UP", SOUTH: "DOWN", WEST: "LEFT", EAST: "RIGHT"}
# bool [16, 8]: the actions a table entry's bits leave allowed
ALLOWED = np.ones((16, NUM_ACTIONS), dtype=bool)
for _b, _name in DIR_NAMES.items():
    ALLOWED[[i for i in range(16) if i & _b], ACTIONS.index(_name)] = False
ALLOWED.setflags(write=False)

ARROW_WARPS = {"MB_NORTH_ARROW_WARP": NORTH, "MB_SOUTH_ARROW_WARP": SOUTH, "MB_WEST_ARROW_WARP": WEST,
               "MB_EAST_ARROW_WARP": EAST, "MB_UP_LEFT_STAIR_WARP": WEST, "MB_DOWN_LEFT_STAIR_WARP": WEST,
               "MB_UP_RIGHT_STAIR_WARP": EAST, "MB_DOWN_RIGHT_STAIR_WARP": EAST}
STEP_WARPS = ("MB_WARP_DOOR", "MB_LADDER", "MB_UP_ESCALATOR", "MB_DOWN_ESCALATOR", "MB_CAVE_DOOR",
              "MB_LAVARIDGE_1F_WARP", "MB_REGULAR_WARP", "MB_FALL_WARP", "MB_UNION_ROOM_WARP")
# tiles that move a player on by themselves, by the direction they carry (uncovered)
CARRIERS = {"MB_JUMP_NORTH": NORTH, "MB_JUMP_SOUTH": SOUTH, "MB_JUMP_WEST": WEST, "MB_JUMP_EAST": EAST,
            "MB_NORTHWARD_CURRENT": NORTH, "MB_SOUTHWARD_CURRENT": SOUTH, "MB_WESTWARD_CURRENT": WEST,
            "MB_EASTWARD_CURRENT": EAST, "MB_SPIN_UP": NORTH, "MB_SPIN_DOWN": SOUTH, "MB_SPIN_LEFT": WEST,
            "MB_SPIN_RIGHT": EAST, "MB_SLIDE_NORTH": NORTH, "MB_SLIDE_SOUTH": SOUTH, "MB_SLIDE_WEST": WEST,
            "MB_SLIDE_EAST": EAST, "MB_ICE": NORTH | SOUTH | WEST | EAST, "MB_THIN_ICE": NORTH | SOUTH | WEST | EAST,
            "MB_CRACKED_ICE": NORTH | SOUTH | WEST | EAST}
CONNECTIONS = {"up": NORTH, "down": SOUTH, "left": WEST, "right": EAST}
NUM_PRIMARY_METATILES = 640  # NUM_METATILES_IN_PRIMARY (include/fieldmap.h): later ids index the secondary tileset


class MapTiles(NamedTuple):
    """What a map's exits are read from."""
    name: str
    width: int
    height: int
    behaviour: np.ndarray                          # int [height, width], MB_* values
    warps: Tuple[Tuple[int, int, str], ...]        # (x, y, dest_map)
    connections: Tuple[Tuple[int, int, str], ...]  # (direction bit, offset, map); dive / emerge left out


class GameData(NamedTuple):
    """The game directory's map files, read once."""
    root: str
    behaviours: Dict[str, int]    # MB_* name -> value
    attributes: Dict[str, str]    # tileset name -> metatile_attributes.bin path
    layouts: Dict[str, dict]      # LAYOUT_* -> layouts.json entry
    maps: Dict[str, dict]         # MAP_* -> map.json


@functools.lru_cache(maxsize=None)
def game_data(root: str) -> GameData:
    with open(os.path.join(root, "include", "constants", "metatile_behaviors.h"), encoding="utf-8") as f:
        behaviours = {m.group(1): int(m.group(2), 16)
                      for m in re.finditer(r"^#define (MB_\w+) (0x[0-9A-Fa-f]+)", f.read(), re.M)}
    with open(os.path.join(root, "src", "data", "tilesets", "metatiles.h"), encoding="utf-8") as f:
        attributes = {m.group(1): m.group(2)
                      for m in re.finditer(r'gMetatileAttributes_(\w+)\[\] = INCBIN_U32\("([^"]+)"', f.read())}
    with open(os.path.join(root, "data", "layouts", "layouts.json"), encoding="utf-8") as f:
        layouts = {la["id"]: la for la in json.load(f)["layouts"] if la.get("id")}
    maps = {}
    for p in sorted(glob.glob(os.path.join(root, "data", "maps", "*", "map.json"))):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        maps[d["id"]] = d
    missing = [n for n in (*ARROW_WARPS, *STEP_WARPS) if n not in behaviours]
    if missing:
        raise ValueError(f"relay exits: {', '.join(missing)} not in metatile_behaviors.h")
    return GameData(root, behaviours, attributes, layouts, maps)


def _attributes(data: GameData, tileset: str) -> np.ndarray:
    path = data.attributes[tileset.replace("gTileset_", "")]
    return np.fromfile(os.path.join(data.root, path), dtype="<u4")


@functools.lru_cache(maxsize=None)
def map_tiles(name: str, root: str) -> MapTiles:
    """Map ``name``'s tile behaviours (metatile id in the blockdata's low 10 bits, behaviour in the attributes' low 9:
    ``sMetatileAttrMasks``), warps and connections."""
    data = game_data(root)
    d = data.maps[name]
    la = data.layouts[d["layout"]]
    w, h = int(la["width"]), int(la["height"])
    grid = np.fromfile(os.path.join(root, la["blockdata_filepath"]), dtype="<u2")
    if grid.size != w * h:
        raise ValueError(f"relay exits: {name}: blockdata of {grid.size} blocks for a {w}x{h} layout")
    table = np.zeros(1024, dtype=np.int64)
    prim = _attributes(data, la["primary_tileset"])[:NUM_PRIMARY_METATILES]
    sec = _attributes(data, la["secondary_tileset"])[:1024 - NUM_PRIMARY_METATILES]
    table[:prim.size] = prim & 0x1FF
    table[NUM_PRIMARY_METATILES:NUM_PRIMARY_METATILES + sec.size] = sec & 0x1FF
    warps = tuple((int(e["x"]), int(e["y"]), e["dest_map"]) for e in d.get("warp_events") or [])
    conn = tuple((CONNECTIONS[c["direction"]], int(c["offset"]), c["map"]) for c in d.get("connections") or []
                 if c["direction"] in CONNECTIONS)
    return MapTiles(name, w, h, table[grid & 0x3FF].reshape(h, w), warps, conn)


def exit_presses(area: Iterable[str], root: str) -> Dict[Tuple[str, int, int], int]:
    """``{(map, x, y): direction bits}``: the presses that walk out of ``area`` (module docstring)."""
    area = set(area)
    data = game_data(root)
    arrows = {data.behaviours[n]: b for n, b in ARROW_WARPS.items()}
    steps = {data.behaviours[n] for n in STEP_WARPS}
    out: Dict[Tuple[str, int, int], int] = {}

    def add(m: str, x: int, y: int, bit: int) -> None:
        out[(m, x, y)] = out.get((m, x, y), 0) | bit

    for m in sorted(area):
        t = map_tiles(m, root)
        for wx, wy, dest in t.warps:
            if dest == "MAP_DYNAMIC" or dest in area or not (0 <= wx < t.width and 0 <= wy < t.height):
                continue
            b = int(t.behaviour[wy, wx])
            if b in arrows:
                add(m, wx, wy, arrows[b])
            elif b in steps:
                for bit, dx, dy in DIRECTIONS:
                    if 0 <= wx - dx < t.width and 0 <= wy - dy < t.height:
                        add(m, wx - dx, wy - dy, bit)
        for bit, off, other in t.connections:
            if other in area:
                continue
            o = map_tiles(other, root)
            if bit in (NORTH, SOUTH):
                y = 0 if bit == NORTH else t.height - 1
                for x in range(max(0, off), min(t.width, off + o.width)):
                    add(m, x, y, bit)
            else:
                x = 0 if bit == WEST else t.width - 1
                for y in range(max(0, off), min(t.height, off + o.height)):
                    add(m, x, y, bit)
    return out


def uncovered(area: Iterable[str], root: str) -> List[str]:
    """The walks out of ``area`` the table cannot stop: a tile that moves the player on by itself (ledge, current,
    spinner, ice) in a direction its entry lists, and a step warp out of the area at a map edge joined to a map of
    the area (stepped on from the other map)."""
    area = set(area)
    data = game_data(root)
    carriers = {data.behaviours[n]: b for n, b in CARRIERS.items() if n in data.behaviours}
    steps = {data.behaviours[n] for n in STEP_WARPS}
    presses = exit_presses(area, root)
    bad = []
    for (m, x, y), bits in sorted(presses.items()):
        t = map_tiles(m, root)
        if 0 <= x < t.width and 0 <= y < t.height and carriers.get(int(t.behaviour[y, x]), 0) & bits:
            bad.append(f"{m} ({x}, {y}): a carrying tile leaves the area")
    for m in sorted(area):
        t = map_tiles(m, root)
        for wx, wy, dest in t.warps:
            if dest == "MAP_DYNAMIC" or dest in area or not (0 <= wx < t.width and 0 <= wy < t.height) \
                    or int(t.behaviour[wy, wx]) not in steps:
                continue
            edges = {NORTH: wy == 0, SOUTH: wy == t.height - 1, WEST: wx == 0, EAST: wx == t.width - 1}
            if any(edges[bit] and other in area for bit, _, other in t.connections):
                bad.append(f"{m} ({wx}, {wy}): a step warp out at an edge joined to the area")
    return bad


class ExitTable:
    """Every scope's presses as one sorted lookup; ``presses`` per scope, ``bits`` for a batch of rows."""

    def __init__(self, presses: Sequence[Mapping[Tuple[str, int, int], int]]):
        self.presses = tuple(presses)
        keys, vals = [], []
        for s, t in enumerate(self.presses):
            for (m, x, y), b in t.items():
                keys.append(self._key(s, MAP_DENSE[m], x, y))
                vals.append(b)
        order = np.argsort(np.array(keys, dtype=np.int64))
        self.keys = np.array(keys, dtype=np.int64)[order]
        self.vals = np.array(vals, dtype=np.uint8)[order]

    @staticmethod
    def _key(scope, map_idx, x, y):  # (a negative or huge coordinate wraps to a tile no map has)
        return (scope << 32) | (map_idx << 20) | ((x & 1023) << 10) | (y & 1023)

    def bits(self, scope: np.ndarray, map_idx: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """uint8 [k]: the direction bits of row j's tile (x[j], y[j]) on dense map ``map_idx[j]`` in scope
        ``scope[j]`` (0: nothing leaves from there)."""
        k = self._key(np.asarray(scope, dtype=np.int64), np.asarray(map_idx, dtype=np.int64),
                      np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64))
        if self.keys.size == 0:
            return np.zeros(k.shape, dtype=np.uint8)
        i = np.searchsorted(self.keys, k).clip(max=self.keys.size - 1)
        return np.where(self.keys[i] == k, self.vals[i], 0).astype(np.uint8)


@functools.lru_cache(maxsize=4)
def exit_table(scopes: ScopeSet) -> ExitTable:
    """The table of ``scopes`` (built once per scope set: the env groups share it)."""
    return ExitTable([exit_presses(area, scopes.data.root) for area in scopes.areas])


def describe(scopes: ScopeSet = None) -> str:
    scopes = scopes or scope_set()
    table = exit_table(scopes)
    lines = []
    for name, t in zip(scopes.names, table.presses):
        lines.append(f"{name}: {sum(bin(b).count('1') for b in t.values())} presses on {len(t)} tiles")
        for (m, x, y), b in sorted(t.items()):
            lines.append(f"    {m[4:]} ({x}, {y}) " + " ".join(n for bit, n in DIR_NAMES.items() if b & bit))
    return "\n".join(lines)


__all__ = ["NORTH", "SOUTH", "WEST", "EAST", "DIRECTIONS", "DIR_NAMES", "ALLOWED", "MapTiles", "GameData", "game_data",
           "map_tiles", "exit_presses", "uncovered", "ExitTable", "exit_table", "describe"]


if __name__ == "__main__":
    print(describe())
