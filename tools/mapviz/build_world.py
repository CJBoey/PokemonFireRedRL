#!/usr/bin/env python3
"""Builds the FireRed world maps for the map visualizer from the decomp's data (game/data).

Two variants: "story" (Kanto and Sevii Islands 1-3, everything reachable before the Hall of Fame) and
"full" (also the post-game: Sevii 4-7 with Trainer Tower and Tanoby, Cerulean Cave,
the Ruby Path, and the event islands Navel Rock and Birth Island). Never drawn: the link rooms, the unused
prototype maps and the unused houses.

Every map is drawn at 16 px per metatile from its layout's blocks and tilesets. The outdoor maps are placed
by their map connections: Kanto from Pallet Town, the Sevii Islands in a band east of it. Indoor and cave maps (no connections) are grouped by
region map section (a city's buildings, a dungeon's floors), each group laid out in rows and placed in the
free space nearest to the doors that lead into it (warp events), with the map name above each map.

Writes OUT/player.png (the FireRed player's overworld frames down, up, left; RGBA), OUT/icons.png (every
species' party menu icon, first frame, 32 x 32 px each, ICON_COLS per row in species id order, SPECIES_NONE to
SPECIES_EGG, in its icon palette; RGBA, read by party.js) and, per variant,
into OUT/<variant>/ (OUT defaults to tools/mapviz/www/world):
  world.json    tile size, image size, chunk list and the regions: one per map, id = map_group * 256 + map_num
                (what serve.py sends), the map's top-left metatile on the world image, its size
  chunk_R_C.png the world image cut into CHUNK x CHUNK px pieces (empty pieces are left out)
  overview.png  the whole world at 1/4 scale or less (at most OVERVIEW_MAX px), shown when zoomed out
  walk.bin      uint8 per world metatile, row-major (width / 16 per row): WALK_NONE outside every map, WALK_LAND
                passable, WALK_WATER surfable (behavior), WALK_BLOCKED (collision bits); the novelty view's frontier

    python3 tools/mapviz/build_world.py [--game game] [--out tools/mapviz/www/world] [--variants story,full]
    python3 tools/mapviz/build_world.py --variants ""     # only player.png and icons.png
"""
import argparse
import collections
import json
import os
import re
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = 16                 # px per metatile
NUM_TILES_IN_PRIMARY = 640
NUM_METATILES_IN_PRIMARY = 640
NUM_PALS_IN_PRIMARY = 7
WALK_NONE, WALK_LAND, WALK_WATER, WALK_BLOCKED = 0, 1, 2, 3  # walk.bin classes (viewer.js WALK_*)
GAP = 2                   # free metatiles kept around every placed map or group
SEVII_GAP = 12
CHUNK = 2048              # px per side of the full-resolution pieces the page loads when zoomed in
OVERVIEW_MAX = 4096       # px, longest side of the overview image
SKIP_GROUPS = {"gMapGroup_Link"}
ICON_COLS = 32            # icons.png: species per row (party.js ICON_COLS)
POSTGAME = ("FourIsland", "FiveIsland", "SixIsland", "SevenIsland", "TrainerTower", "NavelRock", "BirthIsland",
            "CeruleanCave", "MtEmber_RubyPath")  # map folder prefixes left out of the story variant


def pretty(folder: str) -> str:
    """CeladonCity_DepartmentStore_1F -> Celadon City Department Store 1F"""
    s = folder.replace("_", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z0-9])|(?<=[0-9])(?=[A-Z][a-z])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Tilesets:
    """Tiles (index arrays), palettes (RGB) and metatiles of the decomp's tilesets, by their C symbol."""

    def __init__(self, game: str):
        self.game = game
        src = ""
        for f in ("src/graphics.c", "src/data/tilesets/graphics.h"):
            p = os.path.join(game, f)
            if os.path.exists(p):
                src += open(p).read()
        self.tiles_path = dict(re.findall(r"gTilesetTiles_(\w+)\[\] = INCBIN_U32\(\"([^\"]+?)\.4bpp(?:\.lz)?\"\)", src))
        self.pal_dir = {m: os.path.dirname(f) for m, f in
                        re.findall(r"gTilesetPalettes_(\w+)\[\]\[16\] =\s*\{\s*INCBIN_U16\(\"([^\"]+?)\"\)", src)}
        mth = open(os.path.join(game, "src/data/tilesets/metatiles.h")).read()
        self.metatiles_path = dict(re.findall(r"gMetatiles_(\w+)\[\] = INCBIN_U16\(\"([^\"]+?)\"\)", mth))
        self.attrs_path = dict(re.findall(r"gMetatileAttributes_(\w+)\[\] = INCBIN_U32\(\"([^\"]+?)\"\)", mth))
        # a tileset's tiles, palettes and metatiles are separate symbols; SilphCo shares the Condominiums tiles
        # and palettes but has its own metatiles
        hdr = open(os.path.join(game, "src/data/tilesets/headers.h")).read()
        self.ts = {}
        for m in re.finditer(r"gTileset_(\w+) =\s*\{(.*?)\};", hdr, re.S):
            body = m.group(2)
            self.ts[m.group(1)] = (re.search(r"\.tiles = gTilesetTiles_(\w+)", body).group(1),
                                   re.search(r"\.palettes = gTilesetPalettes_(\w+)", body).group(1),
                                   re.search(r"\.metatiles = gMetatiles_(\w+)", body).group(1),
                                   re.search(r"\.metatileAttributes = gMetatileAttributes_(\w+)", body).group(1))
        self.cache = {}
        self.attr_cache = {}
        mbh = open(os.path.join(game, "include/constants/metatile_behaviors.h")).read()
        mb = {k: int(v, 0) for k, v in re.findall(r"#define (MB_\w+) (0x[0-9A-Fa-f]+|\d+)", mbh)}
        src = open(os.path.join(game, "src/metatile_behavior.c")).read()
        body = re.search(r"sBehaviorSurfable\[\w+\] = \{(.*?)\};", src, re.S).group(1)
        self.surfable = np.zeros(512, bool)  # metatile behavior (bits 0-8) -> MetatileBehavior_IsSurfable
        self.surfable[[mb[k] for k in re.findall(r"\[(MB_\w+)\]\s*=\s*TRUE", body)]] = True

    def get(self, sym: str):
        name = sym.replace("gTileset_", "")
        if name not in self.cache:
            tsym, psym, msym, _ = self.ts[name]
            d = os.path.join(self.game, os.path.dirname(self.tiles_path[tsym]))
            pd = os.path.join(self.game, self.pal_dir[psym])
            if os.path.exists(os.path.join(d, "tiles.4bpp")):  # built tree: the exact tile data
                raw = np.fromfile(os.path.join(d, "tiles.4bpp"), np.uint8)
                tiles = np.stack([raw & 15, raw >> 4], -1).reshape(-1, 8, 8)  # low nibble = left pixel
            else:  # fresh checkout: the indexed PNG the build converts
                im = np.array(Image.open(os.path.join(d, "tiles.png")))
                h, w = im.shape
                tiles = im.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8) & 15
            pals = [self._palette(os.path.join(pd, f"{i:02d}")) for i in range(16)]
            mt = np.fromfile(os.path.join(self.game, self.metatiles_path[msym]), "<u2").reshape(-1, 8)
            self.cache[name] = (tiles, np.stack(pals), mt)
        return self.cache[name]

    def behaviors(self, sym: str) -> np.ndarray:
        """uint16 metatile behaviors (attribute bits 0-8) of a tileset."""
        name = sym.replace("gTileset_", "")
        if name not in self.attr_cache:
            a = np.fromfile(os.path.join(self.game, self.attrs_path[self.ts[name][3]]), "<u4")
            self.attr_cache[name] = (a & 0x1FF).astype(np.uint16)
        return self.attr_cache[name]

    @staticmethod
    def _palette(base: str) -> np.ndarray:
        if os.path.exists(base + ".gbapal"):  # BGR555
            c = np.fromfile(base + ".gbapal", "<u2")[:16]
            return np.stack([(c & 31) << 3, ((c >> 5) & 31) << 3, ((c >> 10) & 31) << 3], -1).astype(np.uint8)
        lines = open(base + ".pal").read().split("\n")[3:19]  # JASC-PAL
        return np.array([[int(v) for v in ln.split()] for ln in lines], np.uint8)


def render_layout(ts: Tilesets, game: str, lay: dict) -> np.ndarray:
    """RGB image of a layout's blocks: bottom layer, then the top layer over it (colour 0 transparent)."""
    pt, pp, pm = ts.get(lay["primary_tileset"])
    st, sp, sm = ts.get(lay["secondary_tileset"])
    pal = np.concatenate([pp[:NUM_PALS_IN_PRIMARY], sp[NUM_PALS_IN_PRIMARY:]])
    w, h = lay["width"], lay["height"]
    blocks = np.fromfile(os.path.join(game, lay["blockdata_filepath"]), "<u2")[:w * h].reshape(h, w) & 0x3FF
    img = np.empty((h * TILE, w * TILE, 3), np.uint8)
    blank = np.zeros((8, 8), np.uint8)
    memo = {}
    for (by, bx), b in np.ndenumerate(blocks):
        m = memo.get(b)
        if m is None:
            if b < NUM_METATILES_IN_PRIMARY:
                ent = pm[b] if b < len(pm) else np.zeros(8, np.uint16)
            else:
                ent = sm[b - NUM_METATILES_IN_PRIMARY] if b - NUM_METATILES_IN_PRIMARY < len(sm) else np.zeros(8, np.uint16)
            m = np.empty((TILE, TILE, 3), np.uint8)
            m[:] = pal[0][0]
            for k, e in enumerate(ent):  # 0-3 bottom layer, 4-7 top layer; each TL, TR, BL, BR
                ti, hf, vf, p = e & 0x3FF, (e >> 10) & 1, (e >> 11) & 1, e >> 12
                if ti < NUM_TILES_IN_PRIMARY:
                    t = pt[ti] if ti < len(pt) else blank
                else:
                    t = st[ti - NUM_TILES_IN_PRIMARY] if ti - NUM_TILES_IN_PRIMARY < len(st) else blank
                if hf:
                    t = t[:, ::-1]
                if vf:
                    t = t[::-1]
                q = k & 3
                sub = m[(q >> 1) * 8:(q >> 1) * 8 + 8, (q & 1) * 8:(q & 1) * 8 + 8]
                on = t != 0
                sub[on] = pal[p][t[on]]
            memo[b] = m
        img[by * TILE:(by + 1) * TILE, bx * TILE:(bx + 1) * TILE] = m
    return img


def walk_layout(ts: Tilesets, game: str, lay: dict) -> np.ndarray:
    """uint8 [h, w] walk classes of a layout's blocks: collision bits (10-11) set -> WALK_BLOCKED, else a surfable
    behavior -> WALK_WATER, else WALK_LAND."""
    w, h = lay["width"], lay["height"]
    raw = np.fromfile(os.path.join(game, lay["blockdata_filepath"]), "<u2")[:w * h].reshape(h, w)
    b = (raw & 0x3FF).astype(np.int64)
    pb, sb = ts.behaviors(lay["primary_tileset"]), ts.behaviors(lay["secondary_tileset"])
    beh = np.zeros(b.shape, np.int64)
    prim = b < NUM_METATILES_IN_PRIMARY
    i = np.clip(b, 0, len(pb) - 1)
    beh[prim] = np.where(b[prim] < len(pb), pb[i[prim]], 0)
    j = b - NUM_METATILES_IN_PRIMARY
    ok = ~prim & (j < len(sb))
    beh[ok] = sb[j[ok]]
    out = np.where(ts.surfable[beh], WALK_WATER, WALK_LAND).astype(np.uint8)
    out[(raw >> 10) & 3 != 0] = WALK_BLOCKED
    return out


def load_maps(game: str, story: bool):
    groups = json.load(open(os.path.join(game, "data/maps/map_groups.json")))
    layouts = {l["id"]: l for l in json.load(open(os.path.join(game, "data/layouts/layouts.json")))["layouts"]
               if "id" in l}
    raw = []
    for gi, gname in enumerate(groups["group_order"]):
        for mi, folder in enumerate(groups[gname]):
            d = json.load(open(os.path.join(game, "data/maps", folder, "map.json")))
            if gname in SKIP_GROUPS or d["id"].startswith("MAP_PROTOTYPE") or "Unused" in folder:
                continue
            if story and folder.startswith(POSTGAME):
                continue
            raw.append((gi, mi, folder, d))
    ids = {d["id"] for _, _, _, d in raw}
    maps = {}
    for gi, mi, folder, d in raw:
        lay = layouts[d["layout"]]
        maps[d["id"]] = dict(id=d["id"], folder=folder, group=gi, num=mi, w=lay["width"], h=lay["height"],
                             layout=lay, mapsec=d.get("region_map_section", ""), type=d.get("map_type", ""),
                             conns=[c for c in (d.get("connections") or [])
                                    if c["direction"] in ("up", "down", "left", "right") and c["map"] in ids],
                             warps=d.get("warp_events") or [])
    for m in maps:
        copy = m + "_CONNECTION"
        if copy in maps:
            maps[m]["twin"] = (copy, *twin_offset(game, maps[m]["layout"], maps[copy]["layout"]))
    return maps


def twin_offset(game: str, lay: dict, sub: dict):
    """(dx, dy) of the only place where the blocks of layout sub equal those of layout lay (Saffron City's
    48x40 connection copy at (10, 7) in its 66x55 map, whose margins repeat the edges of Routes 5-8)."""
    A = np.fromfile(os.path.join(game, lay["blockdata_filepath"]), "<u2")[:lay["width"] * lay["height"]]
    B = np.fromfile(os.path.join(game, sub["blockdata_filepath"]), "<u2")[:sub["width"] * sub["height"]]
    A, B = A.reshape(lay["height"], lay["width"]) & 0x3FF, B.reshape(sub["height"], sub["width"]) & 0x3FF
    hits = [(dx, dy) for dy in range(A.shape[0] - B.shape[0] + 1) for dx in range(A.shape[1] - B.shape[1] + 1)
            if np.array_equal(A[dy:dy + B.shape[0], dx:dx + B.shape[1]], B)]
    if len(hits) != 1:
        raise SystemExit(f"{lay['id']}: {len(hits)} places match {sub['id']}, expected 1")
    return hits[0]


def neighbour_pos(a: dict, ax: int, ay: int, c: dict, b: dict):
    o, dr = c["offset"], c["direction"]
    if dr == "up":
        return ax + o, ay - b["h"]
    if dr == "down":
        return ax + o, ay + a["h"]
    if dr == "left":
        return ax - b["w"], ay + o
    return ax + a["w"], ay + o


def place_connected(maps: dict):
    """Components of the connection graph, each with its maps' relative positions. A map with a connection
    copy (Saffron City vs the Saffron City copy the routes connect to, nothing connects to the city itself) is
    placed so that its copy's blocks coincide with the copy; another map whose connections lead to an already
    placed map that does not lead back to it is placed from that neighbour. Both join the neighbour's component."""
    pos, comp_of, comps = {}, {}, []
    order = ["MAP_PALLET_TOWN"] + sorted(m for m in maps if m != "MAP_PALLET_TOWN")
    for start in order:
        if start in pos or not maps[start]["conns"]:
            continue
        twin = maps[start].get("twin")
        anchor = next((c for c in maps[start]["conns"] if c["map"] in pos), None)
        if twin is not None and twin[0] in pos:
            pos[start] = (pos[twin[0]][0] - twin[1], pos[twin[0]][1] - twin[2])
            ci = comp_of[twin[0]]
        elif anchor is not None:  # place from the neighbour (the reverse of start's connection to it)
            nb = maps[anchor["map"]]
            bx, by = pos[anchor["map"]]
            # start -> nb is at neighbour_pos(start, sx, sy): solve for (sx, sy)
            dx, dy = neighbour_pos(maps[start], 0, 0, anchor, nb)
            pos[start] = (bx - dx, by - dy)
            ci = comp_of[anchor["map"]]
        else:
            pos[start] = (0, 0)
            ci = len(comps)
            comps.append([])
        comps[ci].append(start)
        comp_of[start] = ci
        q = collections.deque([start])
        while q:
            m = q.popleft()
            x, y = pos[m]
            for c in maps[m]["conns"]:
                n = c["map"]
                if n in pos:
                    continue
                pos[n] = neighbour_pos(maps[m], x, y, c, maps[n])
                comp_of[n] = ci
                comps[ci].append(n)
                q.append(n)
    return pos, comps


def bbox(rects):
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    return x0, y0, x1, y1


def shelf(items, max_w):
    """Rows of (key, w, h) items in order, each row at most max_w wide; returns {key: (x, y)}, width, height."""
    out, x, y, row_h, width = {}, 0, 0, 0, 0
    for key, w, h in items:
        if x > 0 and x + w > max_w:
            x, y, row_h = 0, y + row_h, 0
        out[key] = (x, y)
        x += w
        row_h = max(row_h, h)
        width = max(width, x)
    return out, width, y + row_h


class Grid:
    """Occupancy at metatile resolution; finds the free spot for a w x h rectangle nearest to a point."""

    def __init__(self, w, h):
        self.occ = np.zeros((h, w), bool)

    def mark(self, x, y, w, h):
        self.occ[max(y, 0):y + h, max(x, 0):x + w] = True

    def nearest_free(self, w, h, cx, cy):
        H, W = self.occ.shape
        s = np.zeros((H + 1, W + 1), np.int32)
        s[1:, 1:] = self.occ.cumsum(0).cumsum(1)
        if h > H or w > W:
            return None
        win = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]  # [y, x] = occupied cells of the w x h at (x, y)
        ys, xs = np.nonzero(win == 0)
        if ys.size == 0:
            return None
        d = (xs + w / 2 - cx) ** 2 + (ys + h / 2 - cy) ** 2
        i = int(np.argmin(d))
        return int(xs[i]), int(ys[i])


def build(game: str, ts: Tilesets, out: str, story: bool):
    t0 = time.time()
    maps = load_maps(game, story)
    pos, comps = place_connected(maps)

    # anchors: Kanto (the component with Pallet Town) at the origin, the other components (Sevii Islands)
    # in rows in a band east of it, in island order
    # a town without connections (Four Island) is an island of its own
    for m in sorted(maps):
        if m not in pos and maps[m]["type"] in ("MAP_TYPE_TOWN", "MAP_TYPE_CITY"):
            pos[m] = (0, 0)
            comps.append([m])
    kanto = comps[0]
    kx0, ky0, kx1, ky1 = bbox([(*pos[m], maps[m]["w"], maps[m]["h"]) for m in kanto])
    world = {m: (pos[m][0] - kx0, pos[m][1] - ky0) for m in kanto}
    island = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN"]

    def comp_key(c):
        names = " ".join(c)
        return min([i for i, n in enumerate(island) if f"_{n}_ISLAND" in names + "_"] or [99]), sorted(c)[0]

    others = sorted(comps[1:], key=comp_key)
    items = []
    for ci, c in enumerate(others):
        x0, y0, x1, y1 = bbox([(*pos[m], maps[m]["w"], maps[m]["h"]) for m in c])
        items.append((ci, x1 - x0 + SEVII_GAP, y1 - y0 + SEVII_GAP, x0, y0))
    rows, _, _ = shelf([(k, w, h) for k, w, h, _, _ in items], (kx1 - kx0) * 3 // 4)
    for k, w, h, x0, y0 in items:
        ox, oy = rows[k]
        for m in others[k]:
            world[m] = (pos[m][0] - x0 + ox + (kx1 - kx0) + 4 * SEVII_GAP, pos[m][1] - y0 + oy)

    # doors: the world tile of the warp that leads into each unplaced map, spread through the warps
    # between unplaced maps (upper floors get their building's door)
    door = {}
    q = collections.deque()
    for m in list(world):
        for wv in maps[m]["warps"]:
            d = wv["dest_map"]
            if d in maps and d not in world and d not in door:
                door[d] = (world[m][0] + wv["x"], world[m][1] + wv["y"])
                q.append(d)
    while q:
        m = q.popleft()
        for wv in maps[m]["warps"]:
            d = wv["dest_map"]
            if d in maps and d not in world and d not in door:
                door[d] = door[m]
                q.append(d)
    # also maps that only lead *to* placed or doored maps (one-way warps): use the map they lead to
    for m in maps:
        if m in world or m in door:
            continue
        for wv in maps[m]["warps"]:
            d = wv["dest_map"]
            if d in door:
                door[m] = door[d]
                break
            if d in world:
                door[m] = (world[d][0] + maps[d]["w"] // 2, world[d][1] + maps[d]["h"] // 2)
                break

    # groups of the remaining maps by region map section, each laid out in rows (a label row above each map)
    rest = [m for m in maps if m not in world]
    by_sec = collections.defaultdict(list)
    for m in rest:
        by_sec[maps[m]["mapsec"]].append(m)
    groups = []
    for sec, ms in by_sec.items():
        ms.sort(key=lambda m: (maps[m]["group"], maps[m]["num"]))
        cells = [(m, maps[m]["w"] + 1, maps[m]["h"] + 2) for m in ms]  # +1 gap, +1 label row +1 gap
        area = sum(w * h for _, w, h in cells)
        target = max(max(w for _, w, _ in cells), int((area * 1.6) ** 0.5))
        lay, gw, gh = shelf(cells, target)
        ds = [door[m] for m in ms if m in door]
        home = tuple(np.mean(ds, axis=0)) if ds else None
        groups.append(dict(sec=sec, maps=ms, lay=lay, w=gw, h=gh, home=home))
        if home is None:
            print(f"  no door into {sec}: {', '.join(ms)}")

    # place the groups: biggest first, in the free spot nearest to their doors
    used = [(x, y, maps[m]["w"], maps[m]["h"]) for m, (x, y) in world.items()]
    ux0, uy0, ux1, uy1 = bbox(used)
    margin = 160
    grid = Grid(ux1 - ux0 + 2 * margin, uy1 - uy0 + 2 * margin)
    shift = (margin - ux0, margin - uy0)
    for x, y, w, h in used:
        grid.mark(x + shift[0] - GAP, y + shift[1] - GAP, w + 2 * GAP, h + 2 * GAP)
    fallback = (shift[0] + ux1 - ux0, shift[1] + (uy1 - uy0) / 2)
    labels = []
    for g in sorted(groups, key=lambda g: -g["w"] * g["h"]):
        cx, cy = (g["home"][0] + shift[0], g["home"][1] + shift[1]) if g["home"] else fallback
        p = grid.nearest_free(g["w"] + 2 * GAP, g["h"] + 2 * GAP, cx, cy)
        if p is None:
            raise SystemExit(f"no room for group {g['sec']} ({g['w']}x{g['h']}): raise the margin")
        gx, gy = p[0] + GAP, p[1] + GAP
        grid.mark(gx - GAP, gy - GAP, g["w"] + 2 * GAP, g["h"] + 2 * GAP)
        names = [pretty(maps[m]["folder"]).split() for m in g["maps"]]
        common = 0  # words shared by all the group's names (the city), dropped from a label that does not fit
        while len(names) > 1 and all(len(n) > common + 1 and n[common] == names[0][common] for n in names):
            common += 1
        for m, n in zip(g["maps"], names):
            ox, oy = g["lay"][m]
            world[m] = (gx + ox - shift[0], gy + oy + 1 - shift[1])  # below its label row
            labels.append((m, " ".join(n), " ".join(n[common:])))

    # crop to the used area, render, cut into chunks
    x0, y0, x1, y1 = bbox([(x, y - 1, maps[m]["w"], maps[m]["h"] + 1) for m, (x, y) in world.items()])
    x0, y0 = x0 - 1, y0 - 1
    W, H = x1 - x0 + 1, y1 - y0 + 1
    world = {m: (x - x0, y - y0) for m, (x, y) in world.items()}
    img = np.zeros((H * TILE, W * TILE, 3), np.uint8)
    walk = np.full((H, W), WALK_NONE, np.uint8)
    # draw order: a map with a connection copy first (Saffron City: its margins repeat the edges of Routes 5-8
    # imperfectly, so the routes and the copy, identical to the city's core, draw over it)
    for m in sorted(world, key=lambda m: "twin" not in maps[m]):
        x, y = world[m]
        img[y * TILE:(y + maps[m]["h"]) * TILE, x * TILE:(x + maps[m]["w"]) * TILE] = render_layout(ts, game, maps[m]["layout"])
        walk[y:y + maps[m]["h"], x:x + maps[m]["w"]] = walk_layout(ts, game, maps[m]["layout"])
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=12)
    for m, full, short in labels:
        x, y = world[m]
        room = (maps[m]["w"] + 1) * TILE - 4  # up to the next map of the row
        text = full if draw.textlength(full, font=font) <= room else short
        while len(text) > 1 and draw.textlength(text, font=font) > room:
            text = text[:-2] + "…"
        draw.text((x * TILE + 1, (y - 1) * TILE + 1), text, fill=(235, 235, 235), font=font)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if re.fullmatch(r"chunk_\d+_\d+\.png", f):
            os.remove(os.path.join(out, f))
    chunks = []
    full = np.asarray(pil)
    for r in range(0, full.shape[0], CHUNK):
        for c in range(0, full.shape[1], CHUNK):
            part = full[r:r + CHUNK, c:c + CHUNK]
            if not part.any():
                continue
            name = f"chunk_{r // CHUNK}_{c // CHUNK}.png"
            Image.fromarray(part).save(os.path.join(out, name), optimize=True)
            chunks.append({"file": name, "x": c, "y": r, "w": part.shape[1], "h": part.shape[0]})
    walk.tofile(os.path.join(out, "walk.bin"))
    down = max(4, -(-max(W, H) * TILE // OVERVIEW_MAX))
    pil.resize((W * TILE // down, H * TILE // down), Image.BOX).save(os.path.join(out, "overview.png"), optimize=True)

    regions = [{"id": str(maps[m]["group"] * 256 + maps[m]["num"]), "name": pretty(maps[m]["folder"]),
                "coordinates": list(world[m]), "tileSize": [maps[m]["w"], maps[m]["h"]]}
               for m in sorted(world, key=lambda m: (not m.endswith("_CONNECTION"), maps[m]["group"], maps[m]["num"]))]
    meta = {"version": int(time.time()), "tile": TILE, "width": W * TILE, "height": H * TILE,
            "kanto": [-x0 * TILE, -y0 * TILE, (kx1 - kx0) * TILE, (ky1 - ky0) * TILE],
            "overview": {"file": "overview.png", "scale": down}, "walk": "walk.bin", "chunks": chunks, "regions": regions}
    with open(os.path.join(out, "world.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"{len(regions)} maps ({len(kanto)} Kanto outdoor, {sum(len(c) for c in others)} Sevii outdoor, "
          f"{len(labels)} indoor/cave in {len(groups)} groups, {sum(1 for g in groups if not g['home'])} without a door); "
          f"world {W}x{H} metatiles = {W * TILE}x{H * TILE} px, {len(chunks)} chunks; {time.time() - t0:.1f} s -> {out}")


def build_icons(game: str, path: str):
    """icons.png: gMonIconTable's icon of every species SPECIES_NONE..SPECIES_EGG in gMonIconPaletteIndices' palette."""
    ids = {k: int(v) for k, v in re.findall(r"#define (SPECIES_\w+) (\d+)\b", open(os.path.join(game, "include/constants/species.h")).read())}
    src = open(os.path.join(game, "src/pokemon_icon.c")).read()
    table = lambda name: dict(re.findall(r"\[(SPECIES_\w+)\]\s*=\s*(\w+)", src.split(name, 1)[1].split("};", 1)[0]))
    icon, pal = table("gMonIconTable[]"), table("gMonIconPaletteIndices[]")
    files = {}  # setdefault: Deoxys' first (normal form) definition
    for f in ("src/data/graphics/pokemon.h", "src/graphics.c"):
        for k, v in re.findall(r"(gMonIcon_\w+)\[\] = INCBIN_U8\(\s*\"([^\"]+)\.4bpp\"", open(os.path.join(game, f)).read()):
            files.setdefault(k, v)
    pals = []
    for i in range(3):
        rows = open(os.path.join(game, f"graphics/pokemon/icon_palettes/icon_palette_{i}.pal")).read().split("\n")[3:19]
        pals.append([c for r in rows for c in map(int, r.split())])
    n = ids["SPECIES_EGG"] + 1
    sheet = Image.new("RGBA", (ICON_COLS * 32, (n + ICON_COLS - 1) // ICON_COLS * 32))
    for sp, s in sorted((s, k) for k, s in ids.items() if s < n):
        if s not in icon or icon[s] not in files:
            continue
        im = Image.open(os.path.join(game, files[icon[s]] + ".png")).crop((0, 0, 32, 32))
        im.putpalette(pals[int(pal.get(s, 0))])
        im.info["transparency"] = 0
        sheet.paste(im.convert("RGBA"), (sp % ICON_COLS * 32, sp // ICON_COLS * 32))
    sheet.save(path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default=os.path.join(here, "..", "..", "game"), help="pokefirered decomp dir")
    ap.add_argument("--out", default=os.path.join(here, "www", "world"))
    ap.add_argument("--variants", default="story,full", help="comma-separated: story, full")
    args = ap.parse_args()
    game = os.path.abspath(args.game)
    ts = Tilesets(game)
    for v in filter(None, args.variants.split(",")):
        if v not in ("story", "full"):
            raise SystemExit(f"unknown variant {v!r}")
        print(f"{v}:")
        build(game, ts, os.path.join(args.out, v), story=v == "story")
    # the player's overworld frames 0-2 (down, up, left; right = left mirrored), palette index 0 transparent
    sheet = Image.open(os.path.join(game, "graphics/object_events/pics/people/red_normal.png"))
    sheet.crop((0, 0, 48, 32)).convert("RGBA").save(os.path.join(args.out, "player.png"))
    build_icons(game, os.path.join(args.out, "icons.png"))


if __name__ == "__main__":
    main()
