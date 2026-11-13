"""Relay scopes: regions of the game a life is confined to while it works on them (env.py "Relay scopes").

A :class:`Scope` names a **home** map, an **entry** condition, overworld **anchors**, a **clear** condition and
optional ``include`` / ``exclude`` map lists, porch rectangles and detour anchors. Per life every scope is INACTIVE
(0), ACTIVE (1) or CLEARED (2) (``int8 [n, n_scopes]`` status rows, kept by the env and carried by rewinds).

Predicates. An entry or clear condition is a conjunction of event NAMES of ``events.EVENT_DEFS`` (e.g.
``("badge4", "silph_scope", "tea")``), resolved to column indices of ``events.EVENTS`` (:func:`resolve_predicate`)
and evaluated over the CURRENT record's events, ``events.observe(obs)`` = bool ``[n, events.NUM_EVENTS]``
(:func:`holds`), NOT over the env's latched per-life matrix ``life_events``: a rewind restores the game state but
never rolls ``life_events`` back, so a life that cleared a scope and rewound to before its clear (status ACTIVE
again, restored from the rewind capture) would re-clear at once on the latched matrix. The empty conjunction is
always true. Predicates may use flag, var, item and move events only, which describe the game state a restored
snapshot carries; ``map`` events are rejected (a scope expresses location by its home and area).

Area ("anchors + warp closure"). From every anchor the loader follows ``warp_events[].dest_map`` of
game/data/maps/*/map.json transitively and adds each reached map that has NO ``connections`` (indoor, underground,
caves, and the Safari Zone's four areas, which are ``MAP_TYPE_ROUTE`` without connections); it stops at (does not
add or cross) a map with connections (an overworld map), except the anchors themselves. ``MAP_DYNAMIC`` warps
(elevators, exits to the last map) are not followed; every floor they serve is also reached by stairs.
``MAP_SAFFRON_CITY_CONNECTION`` (the town-type helper the routes around Saffron connect to) joins whenever
``MAP_SAFFRON_CITY`` is an anchor (``HELPER_MAPS``). Then ``include`` is added and ``exclude`` removed. The map data
is read lazily once per game directory (:func:`map_data`, cached) and every list is sorted, so the result is
deterministic. Noise kept as is: every Pokemon Center 2F leads to ``MAP_TRADE_CENTER`` / ``MAP_UNION_ROOM`` (not
enterable in a single-player run), Vermilion's closure crosses Diglett's Cave to its north entrance building (Route
2 itself is out). Viridian's runs from Cinnabar Island (Route 21), Pallet Town and Route 1 to the Pokemon League:
Route 22, Route 23, Victory Road, the Indigo Plateau and, through its Pokemon Center's warps, the Elite Four rooms,
the Champion's room and the Hall of Fame; it activates on Viridian City, its home, with all seven badges 1-7 (what
the Viridian Gym door and the Route 22 / 23 guards check). It holds Cinnabar's home: allowed because Cinnabar's
clear (badge7) is part of Viridian's entry, so Cinnabar is cleared before Viridian can activate (validation below).

Final scope (``Scope.final``: Viridian, clear = the Champion defeated): its clear is the game's goal, which ends the
life on that step (the Champion end, env.py).

Ordered activation (``ScopeSet(ordered=True)``): a scope activates only when every scope BEFORE it in :data:`ORDER`
(the game's own order: Vermilion, Celadon, Fuchsia, Saffron, Cinnabar, Viridian) is CLEARED. ``ScopeSet(confine=
False)`` would report no area exits (the env always confines).

Activation anywhere in the area: ``Scope.activate = "area"`` lets a scope activate on every map of its area instead
of its home map alone (``ScopeSet.act_table``, ``ScopeSet.may_activate``); no scope uses it.

Detour: ``Scope.detour`` names overworld anchors of a region outside the area that a confined life may visit and
come back from; :func:`detour_maps` closes them through warps AND connections (:func:`closure` with ``links``). A
confined life with an ACTIVE scope on a detour map is not ``exited`` (``ScopeSet.detour_table``, used by
:meth:`ScopeSet.step` next to the area): its scope stays ACTIVE, nothing is masked there (the exit tables are the
area's), no scope activates there, and the area test confines it again on its first step back on an area map.
Viridian's detour is the Sevii Islands 1-3 (anchors One, Two and Three Island): Bill's boat sails from Cinnabar
Island (inside Viridian's area) to One Island once Blaine is beaten, and the sailors offer islands 1-3 only until
Bill's boat lands the life back on Cinnabar Island, a closed loop. The area's own maps are left out of the detour
(the Trade Center and the Union Room behind every Center's 2F are reached from both). Validation: detour anchors
known, no scope's home in a detour.

Porch: ``{MAP_*: (x0, y0, x1, y1)}`` inclusive tile rectangles of maps outside the area that count as inside.
:meth:`ScopeSet.porch_table` returns them as four int64 ``[n_scopes, MAP_COUNT + 1]`` tables (-1 = none). The six
scopes have none.

Status transitions (:meth:`ScopeSet.step`), per row and step, on the status at the step's start:

* a row with an ACTIVE scope ``a``: if ``a``'s clear condition holds, ``a`` becomes CLEARED (``cleared = a``; the
  clear is checked before the area on the same step, so clearing while stepping out is a clear, not an exit);
  otherwise, if the map is outside ``a``'s area, porch and detour, ``exited`` (the status is kept; the env ends the
  life);
* a row without an ACTIVE scope: a scope that may activate on the row's map activates (``activated = s``) iff it is
  INACTIVE and its entry condition holds (and, ``ordered``, every scope before it in :data:`ORDER` is CLEARED); when
  two could, the earlier in :data:`ORDER` wins. Evaluated every step, not only on arrival. CLEARED never
  re-activates.
* at most one scope is ACTIVE per row, and a row changes at most one scope per step: a row that clears on step t can
  activate another scope from step t + 1 on.
* rows with ``ingame`` False (no INGAME bit) keep their status and report nothing.

Validation at load (:meth:`ScopeSet.validate`, ValueError listing every problem): known map names, valid porch
rectangles, unique names and homes, predicates resolve and use no ``map`` events (location is the home's and the
area's job), non-empty clear conditions, each home inside its own area and outside every other scope's area and porch
maps (the activation rule relies on it) unless that other scope's clear condition is part of this scope's entry condition (it is
cleared before this one can activate, and CLEARED never re-activates: Cinnabar inside Viridian), every area with a
Pokémon Center map (name contains ``POKEMON_CENTER``) and a map with FireRed land encounters
(game/src/data/wild_encounters.json), and a final scope's clear containing ``champion`` (the life ends there).

    PYTHONPATH=rl:python python3 -m pokerl.relay.scopes      # prints every scope's resolved maps (describe())
"""
from __future__ import annotations

import functools
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from .. import events as E
from .. import tables as T
from ..gen_tables import GAME
from ..reward import MAP_COUNT

INACTIVE, ACTIVE, CLEARED = 0, 1, 2
STATUS_NAMES = ("inactive", "active", "cleared")
ACTIVATE_MODES = ("home", "area")  # Scope.activate
CENTER_TAG = "POKEMON_CENTER"
HELPER_MAPS: Dict[str, Tuple[str, ...]] = {"MAP_SAFFRON_CITY": ("MAP_SAFFRON_CITY_CONNECTION",)}
MAP_DENSE: Dict[str, int] = {name: i + 1 for i, (_, name, _, _) in enumerate(T.MAPS)}  # -> reward.MAP_INDEX values
MAP_NAMES: Tuple[str, ...] = ("<unknown>",) + tuple(name for _, name, _, _ in T.MAPS)  # dense index -> MAP_* name
EVENT_COL: Dict[str, int] = {e.name: i for i, e in enumerate(E.EVENTS)}

Rect = Tuple[int, int, int, int]


# ---- scopes -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scope:
    """One scope. Map names are ``MAP_*`` constants; ``entry`` / ``clear`` are
    conjunctions of ``events.EVENT_DEFS`` names (``entry`` empty: always true)."""
    name: str
    home: str
    anchors: Tuple[str, ...]
    clear: Tuple[str, ...]
    entry: Tuple[str, ...] = ()
    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    porch: Mapping[str, Rect] = field(default_factory=dict)
    final: bool = False  # its clear is the game's goal: the life ends there (module docstring "Final scope")
    activate: str = "home"  # where it may activate: "home" (its home map) or "area" (every map of its area)
    detour: Tuple[str, ...] = ()  # anchors of a region a confined life may visit and come back from (no exit)


SCOPES: Tuple[Scope, ...] = (
    # badge 2: Cut, which the gym tree needs, wants the Cascade Badge (field_moves.inc); the S.S. Ticket: HM01 is on
    # the S.S. Anne, whose gangway wants the ticket from Bill (Route 25, outside the area), so a life without it could
    # never clear
    Scope("vermilion", home="MAP_VERMILION_CITY", entry=("badge2", "ss_ticket"),
          anchors=("MAP_VERMILION_CITY", "MAP_ROUTE6", "MAP_ROUTE11"), clear=("badge3",)),
    Scope("celadon", home="MAP_CELADON_CITY", entry=(),
          anchors=("MAP_CELADON_CITY", "MAP_ROUTE7", "MAP_ROUTE16"), clear=("badge4", "silph_scope", "tea")),
    # badge 5: Saffron opens after Fuchsia's gym, not on the Poke Flute alone
    Scope("saffron", home="MAP_SAFFRON_CITY", entry=("poke_flute", "badge5"),
          anchors=("MAP_SAFFRON_CITY", "MAP_ROUTE6", "MAP_ROUTE7", "MAP_ROUTE8"), clear=("badge6",)),
    # HM03 and HM04 in the bag: the flags are set where the HMs are given, and an HM can never leave the bag (no toss,
    # sell, hold or deposit), so flag = HM in the bag
    Scope("fuchsia", home="MAP_FUCHSIA_CITY", entry=(),
          anchors=("MAP_FUCHSIA_CITY", "MAP_ROUTE15", "MAP_ROUTE18"), clear=("hm03_surf", "hm04_strength", "badge5")),
    Scope("cinnabar", home="MAP_CINNABAR_ISLAND", entry=(), anchors=("MAP_CINNABAR_ISLAND",), clear=("badge7",)),
    # south to Cinnabar Island (Route 21), north to the Pokemon League (the Indigo Plateau, whose Pokemon Center warps
    # lead to the Elite Four rooms, the Champion's room and the Hall of Fame); clear = the Champion, a final scope: the
    # life ends there. Entry = seven badges, all of 1-7: the gym door wants badges 2-7
    # (ViridianCity_EventScript_TryUnlockGym), the Route 22 gate badge 1 and the Route 23 guards badges 2-8, and badge
    # 8 cannot come before the other seven; and bill_answered, Bill's One Island offer after Blaine answered (a life
    # with Blaine's badge has passed Cinnabar Island, where the offer fires, so it holds)
    Scope("viridian", home="MAP_VIRIDIAN_CITY",
          entry=("badge1", "badge2", "badge3", "badge4", "badge5", "badge6", "badge7", "bill_answered"),
          anchors=("MAP_VIRIDIAN_CITY", "MAP_PALLET_TOWN", "MAP_ROUTE1", "MAP_ROUTE22", "MAP_ROUTE23",
                   "MAP_ROUTE21_NORTH", "MAP_ROUTE21_SOUTH", "MAP_CINNABAR_ISLAND", "MAP_INDIGO_PLATEAU_EXTERIOR"),
          clear=("champion",), final=True, activate="home",
          # Bill's boat is the one way out that comes back: the Sevii Islands 1-3 are a closed loop back to Cinnabar
          # Island, out there the life is neither exited nor masked; its scope stays ACTIVE and confines it again on
          # landing. Islands 4-7 are offered only once the detour is done (not reachable from here). It serves a life
          # that said NO to Bill, activated Viridian on Viridian City and then says YES to Bill waiting in the
          # Cinnabar Center.
          detour=("MAP_ONE_ISLAND", "MAP_TWO_ISLAND", "MAP_THREE_ISLAND")),
)
SCOPE_NAMES: Tuple[str, ...] = tuple(s.name for s in SCOPES)
# the game's order of the scopes (``ScopeSet(ordered=True)``): Fuchsia before Saffron, whose entry wants Koga's badge
# (poke_flute & badge5)
ORDER: Tuple[str, ...] = ("vermilion", "celadon", "fuchsia", "saffron", "cinnabar", "viridian")


# ---- predicates ---------------------------------------------------------------------------------------------

class Predicate(NamedTuple):
    """A resolved conjunction: the event names and their ``events.EVENTS`` columns (int64)."""
    names: Tuple[str, ...]
    cols: np.ndarray


def resolve_predicate(names: Sequence[str]) -> Predicate:
    """The columns of a conjunction of event names (ValueError naming the unknown ones)."""
    names = tuple(names)
    bad = [n for n in names if n not in EVENT_COL]
    if bad:
        raise ValueError(f"unknown event name(s) {', '.join(map(repr, bad))} (events.EVENT_DEFS names: "
                         f"{', '.join(EVENT_COL)})")
    return Predicate(names, np.array([EVENT_COL[n] for n in names], dtype=np.int64))


def holds(pred: Union[Predicate, Sequence[str], np.ndarray], ev: np.ndarray) -> np.ndarray:
    """bool [n]: every event of the conjunction ``pred`` (a :class:`Predicate`, event names or column indices) is
    true in the rows of ``ev`` = bool [n, NUM_EVENTS] (the current records' ``events.observe(obs)``, not the latched
    ``life_events``: module docstring). Empty: all True."""
    ev = np.asarray(ev, dtype=bool)
    if ev.ndim != 2 or ev.shape[1] != E.NUM_EVENTS:
        raise ValueError(f"event matrix of shape {ev.shape}, expected [n, {E.NUM_EVENTS}]")
    if isinstance(pred, Predicate):
        cols = pred.cols
    elif isinstance(pred, np.ndarray) and pred.dtype.kind in "iu":
        cols = pred.astype(np.int64)
    elif len(pred) and all(isinstance(p, (int, np.integer)) for p in pred):
        cols = np.asarray(pred, dtype=np.int64)
    else:
        cols = resolve_predicate(pred).cols
    if cols.size == 0:
        return np.ones(ev.shape[0], dtype=bool)
    return ev[:, cols].all(axis=1)


def _req_matrix(preds: Sequence[Predicate]) -> np.ndarray:
    """bool [len(preds), NUM_EVENTS]: the events each conjunction requires."""
    r = np.zeros((len(preds), E.NUM_EVENTS), dtype=bool)
    for i, p in enumerate(preds):
        r[i, p.cols] = True
    return r


def active_scope(status: np.ndarray) -> np.ndarray:
    """int64 [n]: the ACTIVE scope of each status row, -1 where none is."""
    a = np.asarray(status) == ACTIVE
    if a.ndim != 2:
        raise ValueError(f"status of shape {a.shape}, expected [n, n_scopes]")
    if a.shape[1] == 0:
        return np.full(a.shape[0], -1, dtype=np.int64)
    return np.where(a.any(axis=1), a.argmax(axis=1), -1).astype(np.int64)


# ---- map data -----------------------------------------------------------------------------------------------

class MapInfo(NamedTuple):
    """One map of game/data/maps: its type, whether it is an overworld map and its warp targets."""
    name: str                 # MAP_*
    map_type: str             # MAP_TYPE_*
    connected: bool           # has a non-empty ``connections`` list (an overworld map)
    warps: Tuple[str, ...]    # warp_events[].dest_map (MAP_DYNAMIC dropped), file order
    links: Tuple[str, ...] = ()  # connections[].map (the overworld neighbours), file order


class MapData(NamedTuple):
    """The maps by name, the maps with land encounters and the game directory they were read from."""
    maps: Dict[str, MapInfo]
    land: FrozenSet[str]      # maps with FireRed land encounters
    root: str                 # the game directory read


@functools.lru_cache(maxsize=None)
def map_data(game_dir: Optional[str] = None) -> MapData:
    """The warp graph of game/data/maps/*/map.json and the land-encounter maps of
    game/src/data/wild_encounters.json (FireRed headers) of ``game_dir`` (default: the repository's game/)."""
    root = os.path.abspath(game_dir or GAME)
    maps: Dict[str, MapInfo] = {}
    paths = sorted(glob.glob(os.path.join(root, "data", "maps", "*", "map.json")))
    if not paths:
        raise FileNotFoundError(f"no map.json under {os.path.join(root, 'data', 'maps')}")
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        warps = tuple(w["dest_map"] for w in d.get("warp_events") or [] if w.get("dest_map") != "MAP_DYNAMIC")
        links = tuple(c["map"] for c in d.get("connections") or [] if c.get("map"))
        maps[d["id"]] = MapInfo(d["id"], d.get("map_type", ""), bool(d.get("connections")), warps, links)
    with open(os.path.join(root, "src", "data", "wild_encounters.json"), encoding="utf-8") as f:
        groups = json.load(f)["wild_encounter_groups"]
    land = frozenset(e["map"] for g in groups for e in g.get("encounters", [])
                     if e.get("base_label", "").endswith("_FireRed") and e.get("land_mons"))
    return MapData(maps, land, root)


def closure(anchors: Sequence[str], data: Optional[MapData] = None, links: bool = False) -> Tuple[str, ...]:
    """The anchors plus every map reachable from them through warps without entering a map with connections
    (module docstring), sorted. With ``links`` the connections are followed too and connected maps are entered:
    everything reachable on foot and through warps (a detour region, ``Scope.detour``). Unknown anchors raise
    ValueError, unknown warp / connection targets are skipped."""
    data = data or map_data()
    maps = data.maps
    bad = [a for a in anchors if a not in maps]
    if bad:
        raise ValueError(f"unknown anchor map(s) {', '.join(bad)}")
    seen = set(anchors)
    stack = sorted(seen)
    while stack:
        m = stack.pop()
        for t in maps[m].warps + (maps[m].links if links else ()):
            if t in seen or t not in maps or (maps[t].connected and not links):
                continue
            seen.add(t)
            stack.append(t)
    return tuple(sorted(seen))


def detour_maps(scope: Scope, data: Optional[MapData] = None) -> Tuple[str, ...]:
    """A scope's detour (module docstring "Detour"): the closure of ``Scope.detour`` through warps AND connections,
    minus the scope's own area (maps shared through warps, such as the Trade Center and the Union Room behind every
    Pokémon Center's 2F, stay area maps); sorted, empty without anchors."""
    if not scope.detour:
        return ()
    return tuple(sorted(set(closure(scope.detour, data, links=True)) - set(area_maps(scope, data))))


def area_maps(scope: Scope, data: Optional[MapData] = None) -> Tuple[str, ...]:
    """A scope's area: closure of the anchors, the helper maps of its anchors (``HELPER_MAPS``), ``include``, minus
    ``exclude``; sorted."""
    s = set(closure(scope.anchors, data))
    for a in scope.anchors:
        s.update(HELPER_MAPS.get(a, ()))
    s.update(scope.include)
    s.difference_update(scope.exclude)
    return tuple(sorted(s))


# ---- the scope set ------------------------------------------------------------------------------------------

class ScopeSet:
    """The scopes of a run with their resolved areas, predicates and lookup tables (validated on construction).

    ``names``, ``n``; ``areas`` (sorted MAP_* names per scope); ``home_index`` int64 [n] (dense map index of each
    home); ``home_scope`` int64 [MAP_COUNT + 1] (the scope whose home a dense map is, -1); ``act_table`` bool
    [n, MAP_COUNT + 1] (the scope may activate on the dense map: its home, or its whole area with
    ``Scope.activate`` "area"); ``may_activate`` bool [MAP_COUNT + 1] (some scope may); ``entry`` / ``clear``
    (:class:`Predicate` per scope), ``entry_cols`` / ``clear_cols`` (their column arrays), ``entry_req`` /
    ``clear_req`` bool [n, NUM_EVENTS]; ``final`` bool [n] (``Scope.final``)."""

    def __init__(self, scopes: Sequence[Scope] = SCOPES, data: Optional[MapData] = None, validate: bool = True,
                 ordered: bool = False, confine: bool = True, order: Optional[Sequence[str]] = None):
        self.scopes: Tuple[Scope, ...] = tuple(scopes)
        self.data = data or map_data()
        self.names: Tuple[str, ...] = tuple(s.name for s in self.scopes)
        self.n = len(self.scopes)
        self.ordered = bool(ordered)
        self.confine = bool(confine)
        # rank: position in ``order`` (default ORDER); names not in it follow, in list order (module docstring)
        order = tuple(nm for nm in (ORDER if order is None else order) if nm in self.names)
        order += tuple(nm for nm in self.names if nm not in order)
        self.order: Tuple[str, ...] = order
        self.rank = np.array([order.index(nm) for nm in self.names], dtype=np.int64)
        self.rank.setflags(write=False)
        self.before = self.rank[None, :] < self.rank[:, None]  # before[s, t]: t comes before s
        self.before.setflags(write=False)
        self.final = np.array([bool(s.final) for s in self.scopes], dtype=bool)
        self.final.setflags(write=False)
        self._problems: List[str] = []
        self.areas: Tuple[Tuple[str, ...], ...] = tuple(self._area(s) for s in self.scopes)
        self.detours: Tuple[Tuple[str, ...], ...] = tuple(self._detour(s) for s in self.scopes)
        self.entry: Tuple[Predicate, ...] = tuple(self._pred(s, "entry") for s in self.scopes)
        self.clear: Tuple[Predicate, ...] = tuple(self._pred(s, "clear") for s in self.scopes)
        self.entry_cols = tuple(p.cols for p in self.entry)
        self.clear_cols = tuple(p.cols for p in self.clear)
        self.entry_req = _req_matrix(self.entry)
        self.clear_req = _req_matrix(self.clear)
        self.home_index = np.array([MAP_DENSE.get(s.home, 0) for s in self.scopes], dtype=np.int64)
        self.home_scope = np.full(MAP_COUNT + 1, -1, dtype=np.int64)
        for i in range(self.n - 1, -1, -1):  # (duplicate homes are a validation error; the first one wins here)
            if self.home_index[i] > 0:
                self.home_scope[self.home_index[i]] = i
        self._area_t: Optional[np.ndarray] = None
        self._detour_t: Optional[np.ndarray] = None
        self._porch_t: Optional[Tuple[np.ndarray, ...]] = None
        self.has_porch = any(s.porch for s in self.scopes)
        self.act_table = np.zeros((self.n, MAP_COUNT + 1), dtype=bool)
        for i, s in enumerate(self.scopes):
            if s.activate == "area":
                self.act_table[i] = self.area_table()[i]
            elif self.home_index[i] > 0:
                self.act_table[i, self.home_index[i]] = True
        self.act_table.setflags(write=False)
        self.may_activate = self.act_table.any(axis=0)
        self.may_activate.setflags(write=False)
        if validate:
            self.validate()

    def _area(self, s: Scope) -> Tuple[str, ...]:
        try:
            return area_maps(s, self.data)
        except ValueError as e:
            self._problems.append(f"{s.name}: {e}")
            return ()

    def _detour(self, s: Scope) -> Tuple[str, ...]:
        try:
            return detour_maps(s, self.data)
        except ValueError as e:
            self._problems.append(f"{s.name}: detour: {e}")
            return ()

    def _pred(self, s: Scope, which: str) -> Predicate:
        try:
            return resolve_predicate(getattr(s, which))
        except ValueError as e:
            self._problems.append(f"{s.name}: {which}: {e}")
            return Predicate((), np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return self.n

    def index(self, name: str) -> int:
        return self.names.index(name)

    # ---- tables -----------------------------------------------------------------------------------------

    def area_table(self) -> np.ndarray:
        """bool [n, MAP_COUNT + 1]: the dense map index (``reward.MAP_INDEX``) is inside the scope's area (column
        0, an unknown map, is outside for every scope)."""
        if self._area_t is None:
            t = np.zeros((self.n, MAP_COUNT + 1), dtype=bool)
            for i, maps in enumerate(self.areas):
                for m in maps:
                    if m in MAP_DENSE:
                        t[i, MAP_DENSE[m]] = True
            t[:, 0] = False
            t.setflags(write=False)
            self._area_t = t
        return self._area_t

    def detour_table(self) -> np.ndarray:
        """bool [n, MAP_COUNT + 1]: the dense map is in the scope's detour (module docstring "Detour"): a confined
        life there is not outside (no exit), though not inside the area either (nothing masked, no activation)."""
        if self._detour_t is None:
            t = np.zeros((self.n, MAP_COUNT + 1), dtype=bool)
            for i, maps in enumerate(self.detours):
                for m in maps:
                    if m in MAP_DENSE:
                        t[i, MAP_DENSE[m]] = True
            t[:, 0] = False
            t.setflags(write=False)
            self._detour_t = t
        return self._detour_t

    def porch_table(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(x0, y0, x1, y1) int64 [n, MAP_COUNT + 1] each: the porch rectangle (inclusive) of a scope on a map, -1
        where there is none."""
        if self._porch_t is None:
            t = [np.full((self.n, MAP_COUNT + 1), -1, dtype=np.int64) for _ in range(4)]
            for i, s in enumerate(self.scopes):
                for m, r in s.porch.items():
                    if m in MAP_DENSE and len(r) == 4:
                        for k in range(4):
                            t[k][i, MAP_DENSE[m]] = int(r[k])
            for a in t:
                a.setflags(write=False)
            self._porch_t = tuple(t)
        return self._porch_t

    def in_porch(self, scope: np.ndarray, map_idx: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """bool [k]: tile (x, y) of dense map ``map_idx`` lies in scope ``scope``'s porch rectangle of that map."""
        x0, y0, x1, y1 = self.porch_table()
        s, m = np.asarray(scope, dtype=np.int64), np.asarray(map_idx, dtype=np.int64)
        x, y = np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)
        a0 = x0[s, m]
        return (a0 >= 0) & (x >= a0) & (x <= x1[s, m]) & (y >= y0[s, m]) & (y <= y1[s, m])

    def inside(self, scope: np.ndarray, map_idx: np.ndarray, x: Optional[np.ndarray] = None,
               y: Optional[np.ndarray] = None) -> np.ndarray:
        """bool [k]: dense map ``map_idx`` (tile (x, y), for the porch) is inside scope ``scope``'s area."""
        s, m = np.asarray(scope, dtype=np.int64), np.asarray(map_idx, dtype=np.int64)
        out = self.area_table()[s, m]
        if self.has_porch:
            if x is None or y is None:
                raise ValueError("ScopeSet.inside: the scopes have porch rectangles, pass x and y")
            out = out | self.in_porch(s, m, x, y)
        return out

    def entry_holds(self, ev: np.ndarray) -> np.ndarray:
        """bool [rows, n]: each scope's entry condition over ``ev`` = bool [rows, NUM_EVENTS]."""
        return ~(self.entry_req[None, :, :] & ~np.asarray(ev, dtype=bool)[:, None, :]).any(axis=2)

    def clear_holds(self, ev: np.ndarray) -> np.ndarray:
        """bool [rows, n]: each scope's clear condition over ``ev`` = bool [rows, NUM_EVENTS]."""
        return ~(self.clear_req[None, :, :] & ~np.asarray(ev, dtype=bool)[:, None, :]).any(axis=2)

    @staticmethod
    def _pair_holds(req: np.ndarray, ev: np.ndarray, scope: np.ndarray) -> np.ndarray:
        """bool [k]: the conjunction of scope ``scope[j]`` holds in row ``ev[j]``."""
        return ~(req[scope] & ~ev).any(axis=1)

    # ---- per-env status -----------------------------------------------------------------------------------

    def new_status(self, n: int) -> np.ndarray:
        """int8 [n, n_scopes] of INACTIVE (a new bedroom life)."""
        return np.zeros((int(n), self.n), dtype=np.int8)

    def step(self, status: np.ndarray, map_idx: np.ndarray, ev: np.ndarray, x: Optional[np.ndarray] = None,
             y: Optional[np.ndarray] = None, ingame: Optional[np.ndarray] = None, inplace: bool = False):
        """One step of the scope rules (module docstring) for ``n`` rows: ``status`` int8 [n, n_scopes] at the
        step's start, ``map_idx`` int [n] dense map index of the step's record, ``ev`` bool [n, NUM_EVENTS] the
        events of this step's record (``events.observe(obs)``; NOT the latched ``life_events``, which a rewind does
        not roll back: module docstring), ``x`` / ``y`` the record's tile (needed iff a scope has a porch),
        ``ingame`` bool [n] (None: all in game). Returns ``(status, activated, cleared, exited)``: the new
        status (a copy unless ``inplace``), int64 [n] the scope activated / cleared this step (-1 none) and bool [n]
        the row is outside its ACTIVE scope's area."""
        st = np.asarray(status)
        if st.dtype != np.int8 or st.ndim != 2 or st.shape[1] != self.n:
            raise ValueError(f"status {st.dtype} {st.shape}, expected int8 [n, {self.n}]")
        if not inplace:
            st = st.copy()
        n = st.shape[0]
        map_idx = np.asarray(map_idx, dtype=np.int64)
        ev = np.asarray(ev, dtype=bool)
        if map_idx.shape != (n,) or ev.shape != (n, E.NUM_EVENTS):
            raise ValueError(f"map_idx {map_idx.shape} / ev {ev.shape} for {n} status rows")
        activated = np.full(n, -1, dtype=np.int64)
        cleared = np.full(n, -1, dtype=np.int64)
        exited = np.zeros(n, dtype=bool)
        live = np.ones(n, dtype=bool) if ingame is None else np.asarray(ingame, dtype=bool)
        act = active_scope(st)
        rows = np.flatnonzero((act >= 0) & live)
        if rows.size:
            a = act[rows]
            done = self._pair_holds(self.clear_req, ev[rows], a)
            if done.any():
                cr, ca = rows[done], a[done]
                st[cr, ca] = CLEARED
                cleared[cr] = ca
            keep = ~done
            if keep.any() and self.confine:
                rr, ra = rows[keep], a[keep]
                if self.has_porch:
                    if x is None or y is None:
                        raise ValueError("ScopeSet.step: the scopes have porch rectangles, pass x and y")
                    ins = self.inside(ra, map_idx[rr], np.asarray(x)[rr], np.asarray(y)[rr])
                else:
                    ins = self.area_table()[ra, map_idx[rr]]
                ins = ins | self.detour_table()[ra, map_idx[rr]]  # (a detour is not an exit: docstring "Detour")
                exited[rr[~ins]] = True
        free = np.flatnonzero((act < 0) & live)
        if free.size:
            m = np.clip(map_idx[free], 0, MAP_COUNT)
            may = self.may_activate[m]
            free, m = free[may], m[may]
            if free.size:
                ok = self.act_table[:, m].T & (st[free] == INACTIVE) & self.entry_holds(ev[free])  # [k, n]
                if self.ordered and ok.any():  # every scope before (ORDER) is CLEARED
                    ok &= ((st[free][:, None, :] == CLEARED) | ~self.before[None, :, :]).all(axis=2)
                hit = ok.any(axis=1)
                if hit.any():
                    f, okh = free[hit], ok[hit]
                    fs = np.where(okh, self.rank[None, :], self.n).argmin(axis=1)  # the earliest in ORDER wins
                    st[f, fs] = ACTIVE
                    activated[f] = fs
        return st, activated, cleared, exited

    # ---- validation and description ---------------------------------------------------------------------

    def validate(self) -> None:
        """ValueError listing every problem (module docstring)."""
        bad = list(self._problems)
        maps = self.data.maps
        seen_names = set()
        for s in self.scopes:
            if not s.name or s.name in seen_names:
                bad.append(f"scope name {s.name!r} empty or duplicate")
            seen_names.add(s.name)
            for what, names in (("home", (s.home,)), ("anchor", s.anchors), ("include", s.include),
                                ("exclude", s.exclude), ("porch map", tuple(s.porch)), ("detour anchor", s.detour)):
                for m in names:
                    if m not in maps or m not in MAP_DENSE:
                        bad.append(f"{s.name}: unknown {what} map {m!r}")
            if not s.anchors:
                bad.append(f"{s.name}: no anchors")
            if set(s.include) & set(s.exclude):
                bad.append(f"{s.name}: maps both included and excluded: {sorted(set(s.include) & set(s.exclude))}")
            for m, r in s.porch.items():
                if len(r) != 4 or any(int(v) < 0 for v in r) or r[0] > r[2] or r[1] > r[3]:
                    bad.append(f"{s.name}: porch {m}: expected (x0, y0, x1, y1) with 0 <= x0 <= x1, 0 <= y0 <= y1,"
                               f" got {tuple(r)}")
            if not s.clear:
                bad.append(f"{s.name}: empty clear condition (the scope would clear right after activating)")
            if s.activate not in ACTIVATE_MODES:
                bad.append(f"{s.name}: activate {s.activate!r} is not one of {ACTIVATE_MODES}")
            if s.final and "champion" not in s.clear:
                bad.append(f"{s.name}: a final scope's clear must contain 'champion' (the life ends on that step; "
                           f"a later capture would store the final scope CLEARED)")
            for which in ("entry", "clear"):
                hist = [nm for nm in getattr(s, which) if nm in EVENT_COL and E.EVENTS[EVENT_COL[nm]].kind == "map"]
                if hist:
                    bad.append(f"{s.name}: {which} uses map event(s) {', '.join(hist)}: a scope expresses location "
                               f"by its home and area; use flag / var / item / move events (game state)")
        homes = [s.home for s in self.scopes]
        for h in sorted(set(homes)):
            if homes.count(h) > 1:
                bad.append(f"home {h} shared by {', '.join(s.name for s in self.scopes if s.home == h)}")
        for i, (s, area) in enumerate(zip(self.scopes, self.areas)):
            if not area:
                continue
            if s.home not in area:
                bad.append(f"{s.name}: home {s.home} is not inside its own area")
            for j, o in enumerate(self.scopes):
                if j != i and o.home in self.detours[i]:
                    bad.append(f"{s.name}: {o.name}'s home {o.home} lies in its detour (a detour never holds a home)")
            if not any(CENTER_TAG in m for m in area):
                bad.append(f"{s.name}: no Pokémon Center map (name with {CENTER_TAG}) in its area")
            if not any(m in self.data.land for m in area):
                bad.append(f"{s.name}: no map with land encounters in its area")
            for j, o in enumerate(self.scopes):
                if j != i and (o.home in area or o.home in s.porch) and not set(o.clear) <= set(s.entry):
                    bad.append(f"{o.name}'s home {o.home} lies inside {s.name}'s area (an active scope's "
                               f"area never contains another scope's home, unless that scope's clear is part of "
                               f"this one's entry)")
        if bad:
            raise ValueError("relay scopes invalid:\n  - " + "\n  - ".join(bad))

    def describe(self) -> str:
        """Every scope's home, conditions, anchors and resolved area (map names without ``MAP_``)."""
        def short(m):
            return m[4:] if m.startswith("MAP_") else m
        lines = [f"relay scopes: {self.n}, map data {self.data.root} ({len(self.data.maps)} maps)"
                 + (f"; ordered activation: {' -> '.join(self.order)}" if self.ordered else "")
                 + ("" if self.confine else "; unconfined (no area exits)")]
        for i, (s, area) in enumerate(zip(self.scopes, self.areas)):
            centers = [short(m) for m in area if CENTER_TAG in m]
            land = [short(m) for m in area if m in self.data.land]
            lines.append(f"{i} {s.name}: home {short(s.home)}"
                         f"{' (activates anywhere in its area)' if s.activate == 'area' else ''}, "
                         f"entry {' & '.join(s.entry) or '-'}, clear "
                         f"{' & '.join(s.clear)}{' (final)' if s.final else ''}, anchors "
                         f"{', '.join(map(short, s.anchors))}: {len(area)} maps")
            lines.append(f"    centers: {', '.join(centers)}")
            lines.append(f"    land encounters: {', '.join(land)}")
            if self.detours[i]:
                lines.append(f"    detour (no exit, no mask): {len(self.detours[i])} maps from "
                             f"{', '.join(map(short, s.detour))}")
            if s.porch:
                lines.append(f"    porch: {', '.join(f'{short(m)} {tuple(r)}' for m, r in s.porch.items())}")
            for k in range(0, len(area), 6):
                lines.append("    " + " ".join(short(m) for m in area[k:k + 6]))
        return "\n".join(lines)


@functools.lru_cache(maxsize=4)
def scope_set(ordered: bool = False, confine: bool = True) -> ScopeSet:
    """The validated :data:`SCOPES` over the repository's map data (built once per rule set; ``ordered`` /
    ``confine``: module docstring "Ordered activation and unconfined scopes")."""
    return ScopeSet(SCOPES, ordered=ordered, confine=confine)


def describe() -> str:
    return scope_set().describe()


__all__ = ["INACTIVE", "ACTIVE", "CLEARED", "STATUS_NAMES", "Scope", "SCOPES", "SCOPE_NAMES", "ORDER",
           "Predicate", "resolve_predicate", "holds", "active_scope", "MapInfo", "MapData", "map_data", "closure",
           "area_maps", "ScopeSet", "scope_set", "describe", "MAP_DENSE", "MAP_NAMES"]


if __name__ == "__main__":
    print(describe())
