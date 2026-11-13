"""Relay scopes (rl/pokerl/relay/scopes.py): the six scopes' warp-closure areas,
load-time validation, predicates over the latched event matrix and the per-env status transitions."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import records, set_flag
from pokerl import events as E
from pokerl import reward as R
from pokerl import tables as T
from pokerl.relay import scopes as SC
from pokerl.relay.scopes import ACTIVE, CLEARED, INACTIVE, SCOPES, Scope, ScopeSet
from pokerl.reward import MAP_COUNT


def _map_index(obs):
    """Dense map id (1..MAP_COUNT, 0 unknown) of records."""
    return R.MAP_INDEX[obs["map_group"].astype(np.int64) * 256 + obs["map_num"]]


ALL_MAPS = [n for _, n, _, _ in T.MAPS]
V, CE, SA, FU, CI, VI = range(6)  # SCOPES order


@pytest.fixture(scope="module")
def ss() -> ScopeSet:
    return SC.scope_set()


def area_variant() -> ScopeSet:
    """The six scopes with Viridian activating anywhere in its area (``Scope.activate`` "area"; the
    six scopes activate on their homes): the area mode stays supported."""
    return ScopeSet(SCOPES[:VI] + (replace(SCOPES[VI], activate="area"),))


def area(ss, name):
    return set(ss.areas[ss.index(name)])


def prefixed(prefix):
    out = [m for m in ALL_MAPS if m.startswith(prefix)]
    assert out, prefix
    return out


def ev(n=1, names=(), rows=None):
    """bool [n, NUM_EVENTS] with the named events true (in ``rows``, default all)."""
    e = np.zeros((n, E.NUM_EVENTS), dtype=bool)
    for nm in names:
        e[slice(None) if rows is None else rows, SC.EVENT_COL[nm]] = True
    return e


def idx(*names):
    return np.array([SC.MAP_DENSE[n] for n in names], dtype=np.int64)


def status(*rows):
    """int8 status rows from {scope id: status} dicts (6 scopes)."""
    st = np.zeros((len(rows), 6), dtype=np.int8)
    for i, d in enumerate(rows):
        for s, v in d.items():
            st[i, s] = v
    return st


# ---- scopes as data ----------------------------------------------------------------------------------------

def test_six_scopes_as_spec(ss):
    assert ss.names == ("vermilion", "celadon", "saffron", "fuchsia", "cinnabar", "viridian")
    assert [s.home for s in SCOPES] == ["MAP_VERMILION_CITY", "MAP_CELADON_CITY", "MAP_SAFFRON_CITY",
                                        "MAP_FUCHSIA_CITY", "MAP_CINNABAR_ISLAND", "MAP_VIRIDIAN_CITY"]
    assert [s.entry for s in SCOPES] == [("badge2", "ss_ticket"), (), ("poke_flute", "badge5"), (), (),
                                         ("badge1", "badge2", "badge3", "badge4", "badge5", "badge6", "badge7",
                                          "bill_answered")]
    assert [s.activate for s in SCOPES] == ["home"] * 6
    assert [s.clear for s in SCOPES] == [("badge3",), ("badge4", "silph_scope", "tea"), ("badge6",),
                                         ("hm03_surf", "hm04_strength", "badge5"), ("badge7",), ("champion",)]
    assert SCOPES[SA].anchors == ("MAP_SAFFRON_CITY", "MAP_ROUTE6", "MAP_ROUTE7", "MAP_ROUTE8")
    assert SCOPES[VI].anchors == ("MAP_VIRIDIAN_CITY", "MAP_PALLET_TOWN", "MAP_ROUTE1", "MAP_ROUTE22", "MAP_ROUTE23",
                                  "MAP_ROUTE21_NORTH", "MAP_ROUTE21_SOUTH", "MAP_CINNABAR_ISLAND",
                                  "MAP_INDIGO_PLATEAU_EXTERIOR")
    assert ss.final.tolist() == [False] * 5 + [True]  # (Viridian's clear is the Champion)
    for s, p in zip(SCOPES, ss.clear):
        assert [E.EVENTS[c].name for c in p.cols] == list(s.clear)
    assert ss.home_index.tolist() == idx(*(s.home for s in SCOPES)).tolist()
    assert all(ss.home_scope[h] == i for i, h in enumerate(ss.home_index))
    assert (ss.home_scope >= 0).sum() == 6
    # where each scope may activate: the six homes
    t = ss.act_table
    assert t.shape == (6, MAP_COUNT + 1) and not t[:, 0].any()
    for i in range(6):
        assert t[i].sum() == 1 and t[i, ss.home_index[i]]
    assert (ss.may_activate == t.any(axis=0)).all() and ss.may_activate.sum() == 6
    # the area mode: Viridian's whole area, which holds Cinnabar's home
    va = area_variant()
    assert (va.act_table[VI] == va.area_table()[VI]).all() and va.act_table[VI].sum() == 45
    assert (va.act_table[:VI] == t[:VI]).all() and va.may_activate.sum() == 45 + 4


def test_closure_celadon(ss):
    a = area(ss, "celadon")
    assert len(a) == 37
    assert {"MAP_CELADON_CITY_GYM", "MAP_CELADON_CITY_GAME_CORNER", "MAP_CELADON_CITY_GAME_CORNER_PRIZE_ROOM",
            "MAP_UNDERGROUND_PATH_WEST_ENTRANCE", "MAP_ROUTE16_NORTH_ENTRANCE_1F", "MAP_ROUTE16_NORTH_ENTRANCE_2F",
            "MAP_ROUTE16_HOUSE", "MAP_CELADON_CITY_POKEMON_CENTER_1F", "MAP_ROUTE7_EAST_ENTRANCE"} <= a
    assert set(prefixed("MAP_ROCKET_HIDEOUT_")) <= a
    assert set(prefixed("MAP_CELADON_CITY_CONDOMINIUMS_")) <= a
    assert set(prefixed("MAP_CELADON_CITY_DEPARTMENT_STORE_")) <= a
    # overworld maps beyond the anchors stay out (walking onto them is an exit)
    assert not {"MAP_ROUTE8", "MAP_SAFFRON_CITY", "MAP_ROUTE17", "MAP_LAVENDER_TOWN"} & a


def test_closure_vermilion(ss):
    a = area(ss, "vermilion")
    assert set(prefixed("MAP_SSANNE_")) <= a
    assert {"MAP_VERMILION_CITY_GYM", "MAP_DIGLETTS_CAVE_SOUTH_ENTRANCE", "MAP_DIGLETTS_CAVE_B1F",
            "MAP_DIGLETTS_CAVE_NORTH_ENTRANCE", "MAP_ROUTE6", "MAP_ROUTE11", "MAP_ROUTE11_EAST_ENTRANCE_1F",
            "MAP_UNDERGROUND_PATH_SOUTH_ENTRANCE"} <= a
    assert not {"MAP_ROUTE2", "MAP_ROUTE5", "MAP_ROUTE12", "MAP_SAFFRON_CITY"} & a


def test_closure_saffron(ss):
    a = area(ss, "saffron")
    assert set(prefixed("MAP_SILPH_CO_")) <= a
    assert {"MAP_SAFFRON_CITY_GYM", "MAP_SAFFRON_CITY_CONNECTION", "MAP_SAFFRON_CITY_DOJO", "MAP_ROUTE6",
            "MAP_ROUTE7", "MAP_ROUTE8"} <= a
    assert not {"MAP_ROUTE5", "MAP_CELADON_CITY", "MAP_LAVENDER_TOWN", "MAP_VERMILION_CITY"} & a


def test_closure_fuchsia(ss):
    a = area(ss, "fuchsia")
    assert set(prefixed("MAP_SAFARI_ZONE_")) <= a  # MAP_TYPE_ROUTE without connections: crossed by the closure
    assert {"MAP_FUCHSIA_CITY_GYM", "MAP_FUCHSIA_CITY_SAFARI_ZONE_ENTRANCE", "MAP_FUCHSIA_CITY_WARDENS_HOUSE",
            "MAP_ROUTE15", "MAP_ROUTE18"} <= a
    assert not {"MAP_ROUTE14", "MAP_ROUTE17", "MAP_ROUTE19"} & a


def test_closure_cinnabar(ss):
    a = area(ss, "cinnabar")
    assert set(prefixed("MAP_POKEMON_MANSION_")) <= a
    assert {"MAP_CINNABAR_ISLAND_GYM", "MAP_CINNABAR_ISLAND_POKEMON_LAB_ENTRANCE"} <= a
    assert len(a) == 15 and not {"MAP_ROUTE20", "MAP_ROUTE21_SOUTH"} & a


def test_closure_viridian(ss):
    a = area(ss, "viridian")
    assert {"MAP_VIRIDIAN_CITY_GYM", "MAP_ROUTE22_NORTH_ENTRANCE", "MAP_PALLET_TOWN_PLAYERS_HOUSE_2F",
            "MAP_PALLET_TOWN_PROFESSOR_OAKS_LAB", "MAP_ROUTE1", "MAP_ROUTE22", "MAP_ROUTE23"} <= a
    assert set(prefixed("MAP_VICTORY_ROAD_")) <= a  # through Route 23
    # south to Cinnabar Island (all of Cinnabar's area), north to the Pokemon League
    assert {"MAP_ROUTE21_NORTH", "MAP_ROUTE21_SOUTH"} <= a and area(ss, "cinnabar") <= a
    assert {"MAP_INDIGO_PLATEAU_EXTERIOR", "MAP_INDIGO_PLATEAU_POKEMON_CENTER_1F"} <= a
    assert set(prefixed("MAP_POKEMON_LEAGUE_")) <= a  # the Elite Four rooms, the Champion's room, the Hall of Fame
    assert len(a) == 45
    assert not {"MAP_ROUTE2", "MAP_VIRIDIAN_FOREST", "MAP_ROUTE20", "MAP_ONE_ISLAND"} & a


def test_closure_rule(ss):
    """Only anchors (and helper maps) have connections; every other map of an area is reached through warps."""
    data = ss.data
    for s, a in zip(SCOPES, ss.areas):
        helpers = {h for an in s.anchors for h in SC.HELPER_MAPS.get(an, ())}
        assert {m for m in a if data.maps[m].connected} == set(s.anchors) | helpers, s.name
        assert list(a) == sorted(a)
    # deterministic: a second build from fresh map data gives the same areas and tables
    SC.map_data.cache_clear()
    try:
        again = ScopeSet(SCOPES)
    finally:
        SC.map_data.cache_clear()
    assert again.areas == ss.areas
    assert np.array_equal(again.area_table(), ss.area_table())


def test_area_table(ss):
    t = ss.area_table()
    assert t.shape == (6, MAP_COUNT + 1) and t.dtype == bool
    assert not t[:, 0].any()
    for i, a in enumerate(ss.areas):
        assert t[i].sum() == len(a)
        assert all(t[i, SC.MAP_DENSE[m]] for m in a)
    assert all(t[i, h] for i, h in enumerate(ss.home_index))
    x0, y0, x1, y1 = ss.porch_table()
    assert x0.shape == (6, MAP_COUNT + 1) and (x0 == -1).all() and (y1 == -1).all()


def test_describe(ss):
    d = SC.describe()
    for s, a in zip(SCOPES, ss.areas):
        assert f"{s.name}: home {s.home[4:]}" in d and f"{len(a)} maps" in d


# ---- validation --------------------------------------------------------------------------------------------

def test_validation_no_center():
    with pytest.raises(ValueError, match="no Pokémon Center"):
        ScopeSet([Scope("route1", home="MAP_ROUTE1", anchors=("MAP_ROUTE1",), clear=("badge1",))])


def test_validation_no_land_encounters():
    with pytest.raises(ValueError, match="land encounters"):
        ScopeSet([Scope("pewter", home="MAP_PEWTER_CITY", anchors=("MAP_PEWTER_CITY",), clear=("badge1",))])


def test_validation_home_inside_other_area():
    # Saffron with Route 6 (in Vermilion's area) as its home
    with pytest.raises(ValueError, match="saffron's home MAP_ROUTE6 lies inside vermilion's area"):
        ScopeSet([SCOPES[V], replace(SCOPES[SA], home="MAP_ROUTE6")])
    # a porch map counts too
    with pytest.raises(ValueError, match="lies inside vermilion's area"):
        ScopeSet([replace(SCOPES[V], porch={"MAP_CELADON_CITY": (0, 0, 3, 3)}), SCOPES[CE]])


def test_validation_home_inside_cleared_scope_area():
    """A home may lie in another scope's area when its scope's clear is part of that scope's entry (Cinnabar inside
    Viridian: badge 7 clears the one and opens the other); not otherwise."""
    ScopeSet([SCOPES[CI], SCOPES[VI]])
    with pytest.raises(ValueError, match="cinnabar's home MAP_CINNABAR_ISLAND lies inside viridian's area"):
        ScopeSet([SCOPES[CI], replace(SCOPES[VI], entry=("badge6",))])


def test_validation_final_needs_champion():
    with pytest.raises(ValueError, match="final scope's clear must contain 'champion'"):
        ScopeSet([replace(SCOPES[VI], clear=("badge8",))])
    assert ScopeSet([replace(SCOPES[V], final=False)]).final.tolist() == [False]


def test_viridian_clears_at_the_champion(ss):
    """Viridian (badge 7 entry) is confined through Cinnabar, Route 21, Victory Road and the League; badge 8 no longer
    clears it, the Champion does; the cleared Cinnabar inside it never re-activates."""
    st = status({CI: CLEARED, VI: ACTIVE})
    got = ["badge7", "badge8", "bill_answered"]
    for m in ("MAP_CINNABAR_ISLAND", "MAP_ROUTE21_SOUTH", "MAP_VIRIDIAN_CITY_GYM", "MAP_VICTORY_ROAD_3F",
              "MAP_INDIGO_PLATEAU_POKEMON_CENTER_1F", "MAP_POKEMON_LEAGUE_LANCES_ROOM"):
        st, a, c, x = ss.step(st, idx(m), ev(1, got))
        assert (a[0], c[0], x[0]) == (-1, -1, False), m
    assert st[0, VI] == ACTIVE and st[0, CI] == CLEARED
    st, a, c, x = ss.step(st, idx("MAP_ROUTE20"), ev(1, got))
    assert x[0] and st[0, VI] == ACTIVE
    st, a, c, x = ss.step(st, idx("MAP_POKEMON_LEAGUE_CHAMPIONS_ROOM"), ev(1, got + ["champion"]))
    assert (a[0], c[0], x[0]) == (-1, VI, False) and st[0, VI] == CLEARED


def test_validation_misc():
    with pytest.raises(ValueError, match="not inside its own area"):
        ScopeSet([replace(SCOPES[CI], home="MAP_ROUTE21_NORTH")])
    with pytest.raises(ValueError, match="unknown event name"):
        ScopeSet([replace(SCOPES[V], clear=("badge3", "no_such_event"))])
    with pytest.raises(ValueError, match="unknown anchor map"):
        ScopeSet([replace(SCOPES[V], anchors=("MAP_VERMILION_CITY", "MAP_NOWHERE"))])
    with pytest.raises(ValueError, match="shared by"):
        ScopeSet([SCOPES[V], replace(SCOPES[V], name="vermilion2")])
    with pytest.raises(ValueError, match="uses map event"):
        ScopeSet([replace(SCOPES[V], entry=("vermilion_city",))])
    with pytest.raises(ValueError, match="empty clear condition"):
        ScopeSet([replace(SCOPES[V], clear=())])
    with pytest.raises(ValueError, match="porch MAP_ROUTE5"):
        ScopeSet([replace(SCOPES[V], porch={"MAP_ROUTE5": (5, 0, 3, 3)})])
    with pytest.raises(ValueError, match="both included and excluded"):
        ScopeSet([replace(SCOPES[V], include=("MAP_ROUTE5",), exclude=("MAP_ROUTE5",))])
    # every problem is listed in one error
    with pytest.raises(ValueError) as ei:
        ScopeSet([Scope("route1", home="MAP_ROUTE1", anchors=("MAP_ROUTE1",), clear=("nope",))])
    msg = str(ei.value)
    assert "no Pokémon Center" in msg and "unknown event name" in msg


def test_include_exclude():
    s = ScopeSet([replace(SCOPES[CE], include=("MAP_ROUTE8",), exclude=("MAP_TRADE_CENTER", "MAP_UNION_ROOM"))])
    a = set(s.areas[0])
    assert "MAP_ROUTE8" in a and not {"MAP_TRADE_CENTER", "MAP_UNION_ROOM"} & a and len(a) == 36


# ---- predicates --------------------------------------------------------------------------------------------

def test_holds():
    e = ev(3, ["badge4", "tea"], rows=[0, 1])
    e[0, SC.EVENT_COL["silph_scope"]] = True
    assert SC.holds((), e).tolist() == [True, True, True]
    assert SC.holds(("badge4", "silph_scope", "tea"), e).tolist() == [True, False, False]
    p = SC.resolve_predicate(("badge4", "tea"))
    assert SC.holds(p, e).tolist() == [True, True, False]
    assert SC.holds(p.cols, e).tolist() == [True, True, False]
    assert SC.holds(p.cols.tolist(), e).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="unknown event name"):
        SC.resolve_predicate(("badge9",))
    with pytest.raises(ValueError, match="event matrix"):
        SC.holds(p, np.zeros((3, 4), dtype=bool))


def test_holds_matrices(ss):
    rng = np.random.default_rng(0)
    e = rng.random((64, E.NUM_EVENTS)) < 0.7
    eh, ch = ss.entry_holds(e), ss.clear_holds(e)
    for i in range(6):
        assert np.array_equal(eh[:, i], SC.holds(ss.entry[i], e))
        assert np.array_equal(ch[:, i], SC.holds(ss.clear[i], e))


def test_active_scope():
    st = status({}, {V: CLEARED, CE: ACTIVE}, {FU: ACTIVE}, {V: CLEARED})
    assert SC.active_scope(st).tolist() == [-1, CE, FU, -1]


# ---- status transitions ------------------------------------------------------------------------------------

def test_activation_needs_home_and_entry(ss):
    st = ss.new_status(4)
    m = idx("MAP_SAFFRON_CITY", "MAP_SAFFRON_CITY", "MAP_ROUTE6", "MAP_VERMILION_CITY")
    e = ev(4, ["poke_flute", "badge5"], rows=[1, 2]) | ev(4, ["badge2", "ss_ticket"], rows=[3])
    new, act, clr, ex = ss.step(st, m, e)
    assert act.tolist() == [-1, SA, -1, V]
    assert clr.tolist() == [-1] * 4 and not ex.any()
    assert new[1].tolist() == [0, 0, ACTIVE, 0, 0, 0] and new[3].tolist() == [ACTIVE, 0, 0, 0, 0, 0]
    assert not new[[0, 2]].any() and not st.any()  # the input is not modified
    # evaluated every step: the flute arriving while standing on the home map activates
    new2, act2, _, _ = ss.step(new, m, ev(4, ["poke_flute", "badge5"]))
    assert act2.tolist() == [SA, -1, -1, -1]
    # Saffron needs the Poke Flute and badge 5
    e = ev(3, ["poke_flute"], rows=[0]) | ev(3, ["badge5"], rows=[1]) | ev(3, ["poke_flute", "badge5"], rows=[2])
    _, a, _, _ = ss.step(ss.new_status(3), idx(*["MAP_SAFFRON_CITY"] * 3), e)
    assert a.tolist() == [-1, -1, SA]
    # Vermilion needs badge 2 (Cut needs it for the gym tree) and the S.S. Ticket (HM01 is on the ship)
    e = ev(3, ["badge2"], rows=[1]) | ev(3, ["badge2", "ss_ticket"], rows=[2])
    _, a, _, _ = ss.step(ss.new_status(3), idx(*["MAP_VERMILION_CITY"] * 3), e)
    assert a.tolist() == [-1, -1, V]
    # Viridian needs all seven badges 1-7: a life that skipped Sabrina (or any other gym) and holds Blaine's badge
    # stays free; and Bill's offer answered
    seven = ["badge%d" % i for i in range(1, 8)] + ["bill_answered"]
    e = ev(5, ["badge7", "bill_answered"], rows=[0]) | ev(5, [b for b in seven if b != "badge6"], rows=[1]) | \
        ev(5, [b for b in seven if b != "badge3"], rows=[2]) | ev(5, seven, rows=[3]) | \
        ev(5, [b for b in seven if b != "bill_answered"], rows=[4])
    _, a, _, _ = ss.step(ss.new_status(5), idx(*["MAP_VIRIDIAN_CITY"] * 5), e)
    assert a.tolist() == [-1, -1, -1, VI, -1]


SEVEN = tuple("badge%d" % i for i in range(1, 8))


def test_viridian_activates_on_viridian_city(ss):
    """Viridian activates on Viridian City, its home, alone; a seven-badge life
    anywhere else in its area (Cinnabar's pier, Route 21, Pallet, Route 22, the Plateau) stays free until it gets
    there, so the YES to Bill on the pier sails a free life."""
    maps = ["MAP_CINNABAR_ISLAND", "MAP_CINNABAR_ISLAND_POKEMON_CENTER_1F", "MAP_ROUTE21_SOUTH", "MAP_PALLET_TOWN",
            "MAP_ROUTE1", "MAP_ROUTE22", "MAP_VICTORY_ROAD_1F", "MAP_INDIGO_PLATEAU_EXTERIOR", "MAP_POKEMON_MANSION_1F",
            "MAP_VIRIDIAN_CITY_POKEMON_CENTER_1F", "MAP_ONE_ISLAND", "MAP_VIRIDIAN_CITY"]
    n = len(maps)
    st = ss.new_status(n)
    st[:, CI] = CLEARED
    new, act, clr, ex = ss.step(st, idx(*maps), ev(n, SEVEN + ("bill_answered",)))
    assert act.tolist() == [-1] * (n - 1) + [VI] and not ex.any() and (clr == -1).all()
    assert new[-1, VI] == ACTIVE and not new[:-1, VI].any()
    # ordered rules: on Viridian City once every earlier scope is cleared
    o = SC.scope_set(ordered=True)
    st = o.new_status(2)
    st[0, :5] = CLEARED
    st[1, [0, 1, 2, 4]] = CLEARED  # Fuchsia not cleared
    _, act, _, _ = o.step(st, idx("MAP_VIRIDIAN_CITY", "MAP_VIRIDIAN_CITY"), ev(2, SEVEN + ("bill_answered",)))
    assert act.tolist() == [VI, -1]


def test_area_activation_mode():
    """With ``activate`` "area" Viridian activates on any map of its area, not on Viridian City alone, once the seven
    badges are held and Bill's One Island offer was answered (the var event ``bill_answered``)."""
    ss = area_variant()
    maps = ["MAP_CINNABAR_ISLAND", "MAP_CINNABAR_ISLAND_POKEMON_CENTER_1F", "MAP_ROUTE21_SOUTH", "MAP_PALLET_TOWN",
            "MAP_ROUTE1", "MAP_VIRIDIAN_CITY", "MAP_ROUTE22", "MAP_VICTORY_ROAD_1F", "MAP_INDIGO_PLATEAU_EXTERIOR",
            "MAP_POKEMON_MANSION_1F"]
    out = ["MAP_ROUTE20", "MAP_ONE_ISLAND", "MAP_ONE_ISLAND_HARBOR", "MAP_VERMILION_CITY", "MAP_ROUTE2",
           "MAP_SAFFRON_CITY"]
    n = len(maps) + len(out)
    st = ss.new_status(n)
    st[:, CI] = CLEARED  # (badge 7 means Cinnabar was cleared in its gym)
    new, act, clr, ex = ss.step(st, idx(*maps, *out), ev(n, SEVEN + ("bill_answered",)))
    assert act.tolist() == [VI] * len(maps) + [-1] * len(out)
    assert (new[:len(maps), VI] == ACTIVE).all() and not new[len(maps):, VI].any() and not ex.any()
    # Blaine beaten, Bill's YES/NO not yet answered (scene var 1): nothing, anywhere in the area
    _, act, _, _ = ss.step(st[:len(maps)], idx(*maps), ev(len(maps), SEVEN))
    assert act.tolist() == [-1] * len(maps)
    # answered (NO, Bill waits in the Center; or YES and back with Bill): activates on the pier that very step
    new, act, _, _ = ss.step(st[:1], idx("MAP_CINNABAR_ISLAND"), ev(1, SEVEN + ("bill_answered",)))
    assert act.tolist() == [VI] and new[0, VI] == ACTIVE
    # a YES sailed the life to One Island (scene var 2): a free life, no activation out there
    _, act, _, ex = ss.step(st[:1], idx("MAP_ONE_ISLAND"), ev(1, SEVEN + ("bill_answered",)))
    assert act.tolist() == [-1] and not ex.any()
    # the badge-6 skipper: badge 7 first, Sabrina later, then the first area map it reaches (Viridian City from
    # Route 2, Pallet by Fly, Cinnabar by Surf) confines it
    six = tuple(b for b in SEVEN if b != "badge6") + ("bill_answered",)
    _, act, _, _ = ss.step(st[:3], idx("MAP_CINNABAR_ISLAND", "MAP_ROUTE21_NORTH", "MAP_SAFFRON_CITY"), ev(3, six))
    assert act.tolist() == [-1, -1, -1]  # (Saffron's home, but its entry wants the Poke Flute too)
    e = ev(3, six + ("poke_flute",))
    _, act, _, _ = ss.step(st[:3], idx("MAP_CINNABAR_ISLAND", "MAP_ROUTE21_NORTH", "MAP_SAFFRON_CITY"), e)
    assert act.tolist() == [-1, -1, SA]
    e = ev(3, six)
    e[:, SC.EVENT_COL["badge6"]] = True
    _, act, _, _ = ss.step(st[:3], idx("MAP_VIRIDIAN_CITY", "MAP_PALLET_TOWN", "MAP_CINNABAR_ISLAND"), e)
    assert act.tolist() == [VI, VI, VI]


def test_viridian_detour_is_not_an_exit(ss):
    """The game sets the Cinnabar scene var before Bill's boat sails, so a YES to Bill
    activates Viridian on the pier; the Sevii Islands 1-3 are Viridian's detour: a confined life out there is not
    exited (its scope stays ACTIVE, no activation out there either), and it is inside again on landing."""
    d = set(ss.detours[VI])
    assert len(d) == 44 and ss.detours[:VI] == ((),) * VI
    assert {"MAP_ONE_ISLAND", "MAP_ONE_ISLAND_HARBOR", "MAP_ONE_ISLAND_KINDLE_ROAD", "MAP_MT_EMBER_EXTERIOR",
            "MAP_MT_EMBER_SUMMIT", "MAP_ONE_ISLAND_POKEMON_CENTER_1F", "MAP_ONE_ISLAND_TREASURE_BEACH",
            "MAP_TWO_ISLAND", "MAP_TWO_ISLAND_CAPE_BRINK", "MAP_THREE_ISLAND", "MAP_THREE_ISLAND_BOND_BRIDGE",
            "MAP_THREE_ISLAND_BERRY_FOREST", "MAP_THREE_ISLAND_PORT", "MAP_THREE_ISLAND_HARBOR"} <= d
    assert not d & set(ss.areas[VI])  # (the Trade Center and the Union Room behind the Centers' 2F stay area maps)
    assert not {"MAP_FOUR_ISLAND", "MAP_CINNABAR_ISLAND", "MAP_ROUTE20", "MAP_VERMILION_CITY"} & d
    assert all(any(k in m for k in ("ONE_ISLAND", "TWO_ISLAND", "THREE_ISLAND", "MT_EMBER")) for m in d)
    t = ss.detour_table()
    assert t.shape == (6, MAP_COUNT + 1) and t[VI].sum() == 44 and not t[:VI].any() and not t[:, 0].any()
    # a Viridian-active life: on the pier (inside), sailing / on the islands (detour: no exit), on Route 20 (exit)
    maps = ["MAP_CINNABAR_ISLAND", "MAP_ONE_ISLAND_HARBOR", "MAP_ONE_ISLAND", "MAP_MT_EMBER_SUMMIT",
            "MAP_THREE_ISLAND_BERRY_FOREST", "MAP_ONE_ISLAND_POKEMON_CENTER_1F", "MAP_ROUTE20", "MAP_FOUR_ISLAND"]
    st = status(*[{CI: CLEARED, VI: ACTIVE}] * len(maps))
    new, act, clr, ex = ss.step(st, idx(*maps), ev(len(maps), SEVEN + ("bill_answered",)))
    assert ex.tolist() == [False, False, False, False, False, False, True, True]
    assert (act == -1).all() and (clr == -1).all() and (new[:, VI] == ACTIVE).all()
    # the same maps with another scope active: a detour is Viridian's alone
    st = status(*[{SA: ACTIVE}] * 3)
    _, _, _, ex = ss.step(st, idx("MAP_ONE_ISLAND", "MAP_ONE_ISLAND_HARBOR", "MAP_SAFFRON_CITY"), ev(3))
    assert ex.tolist() == [True, True, False]
    # a free life on the islands does not activate Viridian there (the detour is not the area)
    st = ss.new_status(2)
    st[:, CI] = CLEARED
    _, act, _, ex = ss.step(st, idx("MAP_ONE_ISLAND", "MAP_ONE_ISLAND_HARBOR"), ev(2, SEVEN + ("bill_answered",)))
    assert act.tolist() == [-1, -1] and not ex.any()
    # landing back on Cinnabar: inside, nothing happens; a YES from the Center (Bill waiting there) sails the same way
    st = status({CI: CLEARED, VI: ACTIVE})
    _, act, clr, ex = ss.step(st, idx("MAP_CINNABAR_ISLAND"), ev(1, SEVEN + ("bill_answered",)))
    assert act.tolist() == [-1] and clr.tolist() == [-1] and not ex.any()
    assert "detour (no exit, no mask): 44 maps from ONE_ISLAND, TWO_ISLAND, THREE_ISLAND" in ss.describe()
    # closure with links: an overworld walk is followed, a plain closure stops at connected maps
    assert "MAP_ONE_ISLAND_KINDLE_ROAD" in SC.closure(["MAP_ONE_ISLAND"], links=True)
    assert "MAP_ONE_ISLAND_KINDLE_ROAD" not in SC.closure(["MAP_ONE_ISLAND"])


def test_detour_validated():
    with pytest.raises(ValueError, match="unknown detour anchor map 'MAP_NOWHERE'"):
        ScopeSet([replace(SCOPES[VI], detour=("MAP_NOWHERE",))])
    with pytest.raises(ValueError, match="cinnabar's home MAP_CINNABAR_ISLAND lies in its detour"):
        ScopeSet([SCOPES[CI], replace(SCOPES[VI], detour=("MAP_CINNABAR_ISLAND",), anchors=("MAP_VIRIDIAN_CITY",),
                                      entry=("badge7",))])
    # unconfined rule set: nothing exits anyway
    o = SC.scope_set(confine=False)
    _, _, _, ex = o.step(status({VI: ACTIVE}), idx("MAP_ROUTE20"), ev(1))
    assert not ex.any()


def test_area_activation_prefers_the_earlier_scope():
    """Cinnabar's home lies in Viridian's area: with the area mode (``area_variant``) a row that could activate both
    (Cinnabar INACTIVE with Viridian's entry held; only a contrived status) activates Cinnabar, the earlier in ORDER;
    Viridian follows a step later."""
    ss = area_variant()
    st = ss.new_status(1)  # Cinnabar INACTIVE
    e = ev(1, SEVEN + ("bill_answered",))
    new, act, _, _ = ss.step(st, idx("MAP_CINNABAR_ISLAND"), e)
    assert act.tolist() == [CI] and new[0].tolist() == [0, 0, 0, 0, ACTIVE, 0]
    # Cinnabar's clear holds at once (badge7): cleared next step, Viridian active the step after
    new2, act2, clr2, _ = ss.step(new, idx("MAP_CINNABAR_ISLAND"), e)
    assert clr2.tolist() == [CI] and act2.tolist() == [-1]
    new3, act3, _, _ = ss.step(new2, idx("MAP_CINNABAR_ISLAND"), e)
    assert act3.tolist() == [VI] and new3[0].tolist() == [0, 0, 0, 0, CLEARED, ACTIVE]
    # ordered rule set: Viridian waits for every scope before it
    o = ScopeSet(ss.scopes, ordered=True)
    st = o.new_status(2)
    st[0, :5] = CLEARED
    st[1, [0, 1, 2, 4]] = CLEARED  # Fuchsia not cleared
    _, act, _, _ = o.step(st, idx("MAP_PALLET_TOWN", "MAP_PALLET_TOWN"), ev(2, SEVEN + ("bill_answered",)))
    assert act.tolist() == [VI, -1]


def test_activate_mode_validated():
    with pytest.raises(ValueError, match="activate 'anywhere' is not one of"):
        ScopeSet([replace(SCOPES[VI], activate="anywhere")])
    # "area" for a scope with another scope's home inside: the existing rule (clear part of the entry) still decides
    with pytest.raises(ValueError, match="cinnabar's home MAP_CINNABAR_ISLAND lies inside viridian's area"):
        ScopeSet([SCOPES[CI], replace(SCOPES[VI], entry=("bill_answered",))])


def test_one_active_scope(ss):
    # walking onto Celadon City while Vermilion is active: an exit, never a second activation
    st = status({V: ACTIVE})
    new, act, clr, ex = ss.step(st, idx("MAP_CELADON_CITY"), ev(1))
    assert act.tolist() == [-1] and clr.tolist() == [-1] and ex.tolist() == [True]
    assert new.tolist() == st.tolist()
    # with a home inside the active scope's area (invalid set, validation off): still no second activation
    odd = ScopeSet([SCOPES[V], Scope("gym", home="MAP_VERMILION_CITY_GYM", anchors=("MAP_VERMILION_CITY_GYM",),
                                     clear=("badge3",))], validate=False)
    st = np.array([[ACTIVE, INACTIVE], [INACTIVE, INACTIVE]], dtype=np.int8)
    new, act, _, ex = odd.step(st, idx("MAP_VERMILION_CITY_GYM", "MAP_VERMILION_CITY_GYM"), ev(2))
    assert act.tolist() == [-1, 1] and not ex.any()
    assert ((new == ACTIVE).sum(axis=1) <= 1).all()


def test_clear_before_area(ss):
    st = status({V: ACTIVE}, {V: ACTIVE}, {V: ACTIVE})
    m = idx("MAP_ROUTE5", "MAP_VERMILION_CITY_GYM", "MAP_ROUTE5")
    e = ev(3, ["badge3"], rows=[0, 1])
    new, act, clr, ex = ss.step(st, m, e)
    assert clr.tolist() == [V, V, -1]
    assert ex.tolist() == [False, False, True]  # cleared while stepping out: a clear; not cleared: an exit
    assert new[:, V].tolist() == [CLEARED, CLEARED, ACTIVE]
    assert act.tolist() == [-1, -1, -1]


def test_exit_on_leaving(ss):
    st = status(*[{CE: ACTIVE}] * 5)
    m = idx("MAP_ROCKET_HIDEOUT_B4F", "MAP_ROUTE16", "MAP_UNDERGROUND_PATH_EAST_ENTRANCE", "MAP_ROUTE8",
            "MAP_LAVENDER_TOWN")
    _, act, clr, ex = ss.step(st, m, ev(5, ["badge4", "tea"]))  # no Silph Scope: not cleared
    assert ex.tolist() == [False, False, False, True, True]
    assert (act == -1).all() and (clr == -1).all()
    # the unknown map (dense index 0) is outside
    _, _, _, ex = ss.step(status({CE: ACTIVE}), np.array([0]), ev(1))
    assert ex.tolist() == [True]
    # free rows never exit
    _, _, _, ex = ss.step(status({V: CLEARED}), idx("MAP_ROUTE8"), ev(1))
    assert ex.tolist() == [False]


def test_porch_counts_as_inside():
    s = ScopeSet([replace(SCOPES[V], porch={"MAP_ROUTE5": (2, 3, 6, 8)}), SCOPES[CE]])
    x0, y0, x1, y1 = s.porch_table()
    r5 = SC.MAP_DENSE["MAP_ROUTE5"]
    assert (x0[0, r5], y0[0, r5], x1[0, r5], y1[0, r5]) == (2, 3, 6, 8) and x0[1, r5] == -1
    st = np.array([[ACTIVE, 0]] * 4, dtype=np.int8)
    m = idx("MAP_ROUTE5", "MAP_ROUTE5", "MAP_ROUTE5", "MAP_VERMILION_CITY")
    x, y = np.array([2, 6, 7, 40]), np.array([3, 8, 8, 40])
    _, _, _, ex = s.step(st, m, ev(4), x=x, y=y)
    assert ex.tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match="pass x and y"):
        s.step(st, m, ev(4))


def test_cleared_never_reactivates(ss):
    st = status({V: CLEARED})
    new, act, clr, ex = ss.step(st, idx("MAP_VERMILION_CITY"), ev(1, ["badge3"]))
    assert act.tolist() == [-1] and clr.tolist() == [-1] and not ex.any()
    assert new.tolist() == st.tolist()


def test_status_carried_through(ss):
    st = status({V: CLEARED, CE: ACTIVE}, {V: CLEARED, CE: CLEARED, FU: ACTIVE}, {V: CLEARED, CE: CLEARED})
    m = idx("MAP_CELADON_CITY_GYM", "MAP_SAFARI_ZONE_WEST", "MAP_SAFFRON_CITY")
    e = ev(3, ["poke_flute", "badge5"])
    new, act, clr, ex = ss.step(st, m, e)
    assert act.tolist() == [-1, -1, SA] and not ex.any() and (clr == -1).all()
    assert new[0].tolist() == st[0].tolist() and new[1].tolist() == st[1].tolist()
    assert new[2].tolist() == [CLEARED, CLEARED, ACTIVE, 0, 0, 0]
    # inplace
    st2 = st.copy()
    out, _, _, _ = ss.step(st2, m, e, inplace=True)
    assert out is st2 and np.array_equal(st2, new)
    with pytest.raises(ValueError, match="int8"):
        ss.step(st.astype(np.int64), m, e)


def test_clear_then_activate_next_step():
    """A row crosses at most one checkpoint per step: a clear on another scope's home activates it a step later."""
    odd = ScopeSet([SCOPES[V], Scope("gym", home="MAP_VERMILION_CITY_GYM", anchors=("MAP_VERMILION_CITY_GYM",),
                                     clear=("badge4",))], validate=False)
    st = np.array([[ACTIVE, INACTIVE]], dtype=np.int8)
    m = idx("MAP_VERMILION_CITY_GYM")
    st, act, clr, _ = odd.step(st, m, ev(1, ["badge3"]))
    assert clr.tolist() == [0] and act.tolist() == [-1]
    st, act, clr, _ = odd.step(st, m, ev(1, ["badge3"]))
    assert act.tolist() == [1] and clr.tolist() == [-1] and st.tolist() == [[CLEARED, ACTIVE]]


def _rec(name, flags=()):
    """One in-game record on map ``name`` with the event flags ``flags`` set."""
    key = next(k for k, n, _, _ in T.MAPS if n == name)
    r = records(1, map_=(key >> 8, key & 0xFF))
    for f in flags:
        set_flag(r, E.flag_id(f))
    return r


def test_rewound_status_does_not_reclear(ss):
    """The predicates read the CURRENT record's events (events.observe), not the latched life_events: a life that
    cleared Vermilion and rewound to a capture taken before the gym gets that capture's status back (Vermilion
    ACTIVE) and its game state (no badge 3), so it does not clear again until it wins the badge again."""
    gym, city = _rec("MAP_VERMILION_CITY_GYM", ["FLAG_BADGE03_GET"]), _rec("MAP_VERMILION_CITY")
    st = status({V: ACTIVE})
    life_events = E.observe(gym)
    st, _, clr, _ = ss.step(st, _map_index(gym), E.observe(gym))
    assert clr.tolist() == [V] and st[0, V] == CLEARED
    rewound = status({V: ACTIVE})  # the rewind capture's status row
    life_events |= E.observe(city)  # (the latched matrix keeps badge 3 across the rewind)
    st2, act, clr, ex = ss.step(rewound, _map_index(city), E.observe(city))
    assert clr.tolist() == [-1] and act.tolist() == [-1] and not ex.any() and st2[0, V] == ACTIVE
    _, _, clr_latched, _ = ss.step(rewound, _map_index(city), life_events)
    assert clr_latched.tolist() == [V]  # what the latched matrix would wrongly do
    st3, _, clr, _ = ss.step(st2, _map_index(gym), E.observe(gym))  # the badge won again
    assert clr.tolist() == [V] and st3[0, V] == CLEARED


def test_not_ingame_rows_unchanged(ss):
    st = status({V: ACTIVE}, {})
    new, act, clr, ex = ss.step(st, idx("MAP_ROUTE5", "MAP_VERMILION_CITY"), ev(2, ["badge3"]),
                                ingame=np.array([False, False]))
    assert np.array_equal(new, st) and (act == -1).all() and (clr == -1).all() and not ex.any()


def test_story_walk(ss):
    """One life through a realistic scope order: 1, 2, 4, 3, 5, 6."""
    st = ss.new_status(1)
    got = set()
    seen = []

    def go(m, *names):
        nonlocal st
        for n in names:
            got.add(n)
        st, a, c, x = ss.step(st, idx(m), ev(1, sorted(got)))
        seen.append((m[4:], int(a[0]), int(c[0]), bool(x[0])))
        return int(a[0]), int(c[0]), bool(x[0])

    assert go("MAP_PALLET_TOWN_PLAYERS_HOUSE_2F") == (-1, -1, False)
    assert go("MAP_VIRIDIAN_CITY") == (-1, -1, False)        # Viridian's entry (badges 1-7) is false
    assert go("MAP_PEWTER_CITY_GYM", "badge1") == (-1, -1, False)
    assert go("MAP_ROUTE6") == (-1, -1, False)
    assert go("MAP_VERMILION_CITY") == (-1, -1, False)       # Vermilion's entry (badge 2) is false
    assert go("MAP_ROUTE6") == (-1, -1, False)                # so the life may walk back for it
    assert go("MAP_CERULEAN_CITY_GYM", "badge2") == (-1, -1, False)
    assert go("MAP_VERMILION_CITY") == (-1, -1, False)       # no S.S. Ticket yet: free
    assert go("MAP_ROUTE25_SEA_COTTAGE", "ss_ticket") == (-1, -1, False)
    assert go("MAP_VERMILION_CITY") == (V, -1, False)
    assert go("MAP_SSANNE_CAPTAINS_OFFICE") == (-1, -1, False)
    assert go("MAP_VERMILION_CITY_GYM", "badge3") == (-1, V, False)
    assert go("MAP_ROUTE9") == (-1, -1, False)                # free between scopes
    assert go("MAP_CELADON_CITY") == (CE, -1, False)
    assert go("MAP_CELADON_CITY_GYM", "badge4") == (-1, -1, False)
    assert go("MAP_ROCKET_HIDEOUT_B4F", "silph_scope") == (-1, -1, False)
    assert go("MAP_CELADON_CITY_CONDOMINIUMS_1F", "tea") == (-1, CE, False)
    assert go("MAP_POKEMON_TOWER_7F", "poke_flute") == (-1, -1, False)
    assert go("MAP_FUCHSIA_CITY") == (FU, -1, False)
    assert go("MAP_SAFARI_ZONE_SECRET_HOUSE", "hm03_surf") == (-1, -1, False)
    assert go("MAP_FUCHSIA_CITY_GYM", "badge5") == (-1, -1, False)
    assert go("MAP_FUCHSIA_CITY_WARDENS_HOUSE", "hm04_strength") == (-1, FU, False)
    assert go("MAP_SAFFRON_CITY") == (SA, -1, False)
    assert go("MAP_SILPH_CO_11F") == (-1, -1, False)
    assert go("MAP_SAFFRON_CITY_GYM", "badge6") == (-1, SA, False)
    assert go("MAP_CINNABAR_ISLAND") == (CI, -1, False)
    assert go("MAP_CINNABAR_ISLAND_GYM", "badge7") == (-1, CI, False)
    assert go("MAP_CINNABAR_ISLAND", "bill_answered") == (-1, -1, False)  # Bill answered: free (Viridian City alone)
    assert go("MAP_ROUTE21_NORTH") == (-1, -1, False)
    assert go("MAP_PALLET_TOWN") == (-1, -1, False)
    assert go("MAP_VIRIDIAN_CITY") == (VI, -1, False)          # its home: confined from here
    assert go("MAP_ROUTE2") == (-1, -1, True)                  # out of the area: an exit (the env ends the life)
    assert go("MAP_VIRIDIAN_CITY_GYM", "badge8") == (-1, -1, False)  # (badge 8 does not clear it)
    assert go("MAP_INDIGO_PLATEAU_EXTERIOR") == (-1, -1, False)
    assert go("MAP_POKEMON_LEAGUE_LORELEIS_ROOM", "elite4_lorelei") == (-1, -1, False)
    assert go("MAP_POKEMON_LEAGUE_CHAMPIONS_ROOM", "champion") == (-1, VI, False)
    assert (st == CLEARED).all(), seen


# ---- ordered activation and unconfined scopes -----------------------------------------------------------------


def test_order_and_ranks(ss):
    assert SC.ORDER == ("vermilion", "celadon", "fuchsia", "saffron", "cinnabar", "viridian")
    assert not ss.ordered and ss.confine
    o = SC.scope_set(ordered=True, confine=False)
    assert o.ordered and not o.confine and o is SC.scope_set(ordered=True, confine=False) and o is not ss
    assert o.names == ss.names and o.rank.tolist() == [0, 1, 3, 2, 4, 5]  # (list order unchanged: checkpoint ids)
    assert o.before[SA].tolist() == [True, True, False, True, False, False]  # Saffron waits for Fuchsia
    assert o.before[FU].tolist() == [True, True, False, False, False, False]
    assert "ordered activation: vermilion -> celadon -> fuchsia" in o.describe() and "unconfined" in o.describe()
    assert "ordered" not in ss.describe().splitlines()[0]
    # a custom order; names outside it follow in list order
    c = ScopeSet([SCOPES[V], SCOPES[CE]], ordered=True, order=("celadon",), validate=False)
    assert c.order == ("celadon", "vermilion") and c.rank.tolist() == [1, 0]


def test_ordered_activation():
    o = SC.scope_set(ordered=True)
    homes = idx("MAP_CELADON_CITY", "MAP_CELADON_CITY", "MAP_FUCHSIA_CITY", "MAP_SAFFRON_CITY", "MAP_SAFFRON_CITY",
                "MAP_VERMILION_CITY")
    st = status({}, {V: CLEARED}, {V: CLEARED, CE: CLEARED}, {V: CLEARED, CE: CLEARED}, {V: CLEARED, CE: CLEARED, FU: CLEARED},
                {})
    e = ev(6, ["poke_flute", "badge5"], rows=[3, 4]) | ev(6, ["badge2", "ss_ticket"], rows=[5])
    new, act, clr, ex = o.step(st, homes, e)
    # Celadon only after Vermilion; Fuchsia after Celadon; Saffron after Fuchsia (not after Celadon alone); Vermilion first
    assert act.tolist() == [-1, CE, FU, -1, SA, V] and not ex.any() and clr.tolist() == [-1] * 6
    # the same rows under the unordered rules: Celadon, Fuchsia and Saffron activate regardless
    _, act2, _, _ = SC.scope_set().step(st, homes, e)
    assert act2.tolist() == [CE, CE, FU, SA, SA, V]
    # a CLEARED scope stays cleared and an earlier INACTIVE one blocks even with every condition true
    st = status({CE: CLEARED}, {V: CLEARED, FU: CLEARED})
    _, act3, _, _ = o.step(st, idx("MAP_FUCHSIA_CITY", "MAP_SAFFRON_CITY"), ev(2, ["poke_flute", "badge5"]))
    assert act3.tolist() == [-1, -1]


def test_unconfined_no_exit():
    u = SC.scope_set(ordered=True, confine=False)
    st = status({V: ACTIVE}, {V: ACTIVE}, {V: ACTIVE})
    m = idx("MAP_CELADON_CITY", "MAP_ROUTE12", "MAP_VERMILION_CITY_GYM")
    new, act, clr, ex = u.step(st, m, ev(3, ["badge3"], rows=[1]))
    assert not ex.any() and act.tolist() == [-1] * 3 and clr.tolist() == [-1, V, -1]
    assert new[0].tolist() == [ACTIVE, 0, 0, 0, 0, 0] and new[1, V] == CLEARED  # outside: still active / cleared
    # standing on Celadon's home with Vermilion ACTIVE: no second activation (one active scope; ordered anyway)
    _, act2, _, ex2 = u.step(status({V: ACTIVE}), idx("MAP_CELADON_CITY"), ev(1, ["badge3"]))
    assert act2.tolist() == [-1] and not ex2.any()
    # the confined set reports the exits for the same rows
    _, _, _, ex3 = SC.scope_set(ordered=True).step(st, m, ev(3))
    assert ex3.tolist() == [True, True, False]


def test_ordered_story_walk():
    """The game's order under ordered rules: each scope activates on its home once the previous is cleared."""
    o = SC.scope_set(ordered=True, confine=False)
    st = o.new_status(1)
    flags = ["badge1", "badge2", "ss_ticket"]
    walk = [("MAP_VERMILION_CITY", [], V, -1), ("MAP_VERMILION_CITY_GYM", ["badge3"], -1, V),
            ("MAP_CELADON_CITY", [], CE, -1), ("MAP_CELADON_CITY", ["badge4", "silph_scope", "tea"], -1, CE),
            ("MAP_SAFFRON_CITY", ["poke_flute", "badge5"], -1, -1),  # Saffron waits for Fuchsia
            ("MAP_FUCHSIA_CITY", [], FU, -1), ("MAP_ROUTE19", ["hm03_surf", "hm04_strength"], -1, FU),  # unconfined
            ("MAP_SAFFRON_CITY", [], SA, -1), ("MAP_SAFFRON_CITY", ["badge6"], -1, SA),
            ("MAP_CINNABAR_ISLAND", [], CI, -1), ("MAP_CINNABAR_ISLAND", ["badge7"], -1, CI),
            ("MAP_ROUTE21_NORTH", ["bill_answered"], -1, -1),  # (Viridian City alone)
            ("MAP_VIRIDIAN_CITY", [], VI, -1),
            ("MAP_INDIGO_PLATEAU_EXTERIOR", ["champion"], -1, VI)]
    for m, new_flags, want_a, want_c in walk:
        flags += new_flags
        st, a, c, ex = o.step(st, idx(m), ev(1, flags))
        assert (int(a[0]), int(c[0]), bool(ex[0])) == (want_a, want_c, False), (m, a, c, ex)
    assert st[0].tolist() == [CLEARED] * 6
