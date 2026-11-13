"""The table of ``--relay-exit mask`` (rl/pokerl/relay/exits.py), over the repository's map
data. The env side is test_relay_env.py ``test_exit_mask_*``."""
import numpy as np
import pytest

from pokerl.mask import ACTIONS
from pokerl.relay import exits as X
from pokerl.relay import scopes as SC
from pokerl.relay.scopes import MAP_DENSE, SCOPE_NAMES

SCOPES = SC.scope_set()
ROOT = SCOPES.data.root


def presses(name):
    return X.exit_table(SCOPES).presses[SCOPE_NAMES.index(name)]


def test_arrow_warp_only_on_its_tile():
    t = presses("vermilion")  # the user's example: DOWN on the mat of Diglett's Cave north entrance -> Route 2
    assert t[("MAP_DIGLETTS_CAVE_NORTH_ENTRANCE", 4, 6)] == X.SOUTH
    assert sum(k[0] == "MAP_DIGLETTS_CAVE_NORTH_ENTRANCE" for k in t) == 1  # (its ladder stays in the area)


def test_step_warp_from_its_neighbours():
    t = presses("vermilion")  # the gate's cave door (4, 1) to Saffron City: the press that steps onto it
    assert t[("MAP_ROUTE6_NORTH_ENTRANCE", 4, 2)] == X.NORTH
    assert t[("MAP_ROUTE6_NORTH_ENTRANCE", 3, 1)] == X.EAST and t[("MAP_ROUTE6_NORTH_ENTRANCE", 5, 1)] == X.WEST
    assert ("MAP_ROUTE6_NORTH_ENTRANCE", 4, 1) not in t  # (standing on it, nothing fires)


def test_warp_event_without_warp_behaviour_is_not_listed():
    t = presses("vermilion")  # (5, 8) and (7, 8) beside the mat (6, 8) hold warp events on MB_NORMAL tiles: they never fire
    assert t[("MAP_UNDERGROUND_PATH_NORTH_ENTRANCE", 6, 8)] == X.SOUTH
    assert ("MAP_UNDERGROUND_PATH_NORTH_ENTRANCE", 5, 8) not in t
    assert ("MAP_UNDERGROUND_PATH_NORTH_ENTRANCE", 7, 8) not in t


def test_connection_out_of_the_area_only():
    t = presses("vermilion")
    route11 = X.map_tiles("MAP_ROUTE11", ROOT)
    assert t[("MAP_ROUTE11", route11.width - 1, 10)] == X.EAST  # -> Route 12
    assert ("MAP_ROUTE11", route11.width - 2, 10) not in t
    assert not any(k[0] == "MAP_ROUTE11" and b & X.WEST for k, b in t.items())  # Route 11 -> Vermilion stays inside
    vermilion = X.map_tiles("MAP_VERMILION_CITY", ROOT)
    assert any(c[2] == "MAP_ROUTE6" for c in vermilion.connections)
    assert not any(k[0] == "MAP_VERMILION_CITY" for k in t)  # (every connection of the city is in the area)


def test_every_entry_on_a_map_of_its_area():
    for name, area, t in zip(SCOPES.names, SCOPES.areas, X.exit_table(SCOPES).presses):
        assert t, name
        assert {k[0] for k in t} <= set(area), name
        for (m, x, y), b in t.items():
            tiles = X.map_tiles(m, ROOT)
            assert 0 <= x < tiles.width and 0 <= y < tiles.height and 0 < b < 16, (name, m, x, y)


@pytest.mark.parametrize("name,area", list(zip(SCOPES.names, SCOPES.areas)))
def test_nothing_carries_a_player_out(name, area):
    """No ledge, current, spinner or ice tile moves a player through an exit the table lists, and no exit is a step
    warp reached from another map of the area: the table's premise (a walk out is a press from a listed tile)."""
    assert X.uncovered(area, ROOT) == []


def test_bits_lookup():
    table = X.exit_table(SCOPES)
    v, c = SCOPE_NAMES.index("vermilion"), SCOPE_NAMES.index("celadon")
    d = MAP_DENSE["MAP_DIGLETTS_CAVE_NORTH_ENTRANCE"]
    bits = table.bits(np.array([v, v, c, v, v]), np.array([d, d, d, d, 0]), np.array([4, 4, 4, -1, 4]),
                      np.array([6, 5, 6, 6, 6]))
    assert bits.tolist() == [X.SOUTH, 0, 0, 0, 0]  # the tile, another tile, another scope, off the map, unknown map
    assert bits.dtype == np.uint8


def test_allowed_table():
    assert X.ALLOWED.shape == (16, len(ACTIONS)) and X.ALLOWED[0].all()
    for bit, name in X.DIR_NAMES.items():
        off = [ACTIONS[i] for i in np.flatnonzero(~X.ALLOWED[bit])]
        assert off == [name]
    assert not X.ALLOWED[15, [ACTIONS.index(a) for a in ("UP", "DOWN", "LEFT", "RIGHT")]].any()
    assert X.ALLOWED[15, [ACTIONS.index(a) for a in ("NOOP", "A", "B", "START")]].all()
