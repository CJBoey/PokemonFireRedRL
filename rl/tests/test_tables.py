"""Generated tables (rl/pokerl/gen_tables.py): sizes, ace ladder, flag weights, HM learnsets."""
import numpy as np

from pokerl import gen_tables
from pokerl import tables as T

BULBASAUR, SQUIRTLE, PIDGEY, MAGIKARP = 1, 7, 16, 129
CUT, FLY, SURF, STRENGTH, FLASH, ROCK_SMASH, WATERFALL = range(7)


def test_up_to_date():
    assert gen_tables.main(["--check"]) == 0


def test_sizes():
    assert (T.SPECIES_COUNT, T.SPECIES_EGG, T.MOVES_COUNT, T.ITEMS_COUNT, T.FLAGS_COUNT) == (440, 412, 355, 375, 2304)
    assert len(T.HM_LEARN) == 440 and len(T.FLAG_WEIGHTS) == 2304 and len(T.FLAG_CATS) == 2304
    assert len(T.MAPS) == 425
    keys = [k for k, *_ in T.MAPS]
    assert keys == sorted(keys) and len(set(keys)) == 425
    assert all(w > 0 and h > 0 for _, _, w, h in T.MAPS)
    assert T.HM_MOVES == [15, 19, 57, 70, 148, 249, 127]


def test_ace_ladder():
    """Reward v0.7: the 8 gym leaders then the Champion (no Elite Four)."""
    assert [a for _, a, _ in T.ACE_LADDER] == [14, 21, 24, 29, 43, 43, 47, 50, 63]
    assert [f for _, _, f in T.ACE_LADDER] == list(range(0x820, 0x828)) + [0x4BC]
    assert [n for n, _, _ in T.ACE_LADDER] == ["BROCK", "MISTY", "LT_SURGE", "ERIKA", "KOGA", "SABRINA",
                                               "BLAINE", "GIOVANNI", "CHAMPION"]
    assert T.FLAG_DEFEATED_CHAMP == 0x4BC and T.ACE_AFTER_CHAMPION == 63


def test_party_sizes():
    """Reward v0.5: party size K of every ladder opponent (trainer_parties.h), in ACE_LADDER order, and an
    independent count of the ``.species`` entries of each party array."""
    import os
    import re
    want = {"BROCK": 2, "MISTY": 2, "LT_SURGE": 3, "ERIKA": 3, "KOGA": 4, "SABRINA": 4, "BLAINE": 4, "GIOVANNI": 5,
            "CHAMPION": 6}
    assert dict(zip([n for n, _, _ in T.ACE_LADDER], T.ACE_PARTY_SIZES)) == want
    assert len(T.ACE_PARTY_SIZES) == len(T.ACE_LADDER) and T.PARTY_SIZE_AFTER_CHAMPION == 6
    text = open(os.path.join(gen_tables.GAME, "src", "data", "trainer_parties.h"), encoding="utf-8").read()
    arrays = {"BROCK": ["LeaderBrock"], "MISTY": ["LeaderMisty"], "LT_SURGE": ["LeaderLtSurge"],
              "ERIKA": ["LeaderErika"], "KOGA": ["LeaderKoga"], "SABRINA": ["LeaderSabrina"],
              "BLAINE": ["LeaderBlaine"], "GIOVANNI": ["LeaderGiovanni"],
              "CHAMPION": ["ChampionFirstSquirtle", "ChampionFirstBulbasaur", "ChampionFirstCharmander"]}
    for name, parts in arrays.items():
        for a in parts:
            m = re.search(r"sParty_%s\[\]\s*=\s*\{(.*?)\n\};" % a, text, re.S)
            assert m and len(re.findall(r"\.species\s*=", m.group(1))) == want[name], a


def test_flag_weights():
    """Reward v0.10: story 1, item ball of a non-key item 0.25, trainer 0.5, badge / Elite Four / Champion 20."""
    w = np.array(T.FLAG_WEIGHTS)
    assert (w[0x000:0x020] == 0).all()
    assert (w[0x3E8:0x3E8 + 191] == 0).all() and w[0x3E8 + 192] == 1
    assert w[0x3E8 + 191] == 0  # FLAG_PN_BOULDERS_SOLVED (= FLAG_UNUSED_0x4A7; the new-game preset)
    assert (w[0x500:0x800] == 0.5).all()
    assert (w[0x820:0x828] == 20).all()
    assert (w[0x4B8:0x4BD] == 20).all()
    assert w[0x020] == 1 and w[0x4B7] == 1 and w[0x4BD] == 1 and w[0x8FF] == 1
    # reward v0.1: FLAG_SYS_* (except the kept list), FLAG_OPENED_START_MENU, FLAG_WORLD_MAP_*, FLAG_VISITED_*;
    # FLAG_PN_BOULDERS_SOLVED
    zero = (29 - len(gen_tables.SYS_FLAGS_KEPT)) + 1 + 51 + 2 + 1
    assert set(w.tolist()) == {0, 0.25, 0.5, 1, 20}
    assert w.sum() == (2304 - 32 - 191 - 768 - 13 - zero - 165) * 1 + 165 * 0.25 + 0.5 * 768 + 20 * 13
    c = np.array(T.FLAG_CATS)
    assert ((w == 0) == (c == T.FLAG_CAT_NONE)).all()
    assert (c[w == 0.25] == T.FLAG_CAT_STORY).all()   # item balls are story flags


def test_flag_weights_v01():
    c = gen_tables.Constants()
    w = lambda name: T.FLAG_WEIGHTS[c.value(name)]  # noqa: E731
    assert c.value("FLAG_OPENED_START_MENU") == 0x83E and w("FLAG_OPENED_START_MENU") == 0
    assert w("FLAG_SYS_QL_DEPARTED") == 0 and w("FLAG_SYS_SAFARI_MODE") == 0 and T.FLAG_WEIGHTS[0x800] == 0
    assert w("FLAG_WORLD_MAP_PALLET_TOWN") == 0
    assert c.value("FLAG_VISITED_OAKS_LAB") == 0x2CF and T.FLAG_WEIGHTS[0x2CF] == 0
    assert w("FLAG_VISITED_TWO_ISLAND") == 0
    for name in gen_tables.SYS_FLAGS_KEPT:
        assert w(name) == 1, name
    assert w("FLAG_SYS_POKEMON_GET") == 1
    assert w("FLAG_BADGE01_GET") == 20 and w("FLAG_DEFEATED_CHAMP") == 20 and w("FLAG_DEFEATED_LORELEI") == 20
    assert T.FLAG_WEIGHTS[0x5AB] == 0.5                   # a trainer flag
    assert T.FLAG_WEIGHTS[0x2CE] == 1                     # FLAG_TUTOR_BODY_SLAM, an ordinary story flag
    assert w("FLAG_GOT_SS_TICKET") == 1


def test_item_balls():
    """Item balls of non-key items 0.25 (v0.10; 0 in v0.7-v0.9); key-item balls and Pokémon shown as balls keep the
    story weight."""
    c = gen_tables.Constants()
    w = lambda name: T.FLAG_WEIGHTS[c.value(name)]  # noqa: E731
    for name in ("FLAG_HIDE_ROUTE2_ETHER", "FLAG_HIDE_ROUTE2_PARALYZE_HEAL", "FLAG_HIDE_ROCKET_HIDEOUT_B4F_TM49"):
        assert w(name) == 0.25, name
    for name in ("FLAG_HIDE_SILPH_SCOPE", "FLAG_HIDE_LIFT_KEY", "FLAG_HIDE_SILPH_CO_5F_CARD_KEY",
                 "FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY", "FLAG_HIDE_SAFARI_ZONE_WEST_GOLD_TEETH",
                 "FLAG_HIDE_BULBASAUR_BALL", "FLAG_HIDE_EEVEE_BALL", "FLAG_HIDE_POWER_PLANT_ELECTRODE_1"):
        assert w(name) == 1, name
    hide = c.names("FLAG_HIDE_")
    assert sum(T.FLAG_WEIGHTS[f] == 0.25 for f in set(hide.values())) == 165
    zero, key, n = gen_tables.item_balls(gen_tables._script_labels())
    assert (n, len(zero), len(key)) == (178, 165, 5) and all(n.startswith("FLAG_HIDE_") for n in zero + key)


def test_scene_vars():
    """Reward v0.7: the story variables are the 49 trigger variables (48 VAR_MAP_SCENE_* + the Vermilion ticket
    check), not every VAR_MAP_SCENE_*."""
    c = gen_tables.Constants()
    idx = T.SCENE_VAR_IDX
    assert len(idx) == 49 and idx == sorted(set(idx)) and max(idx) < 256
    assert c.value("VAR_MAP_SCENE_PALLET_TOWN_OAK") - 0x4000 == 0x50 and 0x50 in idx
    for name in ("VAR_MAP_SCENE_PALLET_TOWN_PROFESSOR_OAKS_LAB", "VAR_MAP_SCENE_SILPH_CO_11F",
                 "VAR_MAP_SCENE_VICTORY_ROAD_3F", "VAR_MAP_SCENE_MT_MOON_B2F", "VAR_VERMILION_CITY_TICKET_CHECK_TRIGGER"):
        assert c.value(name) - 0x4000 in idx, name
    for name in ("VAR_TEMP_0", "VAR_TEMP_F", "VAR_REPEL_STEP_COUNT", "VAR_ICE_STEP_COUNT", "VAR_OBJ_GFX_ID_0",
                 "VAR_QLBAK_MAP_LAYOUT", "VAR_CABLE_CLUB_STATE", "VAR_MAP_SCENE_PALLET_TOWN_RIVALS_HOUSE",
                 "VAR_MAP_SCENE_PALLET_TOWN_SIGN_LADY", "VAR_MAP_SCENE_TWO_ISLAND",
                 "VAR_MAP_SCENE_CINNABAR_ISLAND_POKEMON_LAB_EXPERIMENT_ROOM_REVIVE_STATE",
                 "VAR_MAP_SCENE_CINNABAR_ISLAND_POKEMON_LAB_EXPERIMENT_ROOM_WHICH_FOSSIL"):
        assert c.value(name) - 0x4000 not in idx, name
    scene = {v - 0x4000 for v in c.names("VAR_MAP_SCENE_").values()}
    assert len(set(idx) & scene) == 48


def test_hm_learnsets():
    hm = T.HM_LEARN
    assert hm[BULBASAUR] >> CUT & 1
    assert hm[PIDGEY] >> FLY & 1
    assert hm[SQUIRTLE] >> SURF & 1
    assert hm[MAGIKARP] == 0
    assert hm[0] == 0 and hm[T.SPECIES_EGG] == 0
    assert not hm[PIDGEY] >> SURF & 1
    assert max(hm) < 128


def test_families():
    """Evolutionary family roots (tables.FAMILY), hand-checked."""
    ivysaur, venusaur, eevee = 2, 3, 133
    assert T.FAMILY[BULBASAUR] == T.FAMILY[ivysaur] == T.FAMILY[venusaur] == BULBASAUR
    for s in (134, 135, 136, 196, 197):  # Vaporeon, Jolteon, Flareon, Espeon, Umbreon
        assert T.FAMILY[s] == eevee, s
    assert T.FAMILY[eevee] == eevee
    assert T.FAMILY[25] == T.FAMILY[26] == 172  # Pikachu, Raichu -> Pichu (the baby is the root)
    assert T.FAMILY[106] == T.FAMILY[107] == T.FAMILY[237] == 236  # Hitmonlee / chan / top -> Tyrogue
    assert T.FAMILY[0] == T.FAMILY[T.SPECIES_EGG] == 0 and T.FAMILY[252] == 0  # no family: none, egg, old Unown
    assert T.FAMILY[413] == T.FAMILY[439] == 201  # the Unown form ids -> Unown
    assert T.FAMILY_COUNT == len(T.FAMILY_ROOTS) == 202 and T.FAMILY_ROOTS == sorted(set(T.FAMILY) - {0})
    for s in range(T.SPECIES_COUNT):
        i = T.FAMILY_INDEX[s]
        assert (i == -1) == (T.FAMILY[s] == 0) and (i < 0 or T.FAMILY_ROOTS[i] == T.FAMILY[s])


def test_tm_learn():
    """Venusaur learns TM19 Giga Drain, not TM39 Rock Tomb; bits 50-56 are HM_LEARN; names."""
    venusaur = 3
    assert (T.TM_LEARN[venusaur] >> 18) & 1 and not (T.TM_LEARN[venusaur] >> 38) & 1
    assert T.MOVE_NAMES[T.TM_MOVES[18]] == "GIGA_DRAIN" and T.MOVE_NAMES[T.TM_MOVES[38]] == "ROCK_TOMB"
    assert all((T.TM_LEARN[s] >> 50) & 0x7F == T.HM_LEARN[s] for s in range(T.SPECIES_COUNT))
    assert T.TM_LEARN[MAGIKARP] == 0 and T.TM_LEARN[0] == 0
    assert T.SPECIES_NAMES[venusaur] == "VENUSAUR" and len(T.SPECIES_NAMES) == T.SPECIES_COUNT
    assert len(T.MOVE_NAMES) == T.MOVES_COUNT and T.MOVE_NAMES[0] == "NONE"
