"""The native reward (reward.py terms), the action masks and the trail window on synthetic records, with the v76
weights (train.TrainConfig's defaults; the library is built for them once per process)."""
import numpy as np
import pytest

from conftest import PALLET, ROUTE1, records, set_flag, set_mon
from fireredsim.obs_dtype import OBS_DTYPE
from pokerl import env as ENV
from pokerl import mask as M
from pokerl import native as NV
from pokerl import reward as R
from pokerl import tables as T
from pokerl import train as TR

BULBASAUR, PIDGEY = 1, 16
CUT = int(T.HM_MOVES[0])
STORY_FLAG = int(np.flatnonzero(R._STORY)[0])


@pytest.fixture(scope="module", autouse=True)
def v76_weights():
    ENV.apply_reward_config(TR.TrainConfig())
    TR.ensure_native()


def term(terms, name, row=0):
    return float(terms[name][row])


def test_v76_weights():
    assert R.TILE_REWARD == 0.02 and R.TILE_COUNT_REWARD == 0.05 and R.TILE_COUNT_SCALE == 512
    assert R.MAP_REWARD == 0.05 and R.WHITEOUT_REWARD == -0.2 and R.HM_TAUGHT_REWARD == 10
    assert R.MOVE_REPLACE_REWARD == 0.5 and R._hm_paid() == ("cut", "surf", "strength")
    assert R.FLAG_WEIGHTS[R.BADGE_FLAGS].tolist() == [10.0] * 8 and R.FLAG_WEIGHTS[T.FLAG_DEFEATED_CHAMP] == 20
    w = dict(zip(R.SKILL_EVENTS, R.SKILL_WEIGHTS.tolist()))
    assert {k: v for k, v in w.items() if v} == pytest.approx({"party_menu": 0.02, "bag_menu": 0.02, "party_order": 1.0,
                                                               "pc_storage": 0.1, "pc_player": 0.02})
    assert R.SKILL_COOLDOWNS[R.SKILL_EVENTS.index("party_order")] == 2000


def test_flags_tiles_maps():
    rw = NV.NativeReward(2)
    r = records(2)
    rw.reset(r)
    reward, terms, progress, champion = rw.step(r)  # nothing new
    assert not reward.any() and not progress.any()
    set_flag(r, STORY_FLAG, rows=0)
    r["x"][1] += 1  # a new tile: the floor and the full count bonus (first first-visit of the run)
    reward, terms, progress, _ = rw.step(r)
    assert term(terms, "flags") == 1.0 and progress[0]
    assert term(terms, "tiles", 1) == pytest.approx(0.02 + 0.05) and not progress[1]  # (a tile is not progress)
    assert reward[0] == np.float32(1.0) and reward[1] == terms["tiles"][1]
    t = R.MAP_OFF[R.MAP_INDEX[PALLET[0] * 256 + PALLET[1]]] + 5 * R.MAP_W[R.MAP_INDEX[PALLET[0] * 256 + PALLET[1]]] + 6
    assert rw.tile_count[t] == 1
    reward, terms, progress, _ = rw.step(r)  # paid once per life
    assert not reward.any()
    r["map_group"][0], r["map_num"][0] = ROUTE1
    reward, terms, progress, _ = rw.step(r)
    assert term(terms, "maps") == pytest.approx(0.05) and progress[0]
    # a second life: the tile pays again, the count bonus is the same while the run's count is below S = 512
    rw.reset(records(2))
    r2 = records(2)
    r2["x"][:] = 6
    reward, terms, _, _ = rw.step(r2)
    assert terms["tiles"].tolist() == pytest.approx([0.07, 0.07])


def test_levels_capped_and_entering_mon_rebases():
    rw = NV.NativeReward(1)
    r = records(1)
    set_mon(r, 0, BULBASAUR, level=5)
    r["party"]["pid_priv"][0, 0] = 111
    r["party_count"] = 1
    rw.reset(r)
    r["party"]["level"][0, 0] = 7
    _, terms, progress, _ = rw.step(r)
    assert term(terms, "levels") == 2.0 and progress[0]
    set_mon(r, 1, PIDGEY, level=9)  # a mon enters the party (a catch): re-based, nothing paid
    r["party"]["pid_priv"][0, 1] = 222
    r["party_count"] = 2
    _, terms, _, _ = rw.step(r)
    assert term(terms, "levels") == 0.0
    r["party"]["level"][0, 1] = 10  # (K = 2 at the first gym: both mons count, each capped at Brock's ace)
    _, terms, _, _ = rw.step(r)
    assert term(terms, "levels") == 1.0
    cap = int(rw.cap[0])
    r["party"]["level"][0, 1] = cap + 20
    _, terms, _, _ = rw.step(r)
    assert term(terms, "levels") == cap - 10  # min(level, cap)


def test_hm_taught_once_and_whiteout():
    rw = NV.NativeReward(1)
    r = records(1)
    set_mon(r, 0, BULBASAUR, level=5)
    r["party_count"] = 1
    rw.reset(r)
    r["party"]["moves"][0, 0, 0] = CUT
    _, terms, progress, _ = rw.step(r)
    assert term(terms, "hm_taught") == 10.0 and progress[0]
    r["party"]["moves"][0, 0, 0] = 0  # forgotten and taught again: paid once per life
    rw.step(r)
    r["party"]["moves"][0, 0, 0] = CUT
    _, terms, _, _ = rw.step(r)
    assert term(terms, "hm_taught") == 0.0
    r["party"]["hp"][0, 0] = 0
    reward, terms, _, _ = rw.step(r)
    assert term(terms, "whiteout") == pytest.approx(-0.2)


def test_skill_cooldown_and_key_items():
    rw = NV.NativeReward(1)
    r = records(1)
    rw.reset(r)
    q = R.SKILL_EVENTS.index("party_order")
    r["skill_events_priv"] = 1 << q
    _, terms, _, _ = rw.step(r)
    assert term(terms, "skill") == 1.0
    _, terms, _, _ = rw.step(r)  # on its 2000-step cooldown
    assert term(terms, "skill") == 0.0
    r["skill_events_priv"] = 0
    r["key_items"]["id"][0, 0] = 360  # a key item id (below ITEMS_COUNT)
    _, terms, progress, _ = rw.step(r)
    assert term(terms, "key_items") == 5.0 and progress[0]


def test_move_replace_once_per_mon_and_move():
    rw = NV.NativeReward(1)
    r = records(1)
    set_mon(r, 0, BULBASAUR, level=5)
    r["party"]["pid_priv"][0, 0] = 7
    rw.reset(r)
    r["learned_move_priv"][0, 0] = 33  # learnt in place of a forgotten move
    r["learned_replaced_priv"] = 1
    reward, terms, _, _ = rw.step(r)
    assert term(terms, "move_learn") == 0.5 and reward[0] == np.float32(0.5)
    reward, terms, _, _ = rw.step(r)  # the same (mon, move) again: nothing
    assert term(terms, "move_learn") == 0.0
    r["learned_replaced_priv"] = 0  # learnt into a free slot: nothing (v76 pays replacements only)
    r["learned_move_priv"][0, 0] = 34
    _, terms, _, _ = rw.step(r)
    assert term(terms, "move_learn") == 0.0


def test_reward_is_the_term_sum_in_the_v76_order():
    rng = np.random.default_rng(0)
    t = (rng.integers(-5, 50, size=(1000, len(R.TERMS))) * rng.choice([0.0, 0.02, 0.1, 0.5], size=(1000, len(R.TERMS)))
         ).astype(np.float32)
    old_layout = np.zeros((1000, R.SUM_COLS), np.float32)
    old_layout[:, list(R.TERM_SUM_COL)] = t
    assert np.array_equal(R.reward_sum(t), old_layout.sum(axis=1))
    assert R.TERMS[R.TERM_SUM_COL.index(18)] == "interact" and R.TERMS[R.TERM_SUM_COL.index(24)] == "move_learn"


def test_carry_and_state_dict():
    rw = NV.NativeReward(2)
    r = records(2)
    set_flag(r, STORY_FLAG, rows=0)
    rw.reset(records(2))
    rw.step(r)
    d = rw.carry_capture([0], paid=True)
    rw.reset(records(1), np.array([0]))  # (as a rewind's reset: the flag would pay again)
    rw.carry_restore(np.array([0]), d)
    _, terms, _, _ = rw.step(r)
    assert term(terms, "flags") == 0.0
    sd = rw.state_dict()
    other = NV.NativeReward(1)
    other.load_state_dict(sd)
    assert np.array_equal(other.tile_count, rw.tile_count)


def test_masks():
    r = np.zeros(4, dtype=OBS_DTYPE)
    r["waiting_for_input"] = [0, 1, 1, 1]
    r["input_keys"] = M.KEY["A"] | M.KEY["B"] | M.KEY["UP"]
    r["wait_ctx"][2] = M.WAIT_START_MENU
    r["start_menu_cursor"][2] = M.START_MENU_SAVE
    r["menu_action"][3] = M.MENU_ACTION_RELEASE
    m = NV.native_action_mask(r)
    A = {a: i for i, a in enumerate(M.ACTIONS)}
    assert m[0].tolist() == [True] + [False] * 7  # not waiting: NOOP only
    assert np.flatnonzero(m[1]).tolist() == [A["A"], A["B"], A["UP"]]
    assert np.flatnonzero(m[2]).tolist() == [A["B"], A["UP"]] and np.flatnonzero(m[3]).tolist() == [A["B"], A["UP"]]
    f = NV.native_fallback_mask(r)
    assert f[0].all() and not f[2, A["A"]] and f[2].sum() == 7


def test_trail_window():
    rw = NV.NativeReward(1)
    r = records(1, x=5, y=5)
    r["tiles"] = 1  # (no OUTSIDE cells)
    rw.reset(r)
    out = np.zeros((1, R.SEEN_ROWS, R.SEEN_COLS), np.uint8)
    rw.trail_window(r, out)
    assert out[0, R.SEEN_ROW0, R.SEEN_COL0] == 255 and out.sum() == 255
    for x in (6, 7):
        r["x"] = x
        rw.step(r)
    rw.trail_window(r, out)
    c = R.SEEN_COL0
    assert out[0, R.SEEN_ROW0, c - 2:c + 1].tolist() == [253, 254, 255]
    r["tiles"][0, R.SEEN_ROW0 * R.SEEN_COLS + c - 1] = 0  # an OUTSIDE cell reads 0
    rw.trail_window(r, out)
    assert out[0, R.SEEN_ROW0, c - 1] == 0
