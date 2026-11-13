"""Pure-Python parts: action tables, button parsing, seeding."""
import numpy as np
import pytest

import fireredsim as pn
from fireredsim import seeding
from fireredsim.actions import multibinary_to_buttons


def test_buttons():
    assert pn.parse_buttons("A") == 1
    assert pn.parse_buttons("b+left") == 2 | 32
    assert pn.parse_buttons("NONE") == 0
    assert pn.buttons_to_str(2 | 32) == "B+LEFT"
    with pytest.raises(ValueError):
        pn.parse_buttons("Z")


def test_action_tables():
    basic = pn.action_table("basic")
    assert basic.tolist() == [0, 1, 2, 8, 4, 64, 128, 32, 16]
    full = pn.action_table("full")
    assert len(full) == 15 and (2 | 16) in full.tolist()
    assert pn.action_table(["A", "B+UP", 0x200]).tolist() == [1, 2 | 64, 0x200]
    with pytest.raises(ValueError):
        pn.action_table([0x400])
    bits = np.zeros((2, 10), dtype=np.int8)
    bits[0, 0] = bits[1, 3] = bits[1, 7] = 1
    assert multibinary_to_buttons(bits).tolist() == [1, 8 | 128]


def test_seeding_is_deterministic_and_spread():
    s1 = seeding.episode_seeds(123, 8)
    s2 = seeding.episode_seeds(123, 8)
    assert (s1 == s2).all() and len(set(s1.tolist())) == 8
    # env i's seed does not depend on how many envs there are
    assert (seeding.episode_seeds(123, 4) == s1[:4]).all()
    assert seeding.episode_seeds(124, 1)[0] != s1[0]
    t = [seeding.timer1_for_seed(int(s)) for s in s1]
    assert all(0 <= a < 65536 and 0 <= b < 65536 for a, b in t)
    assert len(set(t)) == 8
    assert seeding.next_episode_seed(int(s1[0])) != s1[0]
    assert 0 <= seeding.noop_frames_for_seed(5, 30) <= 30
    assert seeding.noop_frames_for_seed(5, 0) == 0
