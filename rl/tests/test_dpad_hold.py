"""EnvConfig.dpad_hold (env.py module docstring "D-pad hold"): a D-pad action toward a direction the player
is not facing, chosen on a waiting free-overworld observation, is held longer (dpad_hold on foot and surfing,
DPAD_HOLD_BIKE on a bike) and on foot the step runs 25 frames; a D-pad action toward the facing direction, every other
action and every other state keep hold_frames and frameskip.

* the rule (``dpad_hold_frames``) on synthetic records: turn only, free overworld only, waiting only, per mount;
* with a worker, in the bedroom: a 1-frame tap toward a new direction turns without moving (two taps move one tile);
  with dpad_hold 10 one decision turns and moves one tile in 25 frames (no auto-advance chunk), the game state equals
  the tap env's after its two taps, the next step toward the same (now facing) direction is a plain 24-frame tap and
  ``last_hold`` / ``last_n`` report the per-env values (A keeps hold_frames).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import needs_worker  # noqa: E402

from fireredsim.obs_dtype import OBS_DTYPE, INPUT_OVERWORLD_FREE, INPUT_MENU  # noqa: E402

from pokerl import env as ENV  # noqa: E402
from pokerl import train as TR  # noqa: E402
from pokerl.env import dpad_hold_frames, DPAD_HOLD_BIKE, DPAD_TURN_FRAMES_FOOT, ACTION_DIR  # noqa: E402
from pokerl.mask import ACTIONS  # noqa: E402

A = {name: i for i, name in enumerate(ACTIONS)}
UP, DOWN, LEFT, RIGHT = A["UP"], A["DOWN"], A["LEFT"], A["RIGHT"]
DIRS = {DOWN: (0, 1, 1), UP: (0, -1, 2), LEFT: (-1, 0, 3), RIGHT: (1, 0, 4)}  # action -> (dx, dy, facing)


def test_rule():
    assert ACTION_DIR.tolist() == [0, 0, 0, 0, 2, 1, 3, 4]
    obs = np.zeros(10, OBS_DTYPE)
    obs["waiting_for_input"] = 1
    obs["input_kind"] = INPUT_OVERWORLD_FREE
    obs["facing"] = 1                         # facing south (DOWN)
    obs["player_flags"] = 1                   # on foot
    obs["input_kind"][2] = INPUT_MENU         # a menu: unchanged
    obs["waiting_for_input"][3] = 0           # not waiting (fallback / cap): unchanged
    obs["player_flags"][4] = 1 << 3           # surfing: the foot hold, frameskip frames
    obs["player_flags"][5] = 1 << 1           # mach bike (the Bicycle): DPAD_HOLD_BIKE, frameskip frames
    obs["player_flags"][6] = 1 << 7           # dash: unchanged
    obs["player_flags"][7] = 1 << 6           # forced movement: unchanged
    obs["player_flags"][9] = (1 << 1) | (1 << 7)  # bike + dash: unchanged
    acts = np.array([UP, A["A"], LEFT, RIGHT, UP, UP, LEFT, RIGHT, DOWN, UP])
    #                turn  A     menu  nowait surf bike dash forced facing bike+dash
    h, n = dpad_hold_frames(obs, acts, 1, 10, 24)
    assert h.tolist() == [10, 1, 1, 1, 10, DPAD_HOLD_BIKE, 1, 1, 1, 1]
    assert n.tolist() == [DPAD_TURN_FRAMES_FOOT, 24, 24, 24, 24, 24, 24, 24, 24, 24]
    assert h.dtype == n.dtype == np.int64
    # a frameskip above the turn frames keeps its own length
    h, n = dpad_hold_frames(obs, acts, 1, 12, 30)
    assert h[0] == 12 and n.tolist() == [30] * 10
    # the on-foot bit alone, facing north, pressing north: a plain tap
    obs["player_flags"][:] = 1
    obs["facing"][:] = 2
    h, n = dpad_hold_frames(obs, np.full(10, UP), 1, 10, 24)
    assert h.tolist() == [1] * 10 and n.tolist() == [24] * 10


def _env(dpad_hold, seed=3):
    """The v76 env (train.TrainConfig's settings, as train.make_envs passes them) on one env."""
    kw = {k: getattr(TR.TrainConfig(), k) for k in TR.ENV_KEYS}
    kw.update(num_envs=1, seed=seed, max_frames=384, fallback_after=72, max_episode_steps=5000, stagnation_steps=4000,
              dpad_hold=dpad_hold)
    cfg = ENV.EnvConfig(**kw)
    ENV.apply_reward_config(cfg)
    TR.ensure_native()  # (the native library is built for these weights)
    return ENV.PokeRLEnv(cfg)


def _pos(e):
    r = e._obs[0]
    return int(r["x"]), int(r["y"]), int(r["facing"])


def _step(e, a):
    _, _, _, term, trunc, info = e.step(np.array([a]))
    assert not (term[0] or trunc[0])
    return int(info["frames"][0])


@needs_worker
def test_bedroom_turn_and_step():
    tap = _env(0)
    tap.reset()
    x0, y0, f0 = _pos(tap)
    assert tap._obs[0]["input_kind"] == INPUT_OVERWORLD_FREE and tap._obs[0]["player_flags"] == 1
    # a direction the player is not facing whose tile is free: two taps (turn, then step) move one tile
    chosen = None
    for a, (dx, dy, f) in DIRS.items():
        if f == f0:
            continue
        probe = _env(0)
        probe.reset()
        _step(probe, a)
        x1, y1, f1 = _pos(probe)
        assert (x1, y1, f1) == (x0, y0, f), ("a tap toward a new direction only turns", (x1, y1, f1), (x0, y0, f))
        _step(probe, a)
        probe.close()
        if _pos(probe) == (x0 + dx, y0 + dy, f):
            chosen = a
            break
    assert chosen is not None, "no free tile next to the start"
    dx, dy, f = DIRS[chosen]
    assert tap.last_hold is None and tap.last_n is None
    assert _step(tap, chosen) == 24 and _step(tap, chosen) == 24
    assert _pos(tap) == (x0 + dx, y0 + dy, f)

    held = _env(10)
    held.reset()
    assert _pos(held) == (x0, y0, f0)
    frames = _step(held, chosen)
    assert _pos(held) == (x0 + dx, y0 + dy, f), "one decision turns and steps"
    assert held.last_hold.tolist() == [10] and held.last_n.tolist() == [DPAD_TURN_FRAMES_FOOT]
    assert frames == DPAD_TURN_FRAMES_FOOT, frames  # turn (1-8) + walk (9-24), waiting after 25: no auto-advance chunk
    assert held._obs[0]["waiting_for_input"] and held._obs[0]["input_kind"] == INPUT_OVERWORLD_FREE
    # the same game state (the frame counter differs: 25 frames against 48): the hold-10 env after one step is where
    # the tap env is after two
    for k in ("x", "y", "facing", "map_group", "map_num", "player_flags", "player_motion", "waiting_for_input",
              "input_kind", "input_keys", "field_state", "controls_locked", "tiles", "objects"):
        assert np.array_equal(held._obs[0][k], tap._obs[0][k]), k
    # already facing: a plain 24-frame tap, one tile (if free)
    frames = _step(held, chosen)
    assert held.last_hold.tolist() == [1] and held.last_n.tolist() == [24] and frames == 24
    p2 = _pos(held)
    assert p2 in ((x0 + dx, y0 + dy, f), (x0 + 2 * dx, y0 + 2 * dy, f)), p2
    # a non-D-pad action keeps hold_frames and frameskip
    held.step_async(np.array([A["A"]]))
    assert held.last_hold.tolist() == [1] and held.last_n.tolist() == [24]
    held.step_wait()
    tap.close()
    held.close()
