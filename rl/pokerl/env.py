"""The training environment: a ``fireredsim.PokeVecEnv`` of N FireRed games with the reward, the action mask, the
observation, the life ends and the episode statistics.

* ``PokeVecEnv(frameskip=24, hold_frames=1, obs_mode="privileged", action_set=ACTIONS, start=<new game, text FAST,
  battle scene OFF, style SHIFT>, copy=False, rng_reseed=True)``; env i's episode seeds derive from ``seed`` and i,
  every life draws a fresh one.
* Life ends: the Champion defeated (``FLAG_DEFEATED_CHAMP`` newly set: ``terminated``, its reward in that step); a
  relay area exit (below: ``terminated``); the length cap ``max_episode_steps``; stagnation (S = ``stagnation_steps``
  steps without a progress event, the window drawn per life from U[0.75 S, 1.25 S] so that envs do not end in
  waves; 0 = off); the simulator's own terminal (``info["sim_terminal"]``). All but the first two are ``truncated``.
* Stagger: the first life of every env has a length cap drawn from U[``stagger_min``, ``stagger_max``] (seeded), so
  the envs' ends spread over the run; later lives get ``max_episode_steps``.
* Progress events (the stagnation window restarts): the reward's (a flag of weight > 0, a new map, a capped level
  gain, an HM gain, a scene variable or a new key item), a rise of the life's total party level above its maximum
  (no pay), and an opponent battler knocked out in any battle (no pay; at 8 badges a level takes ~14 wild KOs,
  longer than the stagnation window). The level maximum and the opponents' state start from a life's first record.
* Same-step reset: an ended env is reset at once (``PokeVecEnv.reset(env_ids=...)``), so the returned observation
  of that env is the first one of its next life and the returned reward is the last one of the ended life
  (CleanRL / SB3 convention). PokeVecEnv's own autoreset never triggers; if it does (``info["autoreset"]``) the row
  starts a new life with reward 0.

``step`` returns ``(obs, mask, reward, terminated, truncated, info)``: ``obs`` = ``{"pixels": uint8 [N, 80, 120],
"trail": uint8 [N, 11, 15]}`` (valid until the next call), ``mask`` bool [N, 8], ``reward`` float32 [N],
``info["terms"]`` the per-term rewards, ``info["episodes"]`` a list of finished-life statistics (``term_<t>`` per
term), ``info["waiting"]`` the waiting_for_input bits of the observation the actions were chosen on.

Observation: the gray2 screen (the luma of the step's last frame averaged over 2 x 2 pixel blocks: 80 x 120 uint8,
rendered by the worker) and the trail plane, an 11 x 15 window of map tiles around the player: 255 - the moves since
the env last entered the tile (0 beyond 254 moves; a new life and a rewind restore start a fresh trail).

Decision-point stepping: the workers run the step's 24 frames and then 24-frame NOOP chunks while the game is not
waiting for input, up to ``max_frames`` frames per step (384), so every returned step is a decision step.
``info["frames"]`` holds the frames each step ran; ``drain_frames()`` returns their totals.

D-pad hold (``dpad_hold``, 0 = off): a D-pad action toward a direction the player is not facing, on a waiting
free-overworld observation, is held longer than ``hold_frames`` so that one decision turns and moves one tile: on
foot ``dpad_hold`` frames (a hold of 9-16 turns and walks one tile; the step runs 25 frames so that it ends when the
game waits again), surfing ``dpad_hold`` (9-16), on a bike ``DPAD_HOLD_BIKE`` (2-7); underwater, forced movement and
dash: ``hold_frames``. Already facing, a hold would double-step on the bike and surfing, hence the facing test.

Mask (pokerl.native, mask.py): NOOP only while the game does not wait for input; else the buttons the game reads
(NOOP excluded), A forbidden on the start menu's dead entries, the save prompt and RELEASE. Mask fallback:
``waiting_for_input`` has no false positives but can have false negatives, so an env whose observations have not
waited for ``fallback_after`` steps' worth of frames gets every action until it waits again (``fallback_mask``);
``drain_fallback()`` returns the activations.

Split step (pipelined collector, ppo.py): ``step_async(actions)`` posts the actions and returns at once;
``step_wait()`` = ``step_wait_begin()`` + ``step_wait_end()`` waits for the workers and does the host work. With
``step_wait_begin(async_reset=True)`` the resets of ended envs run in a helper thread so that the collector can
serve the other env group meanwhile; the trajectories are unchanged. ``tile_count`` (constructor) shares the
reward's run-long tile visit table between env groups. ``timing`` accumulates seconds per phase.

Rewind (``rewind``): every env keeps a ring of its own recent captures (``rewind.RewindRing``, ``rewind_ring`` per
env). In ``step_wait_begin``, after the reward and before any reset, the rows with a progress event (not ending)
whose newest capture is at least ``rewind_min_gap`` steps old (or that have none) are captured: one fast snapshot op
(at most ``rewind.MAX_CAPTURES_PER_STEP`` rows, oldest newest-capture first) writes them into worker snapshot slots
reserved for the rings (env i's ring slot s is ``reserved_base + i * rewind_ring + s``); ``info["capture"]`` =
``{rows, slots, serials}`` (the collector stores the matching policy state). A life end by stagnation only (not the
cap, the Champion, a relay exit or a simulator terminal) of an env that holds a capture and has rewinds left is a
**rewind**: a capture chosen uniformly among the newest ``rewind_depth`` (newer ones discarded) is restored with a
fresh RNG stream and the life goes on (``ep_len``, ``ep_return``, ``ep_terms`` and the length cap kept; a new
stagnation window, the mask fallback counters restarted). The reward's per-episode state is re-initialised from the
restored record, then the life's visited tiles and maps and its "already paid" state (``carry_paid``: the flags, key
items and scene-variable maxima seen, the level-sum maximum, the object tiles used, the interaction keys, the skill
cooldowns, the HM bits and the move keys paid) are put back as they were at the capture, as are the relay scope
status and pays, the unique terms' marks and the Indigo Plateau mark: only what the life did after the capture pays
again. The row is ``truncated`` and listed in ``info["rewind"]`` = ``{rows, slots, serials}``, with no episode dict.
The life's budget: one uniform draw u per life, ``rewind_max_min`` + floor(u (``rewind_max`` + ``rewind_per_badge``
b - ``rewind_max_min`` + 1)) for the most badges b the life has held (``rewind_budget_of``: 0..3 at 0 badges, 0..4
at 1, ...; it only grows). Every episode dict has ``rewinds``, ``rewind_budget``, ``rewind_progress`` (progress
events after the first rewind) and ``end`` ("relay_exit", "champion", "sim", "cap", "stagnation").

Levels at Indigo Plateau (``level_reward_indigo``, -1 = off): an in-game record on one of ``INDIGO_MAPS`` marks the
life as arrived, and from that step to the life's end the levels term pays at that weight instead of the levels
weight (episode ``indigo``). A rewind to before the arrival turns it off until the life arrives again.

Unique purchases and item uses (``unique_mart_reward``, ``unique_consumable_reward``, ``unique_battle_item_reward``):
paid once per key and life into the step's reward (episode ``term_unique_mart`` / ``term_unique_consumable`` /
``term_unique_battle_item``, in no ``reward.TERMS`` term), on in-game rows. A ``mart_buy`` skill event keys the
record's map (one key per mart map); a ``use_consumable`` (on the field) or ``use_battle_item`` (in battle) event
keys every item id of which the record's items + berries pockets hold fewer than the env's previous record
(normally exactly the item used). Field and battle uses are separate sets; TMs are in neither.

Relay scopes (relay/scopes.py, relay/exits.py): six regions of the game in story order (Vermilion, Celadon,
Fuchsia, Saffron, Cinnabar, Viridian), per env a status ``relay_status`` int8 [N, 6] (INACTIVE / ACTIVE / CLEARED).
A scope activates on its home map once its entry condition holds and every scope before it is CLEARED (paying
``relay_activation_reward``), and clears when its clear condition holds (paying ``relay_clear_reward``; the last
one's clear is the Champion); each pays at most once per scope and life. While a scope is ACTIVE the life is
confined to its area: the D-pad presses that would walk out of it (arrow, stair and step warps, map connections)
are taken out of the mask on free-overworld records, and a life found outside its area all the same (Fly, Dig,
Escape Rope, Teleport, a whiteout's respawn) ends: ``terminated``, ``end`` "relay_exit" (Viridian's detour maps,
the Sevii Islands 1-3 behind Bill's boat, do not count as outside). ``drain_relay()`` counts activations, exits
and masked records. Every episode dict has ``relay_active`` (the ACTIVE scope at the end or -1),
``relay_cleared`` and ``term_relay``.

Milestone events (``events.EVENTS``): ``life_events`` bool [N, E] holds, per env, the events true in any
observation of its current life; it is cleared when a life starts, never at a rewind. ``event_fracs()`` = its mean
over the envs; every episode dict has ``events`` (the indices of the life's events).

New-game preset (``preset_flags`` / ``preset_vars``): event flags and game variables the worker sets in every new
game before its start cache: the Strength puzzles of Seafoam Islands and Victory Road count as solved
(``PRESET_BOULDERS_SOLVED_*``). They are set in the life's first record, so the reward takes them as the baseline.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import fireredsim
from fireredsim.seeding import rng_stream_seed

from . import events as EV
from . import native as NV
from . import rewind as RW
from .relay import exits as RX
from .relay import scopes as RSC
from .mask import ACTIONS
from . import reward as R
from .reward import TERMS

# direction code of each action, as the observation's ``facing`` (1 S, 2 N, 3 W, 4 E); 0: not a D-pad action
ACTION_DIR = np.array([{"DOWN": 1, "UP": 2, "LEFT": 3, "RIGHT": 4}.get(a, 0) for a in ACTIONS], dtype=np.int64)
# gPlayerAvatar.flags bits (global.fieldmap.h PLAYER_AVATAR_FLAG_*)
FLAG_BIKE = (1 << 1) | (1 << 2)   # mach bike (the Bicycle), acro bike
FLAG_SURF = 1 << 3
FLAG_NO_HOLD = (1 << 4) | (1 << 6) | (1 << 7)  # underwater, forced movement, dash: hold_frames
DPAD_HOLD_BIKE = 4          # measured: 2-7 turn + one 6-frame tile, 8+ two (module docstring "D-pad hold")
DPAD_TURN_FRAMES_FOOT = 25  # on foot a turn + walk waits for input after frame 25 (8-frame turn, 16-frame tile)
_INPUT_OVERWORLD_FREE = 5   # fireredsim.obs_dtype.INPUT_OVERWORLD_FREE


def dpad_hold_frames(obs, actions: np.ndarray, hold: int, dpad_hold: int, frameskip: int):
    """Per-env ``(hold_frames, n_frames)`` (int64 [N] each) of the step that applies ``actions`` (discrete,
    mask.ACTIONS indices) to the envs whose last records are ``obs`` (module docstring "D-pad hold"): a D-pad action
    toward a direction other than ``facing`` on a waiting free-overworld observation is held ``dpad_hold`` frames on
    foot and surfing, ``DPAD_HOLD_BIKE`` on a bike, with ``max(frameskip, DPAD_TURN_FRAMES_FOOT)`` frames on foot;
    everything else ``hold`` and ``frameskip``."""
    a = np.asarray(actions).reshape(-1)
    h = np.full(a.size, int(hold), dtype=np.int64)
    n = np.full(a.size, int(frameskip), dtype=np.int64)
    d = ACTION_DIR[a]
    facing = np.asarray(obs["facing"]).reshape(-1).astype(np.int64)
    turn = (d != 0) & (d != facing) & (np.asarray(obs["input_kind"]).reshape(-1) == _INPUT_OVERWORLD_FREE) & \
        (np.asarray(obs["waiting_for_input"]).reshape(-1) != 0)
    flags = np.asarray(obs["player_flags"]).reshape(-1).astype(np.int64)
    turn &= (flags & FLAG_NO_HOLD) == 0
    bike = turn & ((flags & FLAG_BIKE) != 0)
    foot = turn & ((flags & (FLAG_BIKE | FLAG_SURF)) == 0)
    h[turn] = int(dpad_hold)
    h[bike] = DPAD_HOLD_BIKE
    n[foot] = max(int(frameskip), DPAD_TURN_FRAMES_FOOT)
    return h, n


log = logging.getLogger(__name__)


def apply_reward_config(cfg) -> None:
    """reward.configure with the reward fields of ``cfg`` (an EnvConfig or train.TrainConfig)."""
    g = lambda k, d=None: getattr(cfg, k, d)  # noqa: E731
    R.configure(g("skill_reward"), g("skill_menu_reward"), g("trainer_reward"),
                temp_flag_reward=g("temp_flag_reward"), skill_event_reward=g("skill_event_reward"),
                badge_reward=g("badge_reward"), elite4_reward=g("elite4_reward"), champion_reward=g("champion_reward"),
                tile_reward=g("tile_reward"), map_reward=g("map_reward"), whiteout_reward=g("whiteout_reward"),
                trash_reward=g("trash_reward"), trash_unlock_reward=g("trash_unlock_reward"),
                mansion_switch_reward=g("mansion_switch_reward"), spinner_reward=g("spinner_reward"),
                warp_pad_reward=g("warp_pad_reward"), interact_reward=g("interact_reward"),
                hm_taught_reward=g("hm_taught_reward"), tile_count_reward=g("tile_count_reward"),
                tile_count_scale=g("tile_count_scale"), level_reward=g("level_reward"),
                skill_event_cooldown=g("skill_event_cooldown"), move_replace_reward=g("move_replace_reward"),
                hm_reward_moves=g("hm_reward_moves"))


# the unique purchase and item-use terms (module docstring "Unique purchases and item uses"): EnvConfig weight, the
# skill event (reward.SKILL_EVENTS) whose bit the term reads, the episode dict's term key
UNIQUE_KEYS = ("unique_mart_reward", "unique_consumable_reward", "unique_battle_item_reward")
UNIQUE_EVENTS = ("mart_buy", "use_consumable", "use_battle_item")
UNIQUE_TERMS = ("term_unique_mart", "term_unique_consumable", "term_unique_battle_item")
UNIQUE_ITEMS = R.T.ITEMS_COUNT  # (item ids 1 .. ITEMS_COUNT - 1; others ignored)
UNIQUE_BITS = tuple(R.SKILL_EVENTS.index(e) for e in UNIQUE_EVENTS)  # (bits of skill_events_priv)
UNIQUE_MARKS = ("uq_mart", "uq_consumable", "uq_battle_item")  # (the per-life marks: rewind captures)
# the levels term's weight once a life reached Indigo Plateau (EnvConfig.level_reward_indigo): the maps whose record
# marks the life as arrived (the exterior, its Pokemon Center, the League's rooms)
INDIGO_MAPS = ("MAP_INDIGO_PLATEAU_EXTERIOR", "MAP_INDIGO_PLATEAU_POKEMON_CENTER_1F",
               "MAP_INDIGO_PLATEAU_POKEMON_CENTER_2F", "MAP_POKEMON_LEAGUE_LORELEIS_ROOM",
               "MAP_POKEMON_LEAGUE_BRUNOS_ROOM", "MAP_POKEMON_LEAGUE_AGATHAS_ROOM", "MAP_POKEMON_LEAGUE_LANCES_ROOM",
               "MAP_POKEMON_LEAGUE_CHAMPIONS_ROOM", "MAP_POKEMON_LEAGUE_HALL_OF_FAME")


def level_indigo_plan(cfg: "EnvConfig"):
    """(bool [MAP_COUNT + 1]: the INDIGO_MAPS by reward.MAP_INDEX, the factor - 1 on the levels term) of
    ``cfg.level_reward_indigo`` against the levels weight in force (reward.LEVEL_REWARD, after apply_reward_config), or
    None when off (-1)."""
    w = cfg.level_reward_indigo
    if w == -1.0:
        return None
    if w < 0:
        raise ValueError(f"PokeRLEnv: level_reward_indigo {w} must be >= 0 or -1 (off)")
    if R.LEVEL_REWARD <= 0:
        raise ValueError(f"PokeRLEnv: level_reward_indigo needs the levels term's weight > 0 (level_reward "
                         f"{R.LEVEL_REWARD:g}): it scales the term")
    lut = np.zeros(R.MAP_COUNT + 1, dtype=bool)
    lut[[EV.MAP_IDS[m] for m in INDIGO_MAPS]] = True
    return lut, np.float32(w / R.LEVEL_REWARD - 1.0)


def _bag_totals(ids: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """int64 [k, UNIQUE_ITEMS]: per row the quantity held of each item id (slots summed; ids 0 and >= UNIQUE_ITEMS
    dropped) from the pockets' ids and quantities [k, slots]."""
    k = ids.shape[0]
    out = np.zeros((k, UNIQUE_ITEMS), dtype=np.int64)
    ok = (ids > 0) & (ids < UNIQUE_ITEMS)
    r = np.broadcast_to(np.arange(k)[:, None], ids.shape)
    np.add.at(out, (r[ok], ids[ok]), qty[ok])
    return out


@dataclass
class EnvConfig:
    num_envs: int = 256
    seed: int = 0
    max_episode_steps: int = 327680
    stagnation_steps: int = 2048  # 0 = no stagnation end (then no rewind either)
    stagger_min: int = 2048       # the first life's length cap is drawn from U[stagger_min, stagger_max]
    stagger_max: int = 32768      # (0: max_episode_steps)
    frameskip: int = 24
    hold_frames: int = 1
    dpad_hold: int = 10       # hold of a D-pad action toward a new direction on foot / surfing (module docstring
                              # "D-pad hold"; 0 = off, 9-16 = turn + one tile in one decision; the bike uses 4)
    num_workers: Optional[int] = None
    worker_path: Optional[str] = None
    fallback_after: int = 72  # non-waiting steps' worth of frames before the mask fallback (0 = off)
    bedroom_cache: int = 1    # start states cached per worker process (one env per process needs one)
    cpus: Optional[Sequence[int]] = None  # pin worker w to cpus[w % len(cpus)] (None: not pinned)
    max_frames: int = 16 * 24  # frames per step at most (decision-point stepping; the requested frameskip included)
    rewind: bool = True         # on stagnation, retry from the env's own recent captures (module docstring "Rewind")
    rewind_max: int = 3         # the rewind budget's upper end at 0 badges
    rewind_max_min: int = 0     # its lower end (every life draws its budget; module docstring "Rewind")
    rewind_per_badge: int = 1   # the upper end rises by this per badge the life has held
    rewind_ring: int = 1        # captures kept per env
    rewind_depth: int = 1       # a rewind picks uniformly among the newest min(depth, available) captures
    rewind_min_gap: int = 512   # steps between two captures of an env at least (ep_len - newest capture's ep_step)
    carry_paid: bool = True     # rewind captures also carry the reward's per-episode "already paid" state
    # reward (reward.configure; module docstring of reward.py)
    skill_reward: float = 0.1
    skill_menu_reward: Optional[float] = 0.02
    trainer_reward: Optional[float] = 1.0
    temp_flag_reward: float = 1.0
    skill_event_reward: Optional[Dict[str, float]] = None
    skill_event_cooldown: Optional[Dict[str, int]] = None
    badge_reward: Optional[List[float]] = None
    elite4_reward: Optional[float] = None
    champion_reward: Optional[float] = None
    hm_reward_moves: Optional[List[str]] = None
    tile_reward: Optional[float] = None
    tile_count_reward: Optional[float] = None
    tile_count_scale: Optional[float] = None
    map_reward: Optional[float] = None
    whiteout_reward: Optional[float] = None
    trash_reward: float = 0.0
    trash_unlock_reward: Optional[float] = None
    mansion_switch_reward: float = 0.0
    spinner_reward: float = 0.0
    warp_pad_reward: float = 0.0
    interact_reward: float = 0.0
    hm_taught_reward: Optional[float] = None
    level_reward: Optional[float] = None
    level_reward_indigo: float = -1.0
    move_replace_reward: Optional[float] = None
    unique_mart_reward: float = 0.0
    unique_consumable_reward: float = 0.0
    unique_battle_item_reward: float = 0.0
    relay_activation_reward: float = 0.0  # (>= 0) paid on the step a scope activates (once per scope and life)
    relay_clear_reward: float = 0.0       # (>= 0) paid on the step the ACTIVE scope clears (once per scope and life)
    preset_flags: Tuple[str, ...] = ()  # FLAG_* names set in every new game (TFLAG_TRAINER_* too: events.flag_id)
    preset_vars: Tuple[str, ...] = ()   # "VAR_*=VALUE" (VALUE decimal or 0x hex, 0..65535)


# The Strength-boulder puzzles count as solved from the new game. Victory Road: the four floor-switch variables at
# 100 (the on-load scripts skip SetRockBarrier*, the switches are no-ops), kept across Route 23 by
# FLAG_PN_BOULDERS_SOLVED (its transition script resets them otherwise: game/NATIVE_CHANGES.md); Seafoam Islands: both
# currents stopped (the no-current layouts of B3F / B4F; Route 20 then leaves the boulder flags alone) and all 14
# boulders hidden (nothing to push).
PRESET_BOULDERS_SOLVED_FLAGS = (
    "FLAG_PN_BOULDERS_SOLVED", "FLAG_STOPPED_SEAFOAM_B3F_CURRENT", "FLAG_STOPPED_SEAFOAM_B4F_CURRENT",
    "FLAG_HIDE_SEAFOAM_1F_BOULDER_1", "FLAG_HIDE_SEAFOAM_1F_BOULDER_2",
    "FLAG_HIDE_SEAFOAM_B1F_BOULDER_1", "FLAG_HIDE_SEAFOAM_B1F_BOULDER_2",
    "FLAG_HIDE_SEAFOAM_B2F_BOULDER_1", "FLAG_HIDE_SEAFOAM_B2F_BOULDER_2",
    *(f"FLAG_HIDE_SEAFOAM_B3F_BOULDER_{i}" for i in range(1, 7)),
    "FLAG_HIDE_SEAFOAM_B4F_BOULDER_1", "FLAG_HIDE_SEAFOAM_B4F_BOULDER_2")
PRESET_BOULDERS_SOLVED_VARS = ("VAR_MAP_SCENE_VICTORY_ROAD_1F=100", "VAR_MAP_SCENE_VICTORY_ROAD_2F_BOULDER1=100",
                               "VAR_MAP_SCENE_VICTORY_ROAD_2F_BOULDER2=100", "VAR_MAP_SCENE_VICTORY_ROAD_3F=100")


def parse_preset_var(text: str) -> Tuple[str, int]:
    """``"VAR_NAME=VALUE"`` -> (name, value); ValueError for a malformed entry or a value outside 0..65535."""
    name, eq, val = str(text).partition("=")
    name = name.strip()
    try:
        v = int(val.strip(), 0) if eq else None
    except ValueError:
        v = None
    if not name or v is None or not 0 <= v <= 0xFFFF:
        raise ValueError(f"preset var {text!r}: expected VAR_NAME=VALUE with VALUE 0..65535")
    return name, v


def resolve_preset(flags: Sequence[str] = (), vars: Sequence[str] = ()) -> Tuple[List[int], Dict[int, int]]:
    """(flag ids, {var id: value}) of a preset's ``FLAG_*`` names and ``"VAR_*=VALUE"`` entries, from the game's
    headers (events.flag_id: saved flags 0 .. FLAGS_COUNT - 1; vars VARS_START .. VARS_END). Duplicate flags are
    dropped; a var named twice with different values, an unknown name or an id outside those ranges raises
    ValueError."""
    c = EV._constants()
    fids: List[int] = []
    for n in flags:
        try:
            f = EV.flag_id(str(n).strip())
        except KeyError as e:
            raise ValueError(f"preset flag {n!r}: {e}") from None
        if f == 0:
            raise ValueError(f"preset flag {n!r}: flag 0 cannot be set")
        if f not in fids:
            fids.append(f)
    v0, v1 = c.value("VARS_START"), c.value("VARS_END")
    out: Dict[int, int] = {}
    for text in vars:
        name, val = parse_preset_var(text)
        try:
            vid = c.value(name)
        except KeyError as e:
            raise ValueError(f"preset var {text!r}: {e}") from None
        if not (name.startswith("VAR_") and v0 <= vid <= v1):
            raise ValueError(f"preset var {name} = {vid:#x}: not a saved game variable ({v0:#x}..{v1:#x})")
        if out.get(vid, val) != val:
            raise ValueError(f"preset var {name}: two values ({out[vid]}, {val})")
        out[vid] = val
    return fids, out


# The start of every life: the new game replayed from power-on to Pallet Town (worker/new_game_to_pallet.txt, the
# script the worker also embeds for its "bedroom" start), then the option menu set to text FAST, battle scene OFF,
# style SHIFT (after a trainer's mon faints the game asks "Will <PLAYER> change POKeMON?")
_HERE = os.path.dirname(os.path.abspath(__file__))
START_SCRIPTS = (os.path.normpath(os.path.join(_HERE, "..", "..", "worker", "new_game_to_pallet.txt")),
                 os.path.join(_HERE, "starts", "bedroom_fast_shift_options.txt"))


def run_start():
    """The PokeVecEnv start of every life (``START_SCRIPTS``, replayed from power-on per episode seed)."""
    return fireredsim.ScriptStart(list(START_SCRIPTS))


def _state(obs):
    """The records of a PokeVecEnv observation (``{"state", "pixels"}`` with pixels on, else the records)."""
    return obs["state"] if isinstance(obs, dict) else obs


def popcount8(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(np.asarray(x, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


PIXEL_FORMAT, PIXEL_SHAPE = "gray2", (80, 120)  # the screen: fireredsim's gray2 pixel format, uint8 frames
TIMING_KEYS = ("post", "wait", "reward", "reset", "observe", "other", "rewind", "relay")
HEADER_RNG_STREAM = slice(64, 72)  # EnvHost::rng_stream in a snapshot's header (a rewind writes a fresh one)

_EMPTY_ROWS = np.zeros(0, dtype=np.int64)
_EMPTY_ROWS.setflags(write=False)  # (shared by every info["capture"] / info["rewind"] without rows)

VALID_INGAME = 1  # fireredsim.obs_dtype VALID_BITS["INGAME"]
VALID_BATTLE_MONS = 8  # fireredsim.obs_dtype VALID_BITS["BATTLE_MONS"]: the battlers block holds gBattleMons
KO_BATTLERS = 4      # battler slots (PnBattler[4], MAX_BATTLERS_COUNT)
KO_PARTY = 6         # enemy_party_priv slots
KO_NO_SLOT = 255     # ko_idx: no active opponent in the slot

_PARTY_SLOTS = np.arange(16, dtype=np.int64)[None, :]  # (party slot index per column: _level_total)


class PokeRLEnv:
    def __init__(self, cfg: EnvConfig, tile_count: Optional[np.ndarray] = None):
        """``tile_count``: the reward's run-long tile visit table to share (``reward.tile_count`` of another env
        group)."""
        apply_reward_config(cfg)  # (before the native reward reads the constants / weights)
        if cfg.rewind and cfg.stagnation_steps <= 0:
            raise ValueError("PokeRLEnv: rewind acts on stagnation ends only: it needs stagnation_steps > 0")
        if cfg.rewind:
            if cfg.rewind_ring < 1 or cfg.rewind_depth < 1 or cfg.rewind_max < 0 or cfg.rewind_min_gap < 0:
                raise ValueError("PokeRLEnv: rewind_ring and rewind_depth must be >= 1, rewind_max and "
                                 "rewind_min_gap >= 0")
            if not 0 <= cfg.rewind_max_min <= cfg.rewind_max:
                raise ValueError(f"PokeRLEnv: rewind_max_min {cfg.rewind_max_min} must be 0 .. rewind_max "
                                 f"{cfg.rewind_max}")
        if cfg.rewind_per_badge < 0:
            raise ValueError(f"PokeRLEnv: rewind_per_badge {cfg.rewind_per_badge} must be >= 0")
        for k in UNIQUE_KEYS + ("relay_activation_reward", "relay_clear_reward"):
            if getattr(cfg, k) < 0:
                raise ValueError(f"PokeRLEnv: {k} {getattr(cfg, k)} must be >= 0")
        self._lvl_indigo = level_indigo_plan(cfg)
        self.cfg = cfg
        n = self.num_envs = int(cfg.num_envs)
        kw = {"bedroom_cache": cfg.bedroom_cache, "rng_reseed": True}
        # new-game preset (module docstring): ids from the game's headers
        self.preset_flags, self.preset_vars = resolve_preset(cfg.preset_flags, cfg.preset_vars)
        if self.preset_flags or self.preset_vars:
            kw["preset_flags"], kw["preset_vars"] = self.preset_flags, self.preset_vars
        if cfg.cpus is not None:
            kw["cpus"] = list(cfg.cpus)
        # rewind captures resident in the worker snapshot slots: rewind_ring reserved slots per env after the
        # scratch slots of snapshot()/restore()
        if cfg.rewind:
            kw["reserved_slots"] = n * int(cfg.rewind_ring)
        # the screen: a pixel slot is exactly one frame of the format, so the mapped frames are C-contiguous and the
        # collector's step graphs upload them in place (ppo.Staging.pixels_host)
        self.pixel_format, self.pixel_shape = PIXEL_FORMAT, PIXEL_SHAPE
        kw.update(pixels=True, render=False, pixel_format=self.pixel_format)
        self.env = fireredsim.PokeVecEnv(
            num_envs=n, num_workers=cfg.num_workers, frameskip=cfg.frameskip, hold_frames=cfg.hold_frames,
            obs_mode="privileged", action_set=ACTIONS, start=run_start(), seed=cfg.seed, copy=False,
            worker_path=cfg.worker_path, auto_advance=True, max_frames=cfg.max_frames, **kw)
        try:
            if self.env.obs_kind != "obs":
                raise RuntimeError(f"worker records ({self.env.obs_dtype.itemsize} bytes) do not match "
                                   f"fireredsim.obs_dtype: rebuild the worker")
            NV.load()
            self._pix = self._pixel_view()  # the vec env's mapped frames [N, *pixel_shape]
        except Exception:
            self.env.close()
            raise
        self.reward = NV.NativeReward(n, tile_count=tile_count)
        self._mask_fn, self._fallback_fn = NV.native_action_mask, NV.native_fallback_mask
        # the trail plane, refreshed by every _observe (replaceable, e.g. by the collector's staging array)
        self.plane_out = {"trail": np.zeros((n, R.SEEN_ROWS, R.SEEN_COLS), dtype=np.uint8)}
        rng = self.rng = np.random.default_rng([cfg.seed, 0x5EED])
        smax = min(cfg.stagger_max, cfg.max_episode_steps) if cfg.stagger_max > 0 else cfg.max_episode_steps
        self.length_cap = rng.integers(min(cfg.stagger_min, smax), smax + 1, size=n).astype(np.int64)
        self.stag_cap = self._draw_stagnation(n)
        self.ep_len = np.zeros(n, dtype=np.int64)
        self.since_progress = np.zeros(n, dtype=np.int64)
        self.level_best = np.zeros(n, dtype=np.int64)      # the life's total party level maximum
        # per battler slot the hp_priv / party_index_priv of the opponents that were active in the previous in-game
        # record (0 / 255 elsewhere: not in battle, the own side, no healthbox)
        self.ko_hp = np.zeros((n, KO_BATTLERS), dtype=np.uint16)
        self.ko_idx = np.full((n, KO_BATTLERS), KO_NO_SLOT, dtype=np.uint8)
        self.ep_return = np.zeros(n, dtype=np.float64)
        self.ep_terms = np.zeros((n, len(TERMS)), dtype=np.float64)
        self.ep_tile_bonus = np.zeros(n, dtype=np.float64)  # count-based part of the tiles term
        self._waiting = np.zeros(n, dtype=bool)
        self.nonwait_frames = np.zeros(n, dtype=np.int64)  # frames of consecutive non-waiting observations
        self._frames = np.full(n, cfg.frameskip, dtype=np.int64)  # frames of the observation being observed
        self.in_fallback = np.zeros(n, dtype=bool)       # mask fallback active for the current observation
        self.ep_fallback = np.zeros(n, dtype=np.int64)   # fallback activations this episode
        self._fallback_events: List[Dict] = []           # activations not yet returned in info
        self._drain = {"events": [], "steps": 0, "fallback_steps": 0}
        self._frames_drain = self._new_frames_drain()
        self._obs = None
        self._pending = None  # actions posted by step_async
        self.last_hold = None  # per-env hold_frames of the last posted step (None: hold_frames throughout)
        self.last_n = None     # per-env n_frames of the last posted step (None: frameskip throughout)
        if not 0 <= int(cfg.dpad_hold) <= 0xFFFF:
            raise ValueError("dpad_hold must be 0 (off) .. 65535")
        if cfg.dpad_hold and cfg.frameskip < max(cfg.dpad_hold, DPAD_HOLD_BIKE):
            raise ValueError(f"dpad_hold {cfg.dpad_hold} exceeds frameskip {cfg.frameskip}")
        self._mid = None      # state between step_wait_begin and step_wait_end
        self._executor = None  # helper thread of asynchronous resets
        self.timing = dict.fromkeys(TIMING_KEYS, 0.0)
        self.env_offset = 0  # index of env 0 in the whole collector (set by the trainer)
        # rewind (module docstring): per life, the rewinds used and the progress events after the first one (kept
        # also without rewind: the episode-dict fields)
        self.rewinds_used = np.zeros(n, dtype=np.int64)
        self.rw_progress = np.zeros(n, dtype=np.int64)
        # the life's rewind budget: one uniform draw per life (rw_u), scaled to the most badges it has held
        self.rewind_budget = np.full(n, cfg.rewind_max, dtype=np.int64)
        self._rwb_rng = np.random.default_rng([int(cfg.seed), 0x5E1DB06E])  # (the budget draws; collector thread only)
        self._rwb_badge = bool(cfg.rewind and cfg.rewind_per_badge > 0)
        self.rw_u = np.zeros(n, dtype=np.float64)
        self.rw_badges = np.zeros(n, dtype=np.int64)
        self._unique_alloc(cfg, n)
        self.lvl_indigo = np.zeros(n, dtype=bool)  # the life has been on an Indigo Plateau map
        self._rw_lvl_indigo: Optional[np.ndarray] = None  # (its copy per rewind capture: env, ring slot)
        # the reward's per-episode counters of the life's segments before its last rewind (the reward restarts at a
        # rewind): flags earned, and maps visited not counting each later segment's start map
        self.life_flags = np.zeros(n, dtype=np.int64)
        self.life_maps = np.zeros(n, dtype=np.int64)
        self.life_events = np.zeros((n, EV.NUM_EVENTS), dtype=bool)  # milestone events seen in the current life
        # relay scopes (module docstring "Relay scopes"): per-env status, the pays of this life
        self._scopes = RSC.scope_set(ordered=True, confine=True)
        ns = self._scopes.n
        self.relay_status = np.zeros((n, ns), dtype=np.int8)
        self.relay_bonus = np.zeros(n, dtype=np.float32)    # activation + clear reward paid this life ("term_relay")
        self._rel_paid_act = np.zeros((n, ns), dtype=bool)  # activation / clear of the scope paid this life (a
        self._rel_paid_clr = np.zeros((n, ns), dtype=bool)  # rewind restores the capture's)
        self._rel_act_reward = np.float32(cfg.relay_activation_reward)
        self._rel_clr_reward = np.float32(cfg.relay_clear_reward)
        self._exit_table = RX.exit_table(self._scopes)
        self._rel_false = np.zeros(n, dtype=bool)
        self._rel_false.setflags(write=False)               # (shared read-only "nothing this step" rows)
        self._rel_exit = self._rel_false                    # this step's area exits (_end_stats, _rewind_plan)
        self._rel_drain = self._new_relay_drain()
        self.rewind: Optional[RW.RewindRing] = None
        self._rw_drain = self._new_rw_drain()
        if cfg.rewind:
            if self.env.snapshot_slots < 1:
                raise RuntimeError("rewind needs worker snapshot slots (snapshot_slots >= 1)")
            self.rewind = RW.RewindRing(n, cfg.rewind_ring, capture_bytes=int(self.env.snapshot_size))
            if self.env.reserved_slots != n * cfg.rewind_ring:
                raise RuntimeError("rewind: the worker snapshot slots of the captures were not reserved")
            self._rw_base = int(self.env.reserved_base)  # env i's capture in ring slot s: _rw_base + i * ring + s
            # capture choice and fresh RNG streams of rewinds; drawn on the collector thread only (step_wait_begin)
            self._rw_rng = np.random.default_rng([cfg.seed, 0x5E1D5EED])
            # per (env, ring slot), written in _rewind_step, read in _rewind_end (both on the collector thread): the
            # reward's tile bitmap and visited maps, its carried per-episode state, the relay scope status and pays,
            # the unique terms' marks, the Indigo Plateau mark
            ring = cfg.rewind_ring
            self._rw_tiles = np.zeros((n, ring, R.TILE_BYTES), dtype=np.uint8)
            self._rw_maps = np.zeros((n, ring, R.MAP_COUNT + 1), dtype=bool)
            self._rw_carry = {k: np.zeros((n, ring) + v.shape[1:], dtype=v.dtype)
                              for k, v in self.reward.carry_capture(np.zeros(1, dtype=np.int64), cfg.carry_paid).items()}
            self._rw_rel_status = np.zeros((n, ring, ns), dtype=np.int8)
            self._rw_rel_paid_act = np.zeros((n, ring, ns), dtype=bool)
            self._rw_rel_paid_clr = np.zeros((n, ring, ns), dtype=bool)
            if self._lvl_indigo is not None:
                self._rw_lvl_indigo = np.zeros((n, ring), dtype=bool)
            if self._uq_on:
                self._rw_uq = {k: np.zeros((n, ring) + getattr(self, k).shape[1:], dtype=bool) for k in UNIQUE_MARKS}

    def _pixel_view(self) -> np.ndarray:
        """The vec env's mapped pixel view, checked against ``pixel_shape``."""
        env = self.env
        view = env.pixels
        if tuple(env.pixel_shape) != self.pixel_shape or env.pixel_format != self.pixel_format:
            raise RuntimeError(f"PokeVecEnv pixels {env.pixel_format} {tuple(env.pixel_shape)}, expected "
                               f"{self.pixel_format} {self.pixel_shape}")
        if not isinstance(view, np.ndarray) or view.dtype != np.uint8 or \
                tuple(view.shape) != (self.num_envs,) + self.pixel_shape:
            raise RuntimeError(f"PokeVecEnv pixel view {getattr(view, 'shape', None)}, expected "
                               f"{(self.num_envs,) + self.pixel_shape} uint8")
        return view

    # ---- API -------------------------------------------------------------------------------------------

    def reset(self):
        """Resets every env; returns (obs, mask)."""
        obs, _ = self.env.reset()
        self._frames[:] = self.cfg.frameskip
        obs = _state(obs)
        self._obs = obs
        all_ids = np.arange(self.num_envs)
        self.reward.reset(obs)
        self._level_reset(obs, all_ids)
        self._ko_reset(obs, all_ids)
        self._start_stats(all_ids)
        self._new_lives(all_ids)
        self.life_events |= EV.observe(obs)
        if self._uq_bag_on:
            self._unique_bag(obs)
        return self._observe(obs)

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray) -> None:
        """Posts one step of every env (returns at once; ``step_wait`` collects it)."""
        if self._pending is not None:
            raise RuntimeError("step_async twice without step_wait")
        t0 = time.perf_counter()
        a = np.asarray(actions, dtype=np.int64)
        kw = {}
        self.last_hold = self.last_n = None
        if self.cfg.dpad_hold and self._obs is not None:
            hold, n = dpad_hold_frames(self._obs, a, self.cfg.hold_frames, self.cfg.dpad_hold, self.cfg.frameskip)
            self.last_hold, self.last_n = hold, n
            kw = {"hold_frames": hold, "n_frames": n}
        self.env.step_async(a, **kw)
        self._pending = True
        self.timing["post"] += time.perf_counter() - t0

    def step_wait(self):
        """Waits for the posted step and returns ``(obs, mask, reward, terminated, truncated, info)``."""
        self.step_wait_begin()
        return self.step_wait_end()

    def step_wait_begin(self, async_reset: bool = False) -> None:
        """First half of ``step_wait``: waits for the workers, computes the reward and the episode ends,
        and starts the resets of the ended envs — in a helper thread with ``async_reset`` (the reset of a
        bedroom start with a new seed costs ~8.7 ms of worker time; ``reset_ready()`` tells when it is
        done), else at once. ``step_wait_end`` returns the results. Between the two calls nothing else may
        use this env's PokeVecEnv."""
        if self._pending is None:
            raise RuntimeError("step_wait without step_async")
        self._pending = None
        tm = self.timing
        t0 = time.perf_counter()
        fallback = self.in_fallback.copy()  # the masks the actions were chosen under (before any reset)
        obs, _, _, sim_terminal, info = self.env.step_wait()
        t1 = time.perf_counter()
        tm["wait"] += t1 - t0
        frames = np.asarray(info["frames_run"], dtype=np.int64)
        obs = _state(obs)
        self._obs = obs
        auto = info["autoreset"]
        if auto.any():  # PokeVecEnv reset these itself (not expected: ends are reset at once below)
            ids = np.flatnonzero(auto)
            self.reward.reset(obs[ids], ids)
            self._level_reset(obs[ids], ids)
            self._ko_reset(obs[ids], ids)
            self._start_stats(ids)
            self._new_lives(ids)
        t2 = time.perf_counter()
        reward, terms, progress, champion = self.reward.step(obs)
        progress |= self._level_step(obs)
        progress |= self._ko_step(obs)
        if self._rwb_badge:  # (the rewind budget follows the most badges the life has held)
            self._rewind_badges(obs)
        t3 = time.perf_counter()
        tm["reward"] += t3 - t2
        tile_bonus = self.reward.tile_bonus
        ev_now = EV.observe(obs)  # (this record's events, unmasked: the relay's scope rules read them)
        self.life_events |= ev_now  # (auto-reset rows were cleared above: their new life's first records)
        if auto.any():
            reward[auto] = 0
            tile_bonus = np.where(auto, 0, tile_bonus)
            for v in terms.values():
                v[auto] = 0
            progress[auto] = True
            champion[auto] = False
        if self._uq_on:  # (unique purchases and item uses, before ep_return)
            self._unique_step(obs, reward, auto)
        if self._lvl_indigo is not None:  # (the levels term's weight after Indigo Plateau)
            self._level_indigo(obs, reward, terms, auto)
        self.ep_len += 1
        self.since_progress = np.where(progress, 0, self.since_progress + 1)
        mkey = obs["map_group"].astype(np.int64) * 256 + obs["map_num"]
        tr = time.perf_counter()
        self._relay_step(obs, ev_now, mkey, reward, champion, auto)  # (activation / clear pays before ep_return)
        t_rel = time.perf_counter() - tr
        self.ep_return += reward
        terms_stack = np.stack([terms[k] for k in TERMS], axis=1)  # (also info["terms_stack"]: the collector's copy)
        self.ep_terms += terms_stack
        self.ep_tile_bonus += tile_bonus

        terminated = champion | self._rel_exit  # (a relay area exit TERMINATES the life; _rel_exit excludes the
        # Champion's step: that is a success)
        truncated = ~terminated & (sim_terminal | (self.ep_len >= self.length_cap) |
                                   (self.since_progress >= self.stag_cap))
        done = terminated | truncated
        capturable = progress & ~auto  # (the rows the rewind may capture)
        t_cap = t_plan = 0.0
        capture = None
        if self.rewind is not None:  # (before any reset: the rows' records/states are this step's)
            tc = time.perf_counter()
            capture = self._rewind_step(obs, capturable, done)
            t_cap = time.perf_counter() - tc
        episodes: List[Dict] = []
        ids = None
        reset = None
        t4 = time.perf_counter()
        if done.any():
            ids = np.flatnonzero(done)
            args = (ids,)
            if self.rewind is None:
                episodes = self._end_stats(ids, obs, terminated, sim_terminal)
            else:
                tp = time.perf_counter()
                plan = self._rewind_plan(ids, truncated, sim_terminal)
                t_plan = time.perf_counter() - tp
                ends = ids if plan is None else np.delete(ids, plan[0])
                if ends.size:
                    episodes = self._end_stats(ends, obs, terminated, sim_terminal)
                self._new_lives(ends)
                if plan is not None:
                    args = (ids, plan)
            # (the step's records of these rows are read: the reset may now overwrite them)
            if async_reset:
                if self._executor is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pokerl-reset")
                reset = self._executor.submit(self._reset_rows, *args)
            else:
                reset = self._reset_rows(*args)
        self._mid = (obs, ids, reset, reward, terms, terminated, truncated, episodes, sim_terminal, fallback, frames,
                     capture, terms_stack)
        if self.rewind is not None:
            tm["rewind"] += t_cap + t_plan
        tm["relay"] += t_rel
        tm["reset"] += time.perf_counter() - t4 - t_plan
        tm["other"] += (t2 - t1) + (t4 - t3) - t_cap - t_rel

    def reset_ready(self) -> bool:
        """After ``step_wait_begin``: True once ``step_wait_end`` will not block on a reset."""
        reset = self._mid[2]
        return reset is None or not hasattr(reset, "done") or reset.done()

    def reset_future(self):
        """The pending asynchronous reset (a Future) or None."""
        reset = self._mid[2] if self._mid is not None else None
        return reset if hasattr(reset, "done") else None

    def step_wait_end(self):
        """Second half of ``step_wait`` (see ``step_wait_begin``): finishes the resets (the returned rows of
        ended envs are the first observations of their next lives), observes and masks; returns
        ``(obs, mask, reward, terminated, truncated, info)``."""
        if self._mid is None:
            raise RuntimeError("step_wait_end without step_wait_begin")
        (obs, ids, reset, reward, terms, terminated, truncated, episodes, sim_terminal, fallback, frames, capture,
         terms_stack), self._mid = self._mid, None
        tm = self.timing
        t4 = time.perf_counter()
        rewound = None
        if ids is not None:
            if hasattr(reset, "result"):
                reset = reset.result()
            reset_obs = _state(reset[0])
            # copy=False: the step's records are a view of the shared memory the reset just rewrote
            rec = obs.view(np.uint8).reshape(self.num_envs, -1)
            rb = reset_obs.view(np.uint8).reshape(ids.size, -1)
            if not np.array_equal(rec[ids], rb):
                rec[ids] = rb
            is_rw = reset[1].get("rewind") if isinstance(reset[1], dict) else None
            if is_rw is None:
                self.reward.reset(reset_obs, ids)
                self._level_reset(reset_obs, ids)
                self._ko_reset(reset_obs, ids)
                self.length_cap[ids] = self.cfg.max_episode_steps
                self.stag_cap[ids] = self._draw_stagnation(ids.size)
                self._start_stats(ids)
            else:
                rewound = self._rewind_end(ids, reset_obs, reset[1], is_rw)
            # new lives (cleared by _start_stats) from their first records, rewound lives from the restored ones
            self.life_events[ids] |= EV.observe(reset_obs)
        waiting = self._waiting.copy()
        self._frames[:] = frames
        if ids is not None:
            self._frames[ids] = self.cfg.frameskip  # a reset observation counts as one plain step
        t5 = time.perf_counter()
        tm["reset"] += t5 - t4
        if self._uq_bag_on:  # (the records the next step's item uses are found against: new lives and restores too)
            self._unique_bag(obs)
        enc, mask = self._observe(obs)
        t6 = time.perf_counter()
        tm["observe"] += t6 - t5
        events, self._fallback_events = self._fallback_events, []
        d = self._drain
        d["events"].extend(events)
        d["steps"] += self.num_envs
        d["fallback_steps"] += int(fallback.sum())
        fd = self._frames_drain
        fd["steps"] += self.num_envs
        fd["frames"] += int(frames.sum())
        hist = np.bincount(frames)
        if hist.size > fd["frames_hist"].size:
            hist[:fd["frames_hist"].size] += fd["frames_hist"]
            fd["frames_hist"] = hist
        else:
            fd["frames_hist"][:hist.size] += hist
        out_info = {"terms": terms, "terms_stack": terms_stack, "episodes": episodes, "waiting": waiting,
                    "sim_terminal": sim_terminal,
                    "fallback": fallback, "fallback_active": self.in_fallback.copy(), "fallback_events": events,
                    "frames": frames}
        if self.rewind is not None:
            out_info["capture"] = capture
            out_info["rewind"] = rewound if rewound is not None else self._no_rows()
        tm["other"] += time.perf_counter() - t6
        return enc, mask, reward, terminated, truncated, out_info

    def pop_timing(self) -> Dict[str, float]:
        """Seconds per phase since the last call (``TIMING_KEYS``); then clears them."""
        t, self.timing = self.timing, dict.fromkeys(TIMING_KEYS, 0.0)
        return t

    def drain_fallback(self) -> Dict:
        """Mask fallback totals since the last call: ``events`` (activation dicts), ``steps`` (env-steps),
        ``fallback_steps`` (of them chosen under the fallback mask), ``active`` (envs in fallback now)."""
        d, self._drain = self._drain, {"events": [], "steps": 0, "fallback_steps": 0}
        d["active"] = int(self.in_fallback.sum())
        return d

    @staticmethod
    def _new_rw_drain() -> Dict:
        return {"captures": 0, "rewinds": 0, "progress": 0, "progress_after": 0, "snapshot_calls": 0,
                "restore_calls": 0, "t_snapshot": 0.0, "t_restore": 0.0}

    def drain_rewind(self) -> Dict:
        """Rewind counters since the last call (``captures``, ``rewinds``, ``progress`` events and of them
        ``progress_after`` a life's first rewind, snapshot / restore calls and their seconds), plus the rings'
        captures held now (``held``) and their bytes (``mem_bytes``); then resets them."""
        d, self._rw_drain = self._rw_drain, self._new_rw_drain()
        ring = self.rewind
        d["held"] = int(ring.count.sum()) if ring is not None else 0
        d["mem_bytes"] = ring.memory_bytes() if ring is not None else 0
        return d

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self.env.close()

    # ---- life starts -----------------------------------------------------------------------------------

    def _reset_rows(self, ids: np.ndarray, plan=None):
        """Starts new lives in envs ``ids`` (sorted); returns ``(records, info)`` with one row per id, like
        ``PokeVecEnv.reset``. ``plan`` (``_rewind_plan``): some rows rewind instead (``_rewind_rows``). May run in
        the reset helper thread."""
        if plan is not None:
            return self._rewind_rows(ids, plan)
        rec, info = self.env.reset(env_ids=ids)
        return _state(rec), info

    # ---- rewind (module docstring "Rewind") ---------------------------------------------------------------

    @staticmethod
    def _no_rows() -> Dict[str, np.ndarray]:
        z = _EMPTY_ROWS
        return {"rows": z, "slots": z, "serials": z}

    def _new_lives(self, ids) -> None:
        """Lives starting in envs ``ids``: no rewind used, empty rings, the rewind budget (with ``rewind_per_badge``:
        0 badges held and the life's uniform draw ``rw_u``, ``rewind_budget_of``; else drawn uniformly from
        ``rewind_max_min`` .. ``rewind_max``)."""
        self.rewinds_used[ids] = 0
        if self._rwb_badge:  # one uniform draw per life, scaled to the badges it has held (none yet)
            self.rw_badges[ids] = 0
            self.rw_u[ids] = self._rwb_rng.random(np.asarray(ids).size)
            self.rewind_budget[ids] = self.rewind_budget_of(self.rw_u[ids], self.rw_badges[ids])
        else:
            k = np.asarray(ids).size
            self.rewind_budget[ids] = self._rwb_rng.integers(self.cfg.rewind_max_min, self.cfg.rewind_max + 1, size=k)
        self.rw_progress[ids] = 0
        if self.rewind is not None:
            self.rewind.clear(ids)

    def rewind_budget_of(self, u: np.ndarray, badges: np.ndarray) -> np.ndarray:
        """The budgets of lives with uniform draws ``u`` in [0, 1) that have held ``badges`` badges at most: m +
        floor(u (rewind_max + K b - m + 1)) with ``rewind_max_min`` m and ``rewind_per_badge`` K (uniform over m ..
        rewind_max + K b, non-decreasing in b for one u)."""
        cfg = self.cfg
        top = cfg.rewind_max + cfg.rewind_per_badge * np.asarray(badges, dtype=np.int64)
        m = cfg.rewind_max_min
        return m + np.floor(np.asarray(u, dtype=np.float64) * (top - m + 1)).astype(np.int64)

    def _rewind_badges(self, obs) -> None:
        """Latches the most badges of every in-game record's life and raises the budgets of the rows whose count rose
        (before this step's rewind decisions)."""
        b = popcount8(obs["badges"]).astype(np.int64)
        up = ((obs["valid"] & VALID_INGAME) != 0) & (b > self.rw_badges)
        if up.any():
            self.rw_badges[up] = b[up]
            self.rewind_budget[up] = self.rewind_budget_of(self.rw_u[up], self.rw_badges[up])

    def _rewind_step(self, obs, progress: np.ndarray, done: np.ndarray) -> Dict[str, np.ndarray]:
        """Progress counters and this step's captures (before any reset: the rows' states are the ones whose records
        are ``obs``): rows with a progress event, not ending, whose newest capture is at least ``rewind_min_gap``
        steps old or that have none; at most ``rewind.MAX_CAPTURES_PER_STEP``, oldest newest-capture first. Returns
        ``info["capture"]``."""
        if not progress.any():  # (most steps)
            return self._no_rows()
        d = self._rw_drain
        after = progress & (self.rewinds_used > 0)
        d["progress"] += int(progress.sum())
        if after.any():
            d["progress_after"] += int(after.sum())
            self.rw_progress += after
        ring = self.rewind
        rows = np.flatnonzero(progress & ~done)
        if rows.size == 0:
            return self._no_rows()
        has = ring.count[rows] > 0
        age = self.ep_len[rows] - ring.last_step[rows]
        ok = ~has | (age >= self.cfg.rewind_min_gap)
        if not ok.all():
            rows, has, age = rows[ok], has[ok], age[ok]
            if rows.size == 0:
                return self._no_rows()
        cap = max(1, int(RW.MAX_CAPTURES_PER_STEP))
        if rows.size > cap:
            key = np.where(has, age, np.iinfo(np.int64).max)
            rows = np.sort(rows[np.argsort(-key, kind="stable")[:cap]])
        # the ring hands out the slots, the worker writes the snapshots into them
        slots, serials = ring.push(rows, self.ep_len[rows])
        t0 = time.perf_counter()
        self.env.snapshot_to_slots(rows, self._rw_base + rows * ring.size + slots, fast=True)
        d["t_snapshot"] += time.perf_counter() - t0
        # (after this step's reward.step, relay transitions and unique pays: the state up to the captured record)
        self._rw_tiles[rows, slots] = self.reward.tile_bm[rows]
        self._rw_maps[rows, slots] = self.reward.map_seen[rows]
        for k, v in self.reward.carry_capture(rows, self.cfg.carry_paid).items():
            self._rw_carry[k][rows, slots] = v
        self._rw_rel_status[rows, slots] = self.relay_status[rows]
        self._rw_rel_paid_act[rows, slots] = self._rel_paid_act[rows]
        self._rw_rel_paid_clr[rows, slots] = self._rel_paid_clr[rows]
        if self._uq_on:
            for k, v in self._rw_uq.items():
                v[rows, slots] = getattr(self, k)[rows]
        if self._rw_lvl_indigo is not None:
            self._rw_lvl_indigo[rows, slots] = self.lvl_indigo[rows]
        d["snapshot_calls"] += 1
        d["captures"] += int(rows.size)
        return {"rows": rows, "slots": slots, "serials": serials}

    def _rewind_plan(self, ids: np.ndarray, truncated: np.ndarray, sim_terminal: np.ndarray):
        """The ending envs ``ids`` that rewind: ended by stagnation only (not the length cap, the Champion, a relay
        exit or a simulator terminal), a capture held, ``rewinds_used < rewind_budget``. Per such row a capture is
        chosen uniformly among the newest ``rewind_depth`` (the newer ones are discarded) and a fresh RNG stream seed
        is drawn; ``rewinds_used`` += 1. Returns ``(positions in ids, captures, seeds)`` or None."""
        cfg, ring = self.cfg, self.rewind
        stag = truncated[ids] & ~np.asarray(sim_terminal, dtype=bool)[ids] & \
            (self.ep_len[ids] < self.length_cap[ids]) & (self.since_progress[ids] >= self.stag_cap[ids])
        ok = stag & (ring.count[ids] > 0) & (self.rewinds_used[ids] < self.rewind_budget[ids]) & ~self._rel_exit[ids]
        if not ok.any():
            return None
        pos = np.flatnonzero(ok)
        caps = []
        seeds = np.zeros(pos.size, dtype=np.uint64)
        for j, r in enumerate(ids[pos].tolist()):
            caps.append(ring.choose(r, cfg.rewind_depth, self._rw_rng, now=int(self.ep_len[r])))
            seeds[j] = self._rw_rng.integers(0, 1 << 64, dtype=np.uint64)
        self.rewinds_used[ids[pos]] += 1
        self._rw_drain["rewinds"] += int(pos.size)
        return pos, caps, seeds

    def _rewind_rows(self, ids: np.ndarray, plan):
        """``_reset_rows`` with rewinds: the rows ``plan[0]`` (positions in ``ids``) restore their chosen captures
        with fresh RNG streams (one ``restore_from_slots``), the others start new lives. ``info``: ``rewind`` (bool
        per row), ``rewind_slots`` / ``rewind_serials`` (per rewound row, in order). May run in the reset helper
        thread."""
        pos, caps, seeds = plan
        k = ids.size
        slots = np.fromiter((c.slot for c in caps), dtype=np.int64, count=len(caps))
        t0 = time.perf_counter()
        # the fresh RNG stream goes straight into the capture's slot (header bytes 64-71; the state hash does not
        # cover it), then one restore op reads the slots in place
        rows = ids[pos]
        wslots = self._rw_base + rows * self.rewind.size + slots
        streams = np.fromiter((rng_stream_seed(int(s)) for s in seeds), dtype="<u8", count=len(caps))
        self.env.slot_bytes[wslots, HEADER_RNG_STREAM] = streams.view(np.uint8).reshape(-1, 8)
        rec_r = _state(self.env.restore_from_slots(rows, wslots))
        d = self._rw_drain
        d["t_restore"] += time.perf_counter() - t0
        d["restore_calls"] += 1
        is_rw = np.zeros(k, dtype=bool)
        is_rw[pos] = True
        info = {"rewind": is_rw, "rewind_slots": slots,
                "rewind_serials": np.fromiter((c.serial for c in caps), dtype=np.int64, count=len(caps))}
        new = np.flatnonzero(~is_rw)
        if new.size == 0:
            return rec_r, info
        rec_n, _ = self._reset_rows(ids[new])
        rec = np.empty(k, dtype=rec_n.dtype)
        rec[new] = rec_n
        rec[pos] = rec_r
        return rec, info

    def _rewind_end(self, ids: np.ndarray, reset_obs, info: Dict, is_rw: np.ndarray) -> Dict[str, np.ndarray]:
        """``step_wait_end`` for a reset batch with rewinds: the new lives start as usual; the rewound rows keep
        their life (``ep_len``, ``ep_return``, ``ep_terms``, the length cap) with ``since_progress`` 0, a new
        stagnation window and the mask-fallback counters restarted, their reward state re-initialised from the
        restored records as at an episode start (the counters of the segment that ends go to ``life_flags`` /
        ``life_maps``) and then the capture's carried state put back. Returns ``info["rewind"]``."""
        new = np.flatnonzero(~is_rw)
        rw_ids, new_ids = ids[is_rw], ids[new]
        self.life_flags[rw_ids] += self.reward.flags_earned[rw_ids]
        self.life_maps[rw_ids] += self.reward.maps_visited[rw_ids]
        # every row from its first record: the new lives' starts and the rewound rows' restored records (the rows
        # of the observation the policy sees next)
        self.reward.reset(reset_obs, ids)
        self._level_reset(reset_obs, ids)
        self._ko_reset(reset_obs, ids)  # (the rewound rows too: no KO event on the restored record)
        self.life_maps[rw_ids] -= self.reward.maps_visited[rw_ids]  # the restored map is not a new visit
        if rw_ids.size:  # (rw_ids is ids[pos] in order, as the slots)
            slots = info["rewind_slots"]
            self.reward.tile_bm[rw_ids] = self._rw_tiles[rw_ids, slots]  # the tiles and maps visited up to the capture
            self.reward.map_seen[rw_ids] = self._rw_maps[rw_ids, slots]
            self.reward.carry_restore(rw_ids, {k: v[rw_ids, slots] for k, v in self._rw_carry.items()})
            self.relay_status[rw_ids] = self._rw_rel_status[rw_ids, slots]
            self._rel_paid_act[rw_ids] = self._rw_rel_paid_act[rw_ids, slots]  # (what the capture had paid; a repeat
            self._rel_paid_clr[rw_ids] = self._rw_rel_paid_clr[rw_ids, slots]  # of a later activation / clear pays)
            if self._uq_on:
                for k, v in self._rw_uq.items():
                    getattr(self, k)[rw_ids] = v[rw_ids, slots]
            if self._rw_lvl_indigo is not None:
                self.lvl_indigo[rw_ids] = self._rw_lvl_indigo[rw_ids, slots]
        if new.size:
            self.length_cap[new_ids] = self.cfg.max_episode_steps  # (a rewound row keeps its life's cap)
        self.stag_cap[ids] = self._draw_stagnation(ids.size)
        if new.size:
            self._start_stats(new_ids)
        self.since_progress[rw_ids] = 0
        self.nonwait_frames[rw_ids] = 0
        self.in_fallback[rw_ids] = False
        return {"rows": rw_ids, "slots": info["rewind_slots"], "serials": info["rewind_serials"]}

    # ---- unique purchases and item uses (module docstring "Unique purchases and item uses") ----------------

    def _unique_alloc(self, cfg: "EnvConfig", n: int) -> None:
        """The unique terms' state of ``n`` envs: the weights, per life the mart maps bought on and the item ids used
        on the field / in battle, the pay so far ("term_unique_*"), the bag (items + berries pockets: ids, quantities)
        of each env's previous record (set by ``_unique_bag``), the rewind ring's copies (allocated with the ring)."""
        self._uq_w = np.array([getattr(cfg, k) for k in UNIQUE_KEYS], dtype=np.float32)
        self._uq_on = bool((self._uq_w > 0).any())
        self._uq_bag_on = bool((self._uq_w[1:] > 0).any())
        self.uq_mart = np.zeros((n, R.MAP_COUNT + 1), dtype=bool)
        self.uq_consumable = np.zeros((n, UNIQUE_ITEMS), dtype=bool)
        self.uq_battle_item = np.zeros((n, UNIQUE_ITEMS), dtype=bool)
        self.uq_bonus = np.zeros((n, len(UNIQUE_KEYS)), dtype=np.float32)
        self._uq_ids = self._uq_qty = None
        self._rw_uq: Optional[Dict[str, np.ndarray]] = None

    def _unique_step(self, obs, reward: np.ndarray, auto: np.ndarray) -> None:
        """This step's unique purchase and item-use pays, into ``reward`` (in place) and ``uq_bonus``: a ``mart_buy``
        record on a mart map not in the life's ``uq_mart``; a ``use_consumable`` (``use_battle_item``) record whose bag
        holds fewer of an item id than the env's previous record (``_uq_ids`` / ``_uq_qty``), for each such id not in
        ``uq_consumable`` (``uq_battle_item``). In-game rows only, not PokeVecEnv autoreset rows."""
        bits = obs["skill_events_priv"]
        live = ((obs["valid"] & VALID_INGAME) != 0) & ~auto
        w = self._uq_w
        if w[0] > 0:
            rows = np.flatnonzero(live & (((bits >> UNIQUE_BITS[0]) & 1) != 0))
            if rows.size:
                m = R.MAP_INDEX[obs["map_group"][rows].astype(np.int64) * 256 + obs["map_num"][rows]]
                new = (m > 0) & ~self.uq_mart[rows, m]
                rows, m = rows[new], m[new]
                self.uq_mart[rows, m] = True
                reward[rows] += w[0]
                self.uq_bonus[rows, 0] += w[0]
        for j, marks in ((1, self.uq_consumable), (2, self.uq_battle_item)):
            if w[j] <= 0:
                continue
            rows = np.flatnonzero(live & (((bits >> UNIQUE_BITS[j]) & 1) != 0))
            if rows.size == 0:
                continue
            ids, qty = self._bag_arrays(obs, rows)
            used = _bag_totals(ids, qty) < _bag_totals(self._uq_ids[rows], self._uq_qty[rows])
            used[:, 0] = False
            new = used & ~marks[rows]
            marks[rows] |= used
            pay = w[j] * new.sum(axis=1).astype(np.float32)
            reward[rows] += pay
            self.uq_bonus[rows, j] += pay

    def _level_indigo(self, obs, reward: np.ndarray, terms: Dict[str, np.ndarray], auto: np.ndarray) -> None:
        """In-game rows on an ``INDIGO_MAPS`` map mark their life as arrived (``lvl_indigo``); the levels term of the
        arrived lives pays at ``level_reward_indigo`` instead of the levels weight (the term and ``reward`` in place:
        + term x (level_reward_indigo / level_reward - 1))."""
        lut, extra = self._lvl_indigo
        ingame = ((obs["valid"] & VALID_INGAME) != 0) & ~auto
        self.lvl_indigo |= ingame & lut[R.MAP_INDEX[obs["map_group"].astype(np.int64) * 256 + obs["map_num"]]]
        lv = terms["levels"]
        rows = np.flatnonzero(self.lvl_indigo & (lv != 0) & ingame)
        if rows.size:
            add = lv[rows] * extra
            lv[rows] += add
            reward[rows] += add

    @staticmethod
    def _bag_arrays(obs, rows=None):
        """int64 ids and quantities [k, 42 + 43] of the records' items and berries pockets (all rows, or ``rows``)."""
        it, be = (obs["items"], obs["berries"]) if rows is None else (obs["items"][rows], obs["berries"][rows])
        return (np.concatenate([it["id"], be["id"]], axis=1).astype(np.int64),
                np.concatenate([it["qty"], be["qty"]], axis=1).astype(np.int64))

    def _unique_bag(self, obs) -> None:
        """The bag of the records ``obs`` (the ones the next step's records are compared with) into ``_uq_ids`` /
        ``_uq_qty`` (copies: the records are a view of memory the next step rewrites)."""
        self._uq_ids, self._uq_qty = self._bag_arrays(obs)

    # ---- relay scopes (module docstring "Relay scopes") -------------------------------------------------------

    @staticmethod
    def _new_relay_drain() -> Dict:
        return {"activations": 0, "exits": 0, "masked": 0}

    def drain_relay(self) -> Dict:
        """Relay counters since the last call: ``activations`` (scopes activated), ``exits`` (lives ended by an area
        exit), ``masked`` (records with a press taken out of the mask); then resets them."""
        d, self._rel_drain = self._rel_drain, self._new_relay_drain()
        return d

    def _relay_step(self, obs, ev: np.ndarray, mkey: np.ndarray, reward: np.ndarray, champion: np.ndarray,
                    auto: np.ndarray) -> None:
        """This step's scope transitions (``ScopeSet.step`` on the status at the step's start, in place, over ``ev``
        = this record's events, ``events.observe``: not the latched ``life_events``, which a rewind does not roll
        back; the predicates are flag / item / move events, which a restored state carries), their pays (added to
        ``reward`` in place) and the area exits (``_rel_exit``). Only the rows that can change are evaluated: in-game
        rows with an ACTIVE scope or on a map where a scope may activate (``ScopeSet.may_activate``); on most steps
        there is none and nothing else runs."""
        sc = self._scopes
        live = ((obs["valid"] & VALID_INGAME) != 0) & ~auto
        midx = R.MAP_INDEX[mkey]
        rows = np.flatnonzero(live & ((self.relay_status == RSC.ACTIVE).any(axis=1) | sc.may_activate[midx]))
        self._rel_exit = self._rel_false
        if rows.size == 0:  # (most steps: every row between scopes, away from the homes)
            return
        st = self.relay_status[rows]
        x, y = (obs["x"][rows], obs["y"][rows]) if sc.has_porch else (None, None)
        _, act, clr, exited = sc.step(st, midx[rows], ev[rows], x, y, ingame=live[rows], inplace=True)
        self.relay_status[rows] = st
        if (act >= 0).any():
            self._rel_drain["activations"] += int((act >= 0).sum())
            if self._rel_act_reward != 0:  # (in the step's reward and the return; once per scope and life)
                ar, sa = rows[act >= 0], act[act >= 0]
                fresh = ~self._rel_paid_act[ar, sa]
                ar, sa = ar[fresh], sa[fresh]
                self._rel_paid_act[ar, sa] = True
                reward[ar] += self._rel_act_reward
                self.relay_bonus[ar] += self._rel_act_reward
        if self._rel_clr_reward != 0 and (clr >= 0).any():  # (the same for a clear)
            cr, sc_ = rows[clr >= 0], clr[clr >= 0]
            fresh = ~self._rel_paid_clr[cr, sc_]
            cr, sc_ = cr[fresh], sc_[fresh]
            self._rel_paid_clr[cr, sc_] = True
            reward[cr] += self._rel_clr_reward
            self.relay_bonus[cr] += self._rel_clr_reward
        ex = exited & ~champion[rows]  # (what the mask did not stop ends the life; the Champion's step is a success)
        if ex.any():
            self._rel_exit = np.zeros(self.num_envs, dtype=bool)
            self._rel_exit[rows[ex]] = True
            self._rel_drain["exits"] += int(ex.sum())

    def _relay_exit_mask(self, obs, mask: np.ndarray) -> None:
        """Takes the presses that would walk out of the ACTIVE scope's area out of ``mask`` (in place), on
        free-overworld records (elsewhere the D-pad moves a cursor). Keyed on the tile, so a life outside its area
        has nothing masked."""
        act = RSC.active_scope(self.relay_status)
        rows = np.flatnonzero((act >= 0) & (obs["input_kind"] == _INPUT_OVERWORLD_FREE))
        if rows.size == 0:
            return
        mkey = obs["map_group"][rows].astype(np.int64) * 256 + obs["map_num"][rows]
        bits = self._exit_table.bits(act[rows], R.MAP_INDEX[mkey], obs["x"][rows], obs["y"][rows])
        hit = np.flatnonzero(bits)
        if hit.size == 0:
            return
        rows = rows[hit]
        mask[rows] &= RX.ALLOWED[bits[hit]]
        mask[rows, 0] |= ~mask[rows].any(axis=1)  # (every row allows at least one action: mask.py)
        self._rel_drain["masked"] += int(rows.size)

    # ---- internals -------------------------------------------------------------------------------------

    def drain_frames(self) -> Dict:
        """Game frames since the last call: ``steps`` (env-steps), ``frames`` (frames they ran),
        ``frames_hist`` (int64 bincount of the frames per env-step); then resets them."""
        d, self._frames_drain = self._frames_drain, self._new_frames_drain()
        return d

    @staticmethod
    def _new_frames_drain() -> Dict:
        return {"steps": 0, "frames": 0, "frames_hist": np.zeros(0, dtype=np.int64)}

    @staticmethod
    def _level_total(obs) -> np.ndarray:
        """The sum of the levels of every non-egg party mon per record."""
        p = obs["party"]
        sp = p["species"]
        mon = (_PARTY_SLOTS[:, :p.shape[1]] < obs["party_count"][:, None]) & (sp != 0) & (sp != R.T.SPECIES_EGG) \
            & (p["is_egg"] == 0)
        return np.sum(p["level"] * mon, axis=1, dtype=np.int64)  # (level x bool: 0 where no mon)

    def _level_reset(self, obs, ids: np.ndarray) -> None:
        """The level maximum of rows ``ids`` from their first records ``obs`` (one per id)."""
        if len(ids):
            self.level_best[ids] = self._level_total(obs)

    def _level_step(self, obs) -> np.ndarray:
        """The rows whose total party level rose above the maximum (in the game only): a progress event."""
        tot = self._level_total(obs)
        up = ((obs["valid"] & R.VALID_INGAME) != 0) & (tot > self.level_best)
        self.level_best[up] = tot[up]
        return up

    @staticmethod
    def _ko_now(obs) -> Tuple[np.ndarray, np.ndarray]:
        """(hp uint16 [k, 4], party index uint8 [k, 4]) of records ``obs``: per battler slot the hp_priv /
        party_index_priv of an opponent-side battler with its healthbox shown (``is_active``) in an in-game record in
        battle with the battlers block filled (VALID_BATTLE_MONS), else 0 / KO_NO_SLOT."""
        b = obs["battlers"]
        on = ((obs["valid"] & VALID_INGAME) != 0) & (obs["in_battle"] != 0) & ((obs["valid"] & VALID_BATTLE_MONS) != 0)
        live = on[:, None] & (b["is_player_side"] == 0) & (b["is_active"] != 0)
        return (np.where(live, b["hp_priv"], 0).astype(np.uint16),
                np.where(live, b["party_index_priv"], KO_NO_SLOT).astype(np.uint8))

    def _ko_reset(self, obs, ids: np.ndarray) -> None:
        """The opponent state of rows ``ids`` from their first records ``obs`` (one per id; a new life, a rewind
        restore, an autoreset): the first record is never a KO."""
        if len(ids):
            self.ko_hp[ids], self.ko_idx[ids] = self._ko_now(obs)

    def _ko_step(self, obs) -> np.ndarray:
        """bool [N], the rows on which an opponent battler was knocked out since the previous in-game record, in any
        battle (wild or trainer, singles or doubles): a progress event. Per battler slot k whose opponent was active
        with hp_priv > 0 in that record (``ko_hp``):
        (a) in battle (VALID_BATTLE_MONS), slot k is still the opponent side and its hp_priv is 0 (the slot's
            healthbox may be hidden already: a faint hides it);
        (b) faint and replace within one step (the trainer's next mon already in slot k, hp > 0, which (a) misses):
            slot k's party_index_priv differs from the remembered one and the remembered mon's party copy
            (``enemy_party_priv[idx].hp``) is 0; a switch (Roar, the trainer's switch, Baton Pass) keeps the old
            mon's hp > 0 in its party copy, so it is not counted.
        Not events: a fled / caught / ended battle (the opponent's hp stays > 0, or the next record is outside the
        battle: the state clears), the player's own faints (own side never remembered), rows outside the game
        (the state holds)."""
        b = obs["battlers"]
        ingame = (obs["valid"] & VALID_INGAME) != 0
        mons = ingame & (obs["in_battle"] != 0) & ((obs["valid"] & VALID_BATTLE_MONS) != 0)
        was = (self.ko_hp > 0) & mons[:, None] & (b["is_player_side"] == 0)
        ko = np.zeros(obs.shape[0], dtype=bool)
        if was.any():
            hp = b["hp_priv"]
            down = was & (hp == 0)
            old = self.ko_idx
            moved = was & (hp > 0) & (b["party_index_priv"] != old) & (old < KO_PARTY)
            if moved.any():
                ep = obs["enemy_party_priv"]["hp"]  # [N, 6]
                old_hp = np.take_along_axis(ep, np.minimum(old, KO_PARTY - 1).astype(np.intp), axis=1)
                down |= moved & (old_hp == 0)
            ko = down.any(axis=1)
        if ingame.any():  # (rows outside the game hold the state)
            hp_now, idx_now = self._ko_now(obs)
            self.ko_hp[ingame] = hp_now[ingame]
            self.ko_idx[ingame] = idx_now[ingame]
        return ko

    def _draw_stagnation(self, k: int) -> np.ndarray:
        S = self.cfg.stagnation_steps
        if S <= 0:  # stagnation end off
            return np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
        return self.rng.integers(max(1, (3 * S) // 4), (5 * S) // 4 + 1, size=k).astype(np.int64)

    def _observe(self, obs):
        """The observation (the frames and the trail plane) and the mask of the returned records ``obs``."""
        w = self._waiting = obs["waiting_for_input"] != 0
        mask = self._mask_fn(obs)
        if self.cfg.fallback_after > 0:  # frame-based: fallback_after plain steps' worth of frames
            self.in_fallback &= ~w
            self.nonwait_frames = np.where(w, 0, self.nonwait_frames + self._frames)
            new = ~self.in_fallback & (self.nonwait_frames >= self.cfg.fallback_after * self.cfg.frameskip)
            if new.any():
                self._fallback_activate(np.flatnonzero(new), obs)
            if self.in_fallback.any():
                ids = np.flatnonzero(self.in_fallback)
                mask[ids] = self._fallback_fn(obs[ids])
        self._relay_exit_mask(obs, mask)
        # the trail window after this step's reward.step and every reward.reset of the returned rows
        trail = self.reward.trail_window(obs, self.plane_out["trail"])
        return {"pixels": self._pix, "trail": trail}, mask

    def _fallback_activate(self, ids: np.ndarray, obs) -> None:
        self.in_fallback[ids] = True
        self.ep_fallback[ids] += 1
        for e in ids:
            o = obs[e]
            self._fallback_events.append({
                "env": int(e), "map_group": int(o["map_group"]), "map_num": int(o["map_num"]), "x": int(o["x"]),
                "y": int(o["y"]), "wait_ctx": int(o["wait_ctx"]), "screen": int(o["screen"]),
                "in_battle": int(o["in_battle"]), "frames_non_waiting": int(self.nonwait_frames[e])})
            log.debug("mask fallback: env %d after %d non-waiting frames", e, self.nonwait_frames[e])

    def _start_stats(self, ids: np.ndarray) -> None:
        """Lives starting in envs ``ids`` (not a rewind): their counters, marks and relay status cleared."""
        self.ep_len[ids] = 0
        self.since_progress[ids] = 0
        self.ep_return[ids] = 0
        self.ep_terms[ids] = 0
        self.ep_tile_bonus[ids] = 0
        self.nonwait_frames[ids] = 0
        self.in_fallback[ids] = False
        self.ep_fallback[ids] = 0
        self.life_flags[ids] = 0
        self.life_maps[ids] = 0
        self.life_events[ids] = False
        self.uq_mart[ids] = False
        self.uq_consumable[ids] = False
        self.uq_battle_item[ids] = False
        self.uq_bonus[ids] = 0
        self.lvl_indigo[ids] = False
        self.relay_status[ids] = RSC.INACTIVE
        self.relay_bonus[ids] = 0
        self._rel_paid_act[ids] = False
        self._rel_paid_clr[ids] = False

    def event_fracs(self) -> np.ndarray:
        """float [E]: the share of this group's envs whose current life has each milestone event."""
        return self.life_events.mean(axis=0)

    def _end_stats(self, ids, obs, terminated, sim_terminal) -> List[Dict]:
        badges = popcount8(obs["badges"][ids])
        out = []
        rexit = self._rel_exit
        r_act = RSC.active_scope(self.relay_status[ids])
        r_clr = (self.relay_status[ids] == RSC.CLEARED).sum(axis=1)
        for k, e in enumerate(ids):
            d = {"env": int(e), "return": float(self.ep_return[e]), "length": int(self.ep_len[e]),
                 "badges": int(badges[k]), "maps": int(self.life_maps[e] + self.reward.maps_visited[e]),
                 "flags": int(self.life_flags[e] + self.reward.flags_earned[e]),
                 "champion": bool(terminated[e] and not rexit[e]),
                 "sim_terminal": bool(sim_terminal[e]),
                 "stagnated": bool(self.since_progress[e] >= self.stag_cap[e])}
            for j, t in enumerate(TERMS):
                d["term_" + t] = float(self.ep_terms[e, j])
            d["tiles_count"] = float(self.ep_tile_bonus[e])
            d["fallback_count"] = int(self.ep_fallback[e])
            d["rewinds"] = int(self.rewinds_used[e])
            d["rewind_budget"] = int(self.rewind_budget[e])
            d["length_cap"] = int(self.length_cap[e])
            d["rewind_progress"] = int(self.rw_progress[e])
            d["events"] = np.flatnonzero(self.life_events[e]).tolist()  # (events.EVENTS indices)
            d["end"] = "relay_exit" if rexit[e] else "champion" if terminated[e] else "sim" if sim_terminal[e] else \
                "cap" if self.ep_len[e] >= self.length_cap[e] else "stagnation"
            d["relay_active"] = int(r_act[k])
            d["relay_cleared"] = int(r_clr[k])
            d["term_relay"] = float(self.relay_bonus[e])
            for j, t in enumerate(UNIQUE_TERMS):
                d[t] = float(self.uq_bonus[e, j])
            d["indigo"] = bool(self.lvl_indigo[e])
            out.append(d)
        return out
