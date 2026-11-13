"""PPO training CLI: config, TensorBoard + CSV logging, checkpoints, resume. The defaults are the v76 run.

    PYTHONPATH=rl:python python3 -m pokerl.train --run-name v76_repro          # a fresh run in rl/runs/v76_repro
    PYTHONPATH=rl:python python3 -m pokerl.train --resume v76_repro            # continue it from its last checkpoint

Outputs under ``<runs-dir>/<run-name>/``: ``config.json``, ``tb/`` (TensorBoard), ``log.csv`` (one row per
update), ``mask_fallback.csv`` (one row per mask-fallback activation, env.py; header only in the steady state),
``ckpt_<update>.pt`` every ``--checkpoint-every`` updates (the last 3 kept) plus one at the end / on Ctrl-C, and
``best.pt`` whenever the best-ever badge count at a life's end rises. A checkpoint holds the model, optimiser,
reward scaler, the reward's run-long tile visit counts (key ``reward``), config, update count, global step and
best-ever statistics; a resumed run starts new lives (the env and the recurrent state are not saved) and takes new
values of every flag given, except the network's shape. Milestone events (events.py, ``EventLog``): TensorBoard
``events_env/<NN>_<name>`` (share of the envs whose current life has the event) and ``events_ep/<NN>_<name>`` (share
of the lives that ended in the update with it), the same in ``events.csv``, and ``first_seen.csv`` (the update at
which any env first had each event; read back by ``--resume``).

``--collector pipelined`` (default): the envs are ``--env-groups G`` (3) PokeRLEnv groups of ~N/G (group g: env seed
``seed + g * 0x10000``, sizes differing by at most one env; one shared tile visit table) whose simulation overlaps the
other groups' host work and policy steps (resets run in a helper thread while the other groups keep stepping;
``--post-ahead``: a group's policy step runs on the GPU while the collector does the next groups' host work), and the
PPO update of rollout k runs on its own CUDA stream (``--learner interleave``: one minibatch enqueued after a post
whenever the previous one finished; ``--learner thread``: a Python thread) while rollout k+1 is collected by an
acting copy of the model, which receives the new parameters as soon as the update finishes (mid-rollout).
``--collector sync``: one PokeRLEnv of N envs, collect then update (also the CPU path). Log columns of the cycle:
``sps`` (env-steps per wall second), ``sps_rollout`` / ``rollout_s``, ``sps_env`` (time inside the env step calls),
``update_s``, ``learner_wait_s`` (time the collector waited for the update after its rollout), ``policy_lag`` (the
fraction of the trained rollout collected with parameters one update older), ``t_<phase>`` (seconds per rollout of
the collector's thread: ``act`` policy steps, ``post``, ``wait`` blocked on workers, ``reward``, ``reset``,
``reset_wait``, ``observe``, ``other``, ``host``, ``learner``, ``bootstrap``, ``rewind``, ``relay``).

Frames: ``frames_per_step`` (mean game frames per env-step), ``frames_p50/p90/p99``, ``frames_at_cap`` (fraction of
steps that ran exactly ``--max-frames``), ``frames_total``, ``fps``; ``frames_hist.json`` (the run's frames-per-step
bincount, written at the end).

Rewind columns: ``rw_captures``, ``rw_rewinds``, ``rw_rewinds_per_life``, ``rw_cap_frac`` (lives that used their
whole budget), ``rw_budget_mean``, ``rw_progress_after`` (share of the progress events made after a life's first
rewind), ``rw_mem_mb``, ``rw_snapshot_ms`` / ``rw_restore_ms``. Relay columns: ``relay/activations``,
``relay/exits``, ``relay/masked`` (this update), ``relay/term_relay_mean`` (ended lives). Unique terms:
``ep_term_unique_<kind>_mean``. Reward terms: ``r_<term>`` (mean per env-step of the rollout) and
``std_share/<term>`` (the term's share of the variance of the discounted return the reward scaler divides by).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import glob
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import events as EV
from .env import (EnvConfig, PokeRLEnv, PRESET_BOULDERS_SOLVED_FLAGS, PRESET_BOULDERS_SOLVED_VARS, UNIQUE_KEYS,
                  UNIQUE_TERMS, apply_reward_config, resolve_preset)
from .mask import ACTIONS
from .model import Policy, num_params, policy_kwargs
from .ppo import Collector, Learner, PPOConfig, ppo_update, prepare_learner, sync_actor, warm_up_learner, warmup_lr
from . import reward as R
from .reward import TERMS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS = os.path.normpath(os.path.join(HERE, "..", "runs"))

# v76's skill events: the party / bag menus and the player's PC at skill_menu_reward, Bill's PC 0.1, a party reorder
# 1.0 (cooldown 2000 steps), the rest 0
V76_SKILL_EVENTS = {"tree": 0.0, "boulder": 0.0, "rock": 0.0, "mart": 0.0, "pc_storage": 0.1, "pc_party": 0.0,
                    "party_order": 1.0, "use_item": 0.0, "use_key_item": 0.0, "use_tm": 0.0, "use_consumable": 0.0,
                    "use_battle_item": 0.0, "mart_buy": 0.0}


@dataclass
class TrainConfig:
    run_name: str = "ppo_v76"
    num_envs: int = 1024
    rollout: int = 512
    total_steps: int = 20_000_000_000
    seed: int = 0
    device: str = "cuda"
    checkpoint_every: int = 50
    keep_checkpoints: int = 3
    max_episode_steps: int = 327680
    stagnation_steps: int = 2048  # 0 = no stagnation end (needs --no-rewind)
    stagger_min: int = 2048       # the first life's cap: U[stagger_min, stagger_max]
    stagger_max: int = 32768
    num_workers: Optional[int] = None
    worker_path: Optional[str] = None
    fallback_after: int = 72      # mask fallback after this many non-waiting steps' worth of frames (env.py; 0 = off)
    hidden: int = 512             # core / minGRU size (model.py)
    pixel_feat: int = 512         # features of the screen branch (model.py)
    chrono_tau: float = 512.0     # minGRU gate biases at init: -log tau, tau log-uniform in [1, this] (0 = off)
    collector: str = "pipelined"  # "pipelined" (env groups + overlapped learner) or "sync"
    env_groups: int = 3           # pipelined: number of env groups (sizes N // G or N // G + 1)
    post_ahead: bool = True       # pipelined: the policy step overlaps the next groups' host work (ppo.Collector)
    learner: str = "interleave"   # pipelined: "interleave" (update pieces between posts) or "thread"
    bedroom_cache: int = 1        # start states cached per worker process
    pin_cpus: bool = False        # pin workers round-robin to the CPUs (every group over all of them)
    max_frames: int = 16 * 24     # game frames per env-step at most (decision-point stepping)
    dpad_hold: int = 10           # D-pad hold toward a new direction in the free overworld (env.py; 0 = off)
    rewind: bool = True           # retry from the env's own recent captures on stagnation (env.py "Rewind")
    rewind_max: int = 3           # the rewind budget's upper end at 0 badges
    rewind_max_min: int = 0       # its lower end
    rewind_per_badge: int = 1     # + this per badge the life has held on the upper end
    rewind_ring: int = 1          # captures kept per env
    rewind_depth: int = 1         # a rewind picks uniformly among the newest min(depth, available) captures
    rewind_min_gap: int = 512     # steps between two captures of an env at least
    carry_paid: bool = True       # rewind captures carry the reward's "already paid" state
    # reward (reward.configure; reward.py and env.py docstrings)
    skill_reward: float = 0.1
    skill_menu_reward: Optional[float] = 0.02
    trainer_reward: Optional[float] = 1.0
    temp_flag_reward: float = 1.0
    skill_event_reward: Optional[Dict[str, float]] = field(default_factory=lambda: dict(V76_SKILL_EVENTS))
    skill_event_cooldown: Optional[Dict[str, int]] = field(default_factory=lambda: {"party_order": 2000})
    badge_reward: Optional[List[float]] = field(default_factory=lambda: [10.0] * 8)
    elite4_reward: Optional[float] = 10.0
    champion_reward: Optional[float] = 20.0
    hm_reward_moves: Optional[List[str]] = field(default_factory=lambda: ["cut", "surf", "strength"])
    tile_reward: Optional[float] = 0.02
    tile_count_reward: Optional[float] = 0.05
    tile_count_scale: Optional[float] = 512.0
    map_reward: Optional[float] = 0.05
    whiteout_reward: Optional[float] = -0.2
    trash_reward: float = 0.1
    trash_unlock_reward: Optional[float] = 5.0
    mansion_switch_reward: float = 0.1
    spinner_reward: float = 0.1
    warp_pad_reward: float = 0.1
    interact_reward: float = 0.1
    hm_taught_reward: Optional[float] = 10.0
    level_reward: Optional[float] = 1.0
    level_reward_indigo: float = 3.0
    move_replace_reward: Optional[float] = 0.5
    unique_mart_reward: float = 0.5
    unique_consumable_reward: float = 0.5
    unique_battle_item_reward: float = 0.5
    relay_activation_reward: float = 2.0
    relay_clear_reward: float = 2.0
    # new-game preset (env.py): the boulder puzzles solved
    preset_flags: Tuple[str, ...] = PRESET_BOULDERS_SOLVED_FLAGS
    preset_vars: Tuple[str, ...] = PRESET_BOULDERS_SOLVED_VARS
    max_minutes: float = 0.0      # wall-clock budget from the first update of this process (0 = none): stop + checkpoint
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def to_json(self) -> Dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: Dict) -> "TrainConfig":
        d = dict(d)
        for k in ("preset_flags", "preset_vars"):  # (JSON lists)
            if k in d:
                d[k] = tuple(d[k])
        ppo = PPOConfig(**d.pop("ppo", {}))
        return TrainConfig(ppo=ppo, **d)


# the TrainConfig fields that are the network's shape: a resumed run keeps its own
MODEL_KEYS = ("hidden", "pixel_feat", "chrono_tau")
# the TrainConfig fields that are EnvConfig fields of the same name (make_envs)
ENV_KEYS = tuple(f.name for f in dataclasses.fields(EnvConfig)
                 if f.name in {g.name for g in dataclasses.fields(TrainConfig)} and f.name not in ("num_envs", "seed"))


def _nanmean(x: List[float]) -> float:
    return float(np.mean(x)) if x else float("nan")


def _nanmax(x: List[float]) -> float:
    return float(np.max(x)) if x else float("nan")


class Logger:
    """TensorBoard scalars + a flat CSV (columns fixed by the first row; keys new to an existing file are
    appended to its header once, earlier rows padded with empty cells). ``csv_name``: the file in the run dir;
    ``tb``: a SummaryWriter to share (another Logger's ``tb``; closed by its owner), else one of ``<run>/tb``;
    ``no_tb``: keys written to the CSV only. Values that are not finite numbers (e.g. None) go to the CSV
    only (None as an empty cell)."""

    def __init__(self, run_dir: str, csv_name: str = "log.csv", tb=None, no_tb=()):
        self._own_tb = tb is None
        if tb is None:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(os.path.join(run_dir, "tb"))
        self.tb = tb
        self.no_tb = {"update", "global_step", *no_tb}
        self.csv_path = os.path.join(run_dir, csv_name)
        self.columns = None
        if os.path.exists(self.csv_path):
            with open(self.csv_path) as f:
                header = f.readline().strip()
            self.columns = header.split(",") if header else None

    def log(self, row: Dict[str, float], step: int) -> None:
        for k, v in row.items():
            if k not in self.no_tb and isinstance(v, (int, float)) and math.isfinite(v):
                self.tb.add_scalar(k, v, step)
        new = self.columns is None
        if new:
            self.columns = list(row.keys())
        elif any(k not in self.columns for k in row):
            self._extend([k for k in row if k not in self.columns])
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self.columns)
            w.writerow([row.get(c, "") for c in self.columns])

    def _extend(self, extra: List[str]) -> None:
        with open(self.csv_path, newline="") as f:
            rows = list(csv.reader(f))
        self.columns = self.columns + extra
        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            w.writerows(r + [""] * (len(self.columns) - len(r)) for r in rows[1:])
        os.replace(tmp, self.csv_path)
        print(f"{os.path.basename(self.csv_path)}: added columns {', '.join(extra)}", flush=True)

    def close(self):
        if self._own_tb:
            self.tb.flush()
            self.tb.close()


FIRST_SEEN_COLUMNS = ["update", "global_step", "tag", "kind", "key", "id"]


class EventLog:
    """Milestone events (events.py) per update: TensorBoard ``events_env/<tag>`` (the share of all envs whose current
    life has the event; groups weighted by their env count) and, when a life ended in the update, ``events_ep/<tag>``
    (the share of the lives that ended with it), through the run's SummaryWriter (``tb``), not in log.csv; the same
    columns in ``<run>/events.csv`` (``update, global_step, ep_count`` then the columns, ``events_ep/*`` empty when no
    life ended); ``<run>/first_seen.csv`` (``update, global_step, tag, kind, key, id``): one row the first time any env
    of the run has an event (a current life or a life that ended in the update), with a console line; read back on
    ``--resume`` so nothing is announced twice."""

    def __init__(self, run_dir: str, tb):
        self.csv = Logger(run_dir, "events.csv", tb=tb, no_tb=("ep_count",))
        self.first_path = os.path.join(run_dir, "first_seen.csv")
        self.seen = set()
        if os.path.exists(self.first_path):
            with open(self.first_path, newline="") as f:
                self.seen = {r["tag"] for r in csv.DictReader(f)}
        else:
            with open(self.first_path, "w", newline="") as f:
                csv.writer(f).writerow(FIRST_SEEN_COLUMNS)

    def log(self, envs: List[PokeRLEnv], eps: List[Dict], update: int, global_step: int) -> Dict[str, float]:
        """Logs one update; returns the row (``events_env/*``, ``events_ep/*``)."""
        n = max(sum(e.num_envs for e in envs), 1)
        env_counts = np.sum([e.event_fracs() * e.num_envs for e in envs], axis=0)
        ep = np.zeros((len(eps), EV.NUM_EVENTS), dtype=bool)
        for j, d in enumerate(eps):
            ep[j, d.get("events", [])] = True
        row: Dict[str, Optional[float]] = {"update": update, "global_step": global_step, "ep_count": len(eps)}
        row.update({f"events_env/{t}": float(c / n) for t, c in zip(EV.TAGS, env_counts)})
        ep_frac = ep.mean(axis=0) if len(eps) else None
        row.update({f"events_ep/{t}": None if ep_frac is None else float(ep_frac[i]) for i, t in enumerate(EV.TAGS)})
        self.csv.log(row, global_step)
        now = (env_counts > 0) | ep.any(axis=0)
        new = [i for i in np.flatnonzero(now) if EV.TAGS[i] not in self.seen]
        if new:
            with open(self.first_path, "a", newline="") as f:
                w = csv.writer(f)
                for i in new:
                    ev = EV.EVENTS[i]
                    key = f"{ev.key}>={ev.arg}" if ev.kind == "var" else ev.key
                    w.writerow([update, global_step, ev.tag, ev.kind, key, ev.id])
                    self.seen.add(ev.tag)
                    print(f"event {ev.tag} first seen at update {update}", flush=True)
        return row


FALLBACK_COLUMNS = ["update", "global_step", "env", "map_group", "map_num", "x", "y", "wait_ctx", "screen",
                    "in_battle", "frames_non_waiting"]


class FallbackLog:
    """``<run_dir>/mask_fallback.csv``: one row per mask-fallback activation (wait_ctx and screen by name)."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "mask_fallback.csv")
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(FALLBACK_COLUMNS)
        from fireredsim.obs_dtype import SCREEN_NAMES, WAIT_CTX_NAMES
        self.names = {"screen": SCREEN_NAMES, "wait_ctx": WAIT_CTX_NAMES}

    def _name(self, k: str, v):
        names = self.names.get(k)
        return names[v] if names is not None and 0 <= v < len(names) else v

    def write(self, events: List[Dict], update: int, global_step: int) -> None:
        if not events:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            for e in events:
                d = dict(e, update=update, global_step=global_step)
                w.writerow([self._name(k, d[k]) for k in FALLBACK_COLUMNS])


def _checkpoints(run_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))


def save_checkpoint(path: str, model, opt, scaler, cfg: TrainConfig, update: int, global_step: int, best: Dict,
                    reward=None):
    tmp = path + ".tmp"
    osd = opt.state_dict()
    for gr in osd["param_groups"]:  # (a warm-up lr held in a device tensor, Learner.set_lr: saved as its float)
        if torch.is_tensor(gr.get("lr")):
            gr["lr"] = float(gr["lr"])
    d = {"model": model.state_dict(), "optimizer": osd, "scaler": scaler.state_dict(),
         "config": cfg.to_json(), "update": update, "global_step": global_step, "best": best}
    if reward is not None:
        d["reward"] = reward.state_dict()
    torch.save(d, tmp)
    os.replace(tmp, path)


def _arg_type(parse):
    """An argparse type from a reward.parse_* function (ValueError -> a usage error)."""
    def f(text):
        try:
            return parse(text)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e)) from None
    return f


def _names(text: str) -> Tuple[str, ...]:
    return tuple(t.strip() for t in text.split(",") if t.strip())


# command-line flags of TrainConfig fields (--flag-name -> field flag_name): (type or "bool", help)
CONFIG_ARGS = {
    "num_envs": (int, "envs (default 1024)"),
    "rollout": (int, "rollout length T (default 512)"),
    "total_steps": (int, "env-steps to train (default 2e10)"),
    "run_name": (str, "the run directory's name under --runs-dir"),
    "seed": (int, "base seed (default 0)"),
    "device": (str, "torch device (default cuda; cpu: use --collector sync --no-compile --no-autocast)"),
    "num_workers": (int, "worker processes per env group (default: one per CPU)"),
    "worker_path": (str, "the fireredsim-worker executable (default: $FIREREDSIM_WORKER, else build*/worker/)"),
    "checkpoint_every": (int, "updates between checkpoints (default 50)"),
    "hidden": (int, "core / minGRU size (default 512)"),
    "pixel_feat": (int, "screen branch features (default 512)"),
    "chrono_tau": (float, "minGRU gate bias init: -log tau, tau log-uniform in [1, this] (default 512; 0 = off)"),
    "collector": (["pipelined", "sync"], "pipelined (default): env groups + overlapped learner; sync: collect, then "
                                         "update"),
    "env_groups": (int, "pipelined collector: number of env groups (default 3)"),
    "post_ahead": ("bool", "pipelined: the policy step overlaps the next groups' host work (default on)"),
    "learner": (["interleave", "thread"], "pipelined: how the update overlaps the collection (default interleave)"),
    "pin_cpus": ("bool", "pin the workers round-robin to the CPUs (default off)"),
    "max_episode_steps": (int, "length cap of a life in env-steps (default 327680)"),
    "stagnation_steps": (int, "steps without progress before a life ends (default 2048; 0 = off, needs --no-rewind)"),
    "stagger_min": (int, "the first life's cap is drawn from U[stagger_min, stagger_max] (default 2048)"),
    "stagger_max": (int, "(default 32768)"),
    "fallback_after": (int, "mask fallback after this many non-waiting steps' worth of frames (default 72, 0 = off)"),
    "max_frames": (int, "game frames per env-step at most (default 384)"),
    "dpad_hold": (int, "frames a D-pad press toward a new direction is held (default 10: turn + one tile; 0 = off)"),
    "rewind": ("bool", "rewind to the life's own recent capture on stagnation (default on)"),
    "rewind_max": (int, "the rewind budget's upper end at 0 badges (default 3)"),
    "rewind_max_min": (int, "the rewind budget's lower end (default 0)"),
    "rewind_per_badge": (int, "the budget's upper end rises by this per badge held (default 1)"),
    "rewind_ring": (int, "captures kept per env (default 1)"),
    "rewind_depth": (int, "a rewind picks uniformly among the newest this many captures (default 1)"),
    "rewind_min_gap": (int, "steps between two captures of an env at least (default 512)"),
    "carry_paid": ("bool", "rewind captures carry the reward's already-paid state (default on)"),
    "skill_reward": (float, "skill event weight (default 0.1)"),
    "skill_menu_reward": (float, "weight of the menu-opening skill events (default 0.02)"),
    "trainer_reward": (float, "trainer flag weight (default 1.0)"),
    "temp_flag_reward": (float, "weight of FLAG_TEMP_1..6 (default 1.0)"),
    "skill_event_reward": (_arg_type(R.parse_skill_event_reward), "per-event skill weights, name=W,... (default: "
                                                                  "pc_storage=0.1, party_order=1.0, the rest 0)"),
    "skill_event_cooldown": (_arg_type(R.parse_skill_event_cooldown), "per-event cooldowns, name=STEPS,... "
                                                                      "(default party_order=2000; others 250)"),
    "badge_reward": (_arg_type(lambda s: [float(v) for v in R.parse_badge_reward(s)]), "8 badge weights (default 10 "
                                                                                       "each)"),
    "elite4_reward": (float, "Elite Four flag weight (default 10)"),
    "champion_reward": (float, "the Champion's flag weight (default 20)"),
    "hm_reward_moves": (_arg_type(lambda s: list(R.parse_hm_reward_moves(s) or R.HM_NAMES)),
                        "the HMs paying hm_cover / hm_taught (default cut,surf,strength)"),
    "tile_reward": (float, "per tile first visited in a life (default 0.02)"),
    "tile_count_reward": (float, "count bonus W / sqrt(max(N, S) / S) on a first visit (default 0.05)"),
    "tile_count_scale": (float, "its S (default 512)"),
    "map_reward": (float, "per map first visited in a life (default 0.05)"),
    "whiteout_reward": (float, "on a whiteout, <= 0 (default -0.2)"),
    "trash_reward": (float, "per Vermilion Gym trash can interaction (default 0.1)"),
    "trash_unlock_reward": (float, "weight of FLAG_FOUND_BOTH_VERMILION_GYM_SWITCHES (default 5)"),
    "mansion_switch_reward": (float, "per Pokemon Mansion switch first pressed in a life (default 0.1)"),
    "spinner_reward": (float, "per spinner tile first used in a life (default 0.1)"),
    "warp_pad_reward": (float, "per warp pad first used in a life (default 0.1)"),
    "interact_reward": (float, "per overworld interaction key first started in a life (default 0.1)"),
    "hm_taught_reward": (float, "per paid HM move newly known (default 10)"),
    "level_reward": (float, "the levels term's weight (default 1)"),
    "level_reward_indigo": (float, "the levels term's weight once the life reached Indigo Plateau (default 3; -1 off)"),
    "move_replace_reward": (float, "per move learnt in place of a forgotten one, once per mon and move (default "
                                   "0.5)"),
    "unique_mart_reward": (float, "per mart map bought on, once per life (default 0.5)"),
    "unique_consumable_reward": (float, "per item id used on the field, once per life (default 0.5)"),
    "unique_battle_item_reward": (float, "per item id used in battle, once per life (default 0.5)"),
    "relay_activation_reward": (float, "per relay scope activated, once per scope and life (default 2)"),
    "relay_clear_reward": (float, "per relay scope cleared, once per scope and life (default 2)"),
    "preset_flags": (_names, "FLAG_* names set in every new game (default: the boulder puzzles solved)"),
    "preset_vars": (_names, "VAR_*=VALUE game variables set in every new game (default: the boulder puzzles solved)"),
    "max_minutes": (float, "wall-clock budget of this process in minutes, then a checkpoint (default 0 = none)"),
}
# command-line flags of PPOConfig fields
PPO_ARGS = {
    "minibatches": (int, "minibatches per epoch (default 64; num_envs must be a multiple)"),
    "epochs": (int, "epochs per update (default 1)"),
    "ent_coef": (float, "entropy coefficient (default 0.03)"),
    "gamma": (float, "discount (default 0.997)"),
    "lr": (float, "Adam learning rate (default 2.5e-4)"),
    "lr_warmup": (int, "a linear learning-rate ramp over the first this many updates of a fresh run and after a resume "
                       "(default 100; 0 = off)"),
    "compile": ("bool", "torch.compile the update (CUDA; default on)"),
    "autocast": ("bool", "bf16 autocast in the update (CUDA; default on)"),
}


def _add(ap, name: str, spec) -> None:
    kind, help_ = spec
    flag = "--" + name.replace("_", "-")
    if kind == "bool":
        ap.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=None, help=help_)
    elif isinstance(kind, list):
        ap.add_argument(flag, dest=name, choices=kind, default=None, help=help_)
    else:
        ap.add_argument(flag, dest=name, type=kind, default=None, help=help_)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PPO on the native FireRed simulator (the v76 configuration by default)")
    ap.add_argument("--resume", metavar="RUN", help="continue the run <runs-dir>/RUN from its last checkpoint")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    for k, spec in CONFIG_ARGS.items():
        _add(ap, k, spec)
    for k, spec in PPO_ARGS.items():
        _add(ap, k, spec)
    return ap.parse_args(argv)


def start_preset_summary(cfg: TrainConfig) -> str:
    if not cfg.preset_flags and not cfg.preset_vars:
        return "new-game preset: none"
    return (f"new-game preset: {len(cfg.preset_flags)} flags ({', '.join(cfg.preset_flags)}); "
            f"{len(cfg.preset_vars)} vars ({', '.join(cfg.preset_vars)})")


GROUP_SEED_STRIDE = 0x10000  # env group g uses seed + g * GROUP_SEED_STRIDE (no episode seed repeats)


def ensure_native() -> None:
    """Loads the native library, building it first when it is missing or stale (its spec hash covers the
    observation dtype and the reward tables and constants, so it is rebuilt for every reward config)."""
    from . import native as NV
    try:
        NV.load()
        return
    except NV.NativeUnavailable as e:
        print(f"native library: {e}\nbuilding rl/pokerl/native ...", flush=True)
    try:
        from .native.build import build
        build(verbose=False)
        NV.load()
    except Exception as e:  # noqa: BLE001 (gcc missing, hash still differing, ...)
        raise SystemExit(f"native library unusable ({type(e).__name__}: {e})")


def make_envs(cfg: TrainConfig, seed: int) -> List[PokeRLEnv]:
    """One PokeRLEnv of N envs (sync) or ``cfg.env_groups`` of ~N/G (pipelined; the first N mod G groups one env
    larger) sharing one tile visit table."""
    groups = cfg.env_groups if cfg.collector == "pipelined" else 1
    if not 1 <= groups <= cfg.num_envs:
        raise SystemExit(f"--env-groups {groups}: 1 .. --num-envs ({cfg.num_envs})")
    sizes = [cfg.num_envs // groups + (g < cfg.num_envs % groups) for g in range(groups)]
    ncpu = os.cpu_count() or 1
    envs: List[PokeRLEnv] = []
    tile_count = None
    off_g = 0
    for g in range(groups):
        cpus = None
        if cfg.pin_cpus:  # worker w of group g -> CPU (w + g * ncpu / groups) mod ncpu
            off = g * ncpu // groups
            cpus = [(c + off) % ncpu for c in range(ncpu)]
        kw = {k: getattr(cfg, k) for k in ENV_KEYS}
        if cfg.num_workers is not None:
            kw["num_workers"] = max(1, cfg.num_workers // groups)
        e = PokeRLEnv(EnvConfig(num_envs=sizes[g], seed=seed + g * GROUP_SEED_STRIDE, cpus=cpus, **kw),
                      tile_count=tile_count)
        e.env_offset = off_g
        off_g += sizes[g]
        tile_count = e.reward.tile_count
        envs.append(e)
    return envs


def drain_fallback(envs: List[PokeRLEnv]) -> Dict:
    """``PokeRLEnv.drain_fallback`` over the groups (event env indices of the whole collector)."""
    out = {"events": [], "steps": 0, "fallback_steps": 0, "active": 0}
    off = 0
    for e in envs:
        d = e.drain_fallback()
        out["events"].extend(dict(ev, env=ev["env"] + off) for ev in d["events"])
        for k in ("steps", "fallback_steps", "active"):
            out[k] += d[k]
        off += e.num_envs
    return out


def _drain_sum(envs: List[PokeRLEnv], name: str) -> Dict:
    """``PokeRLEnv.<name>()`` (drain_relay, drain_rewind) summed over the groups."""
    out: Dict = {}
    for e in envs:
        for k, v in getattr(e, name)().items():
            out[k] = out.get(k, 0) + v
    return out


def print_gpu_memory(device: torch.device, when: str) -> None:
    """What this process's allocator holds and the whole card's use, other programs included, in one line."""
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"gpu memory {when}: allocated {torch.cuda.memory_allocated(device) / 2 ** 30:.2f} GiB, reserved "
              f"{torch.cuda.memory_reserved(device) / 2 ** 30:.2f} GiB, card {(total - free) / 2 ** 30:.2f} of "
              f"{total / 2 ** 30:.2f} GiB", flush=True)


def _rewind_columns(rd: Dict, eps: List[Dict]) -> Dict[str, float]:
    """The per-update rewind columns from the drained counters and the ended lives."""
    def per(t, n):
        return 1e3 * t / n if n else float("nan")
    return {"rw_captures": rd["captures"], "rw_rewinds": rd["rewinds"], "rw_lives": len(eps),
            "rw_rewinds_per_life": _nanmean([e["rewinds"] for e in eps]),
            "rw_cap_frac": _nanmean([float(e["rewinds"] >= e["rewind_budget"]) for e in eps]),
            "rw_budget_mean": _nanmean([e["rewind_budget"] for e in eps]),
            "rw_progress_after": rd["progress_after"] / rd["progress"] if rd["progress"] else float("nan"),
            "rw_mem_mb": rd["mem_bytes"] / 2 ** 20, "rw_snapshot_ms": per(rd["t_snapshot"], rd["snapshot_calls"]),
            "rw_restore_ms": per(rd["t_restore"], rd["restore_calls"])}


def drain_frames(envs: List[PokeRLEnv]) -> Dict:
    """``PokeRLEnv.drain_frames`` over the groups."""
    out = {"steps": 0, "frames": 0, "frames_hist": np.zeros(0, dtype=np.int64)}
    for e in envs:
        d = e.drain_frames()
        out["steps"] += d["steps"]
        out["frames"] += d["frames"]
        out["frames_hist"] = _add_hist(out["frames_hist"], d["frames_hist"])
    return out


def _add_hist(a: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
    if b is None or b.size == 0:
        return a
    if b.size > a.size:
        a, b = b.copy(), a
    a[:b.size] += b
    return a


def _hist_percentile(h: np.ndarray, q: float) -> float:
    if h.size == 0 or h.sum() == 0:
        return float("nan")
    c = np.cumsum(h)
    return float(np.searchsorted(c, q * c[-1]))


def _config(args) -> Tuple[TrainConfig, str, Optional[dict]]:
    """(config, run dir, checkpoint or None) of a fresh run (the defaults + the flags given) or a resumed one (its
    config.json + the flags given, the network's shape refused)."""
    ckpt = None
    if args.resume:
        run_dir = os.path.join(args.runs_dir, args.resume)
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = TrainConfig.from_json(json.load(f))
        for k in MODEL_KEYS:
            if getattr(args, k, None) is not None and getattr(args, k) != getattr(cfg, k):
                raise SystemExit(f"--{k.replace('_', '-')} {getattr(args, k)} differs from the run's {getattr(cfg, k)}")
        cks = _checkpoints(run_dir)
        if not cks:
            raise SystemExit(f"no checkpoint in {run_dir}")
        ckpt = torch.load(cks[-1], map_location="cpu", weights_only=False)
        print(f"resuming {cks[-1]}")
    else:
        cfg = TrainConfig()
    for k in CONFIG_ARGS:
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    for k in PPO_ARGS:
        if getattr(args, k) is not None:
            setattr(cfg.ppo, k, getattr(args, k))
    run_dir = os.path.join(args.runs_dir, cfg.run_name if not args.resume else args.resume)
    if not args.resume and os.path.exists(os.path.join(run_dir, "config.json")):
        raise SystemExit(f"{run_dir} exists: use --resume {cfg.run_name} or another --run-name")
    return cfg, run_dir, ckpt


def check_config(cfg: TrainConfig) -> None:
    """SystemExit for an invalid combination of settings."""
    try:
        resolve_preset(cfg.preset_flags, cfg.preset_vars)
    except ValueError as e:
        raise SystemExit(f"--preset-flags / --preset-vars: {e}") from None
    if cfg.rewind and cfg.stagnation_steps == 0:
        raise SystemExit("--stagnation-steps 0 needs --no-rewind (a rewind happens on a stagnation end only)")
    if cfg.rewind and not (cfg.rewind_max >= 0 and cfg.rewind_ring >= 1 and cfg.rewind_depth >= 1
                           and cfg.rewind_min_gap >= 0 and 0 <= cfg.rewind_max_min <= cfg.rewind_max
                           and cfg.rewind_per_badge >= 0):
        raise SystemExit("--rewind-ring and --rewind-depth must be >= 1, --rewind-max, --rewind-min-gap and "
                         "--rewind-per-badge >= 0, --rewind-max-min 0 .. --rewind-max")
    if min(cfg.unique_mart_reward, cfg.unique_consumable_reward, cfg.unique_battle_item_reward,
           cfg.relay_activation_reward, cfg.relay_clear_reward) < 0:
        raise SystemExit("the unique and relay rewards must be >= 0")
    if cfg.ppo.lr <= 0 or cfg.ppo.lr_warmup < 0:
        raise SystemExit("--lr must be > 0 and --lr-warmup >= 0")
    if cfg.stagnation_steps < 0:
        raise SystemExit("--stagnation-steps must be >= 0")
    if cfg.ppo.epochs < 1 or cfg.ppo.ent_coef < 0:
        raise SystemExit(f"--epochs must be >= 1 and --ent-coef >= 0, not {cfg.ppo.epochs} and {cfg.ppo.ent_coef}")
    if cfg.num_envs % cfg.ppo.minibatches:
        raise SystemExit(f"num_envs ({cfg.num_envs}) must be a multiple of {cfg.ppo.minibatches} minibatches")
    if cfg.chrono_tau and cfg.chrono_tau < 1:
        raise SystemExit(f"--chrono-tau must be 0 (off) or >= 1, not {cfg.chrono_tau}")
    if cfg.collector not in ("pipelined", "sync"):
        raise SystemExit(f"collector must be pipelined or sync, not {cfg.collector!r}")


def main(argv=None) -> Dict:
    args = parse_args(argv)
    cfg, run_dir, ckpt = _config(args)
    check_config(cfg)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_json(), f, indent=2)

    seed = cfg.seed + (ckpt["update"] if ckpt else 0)  # a resumed run draws new episodes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg.device)
    pipelined = cfg.collector == "pipelined"

    model = Policy(**policy_kwargs(cfg.to_json())).to(device)
    # fused Adam (one kernel per step; fp32 like the foreach version): the update's host time shrinks by ~30 %
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr, eps=cfg.ppo.adam_eps, fused=device.type == "cuda")
    print(f"model: {num_params(model):,} parameters (hidden {cfg.hidden}, gray2 "
          f"{'x'.join(map(str, model.pixel_shape))} + trail"
          + (f", chrono gate biases tau 1..{cfg.chrono_tau:g}" if cfg.chrono_tau else "")
          + f"); run dir {run_dir}; collector {cfg.collector}" + (f", learner {cfg.learner}" if pipelined else ""))
    update, global_step = 0, 0
    best = {"badges": 0, "flags": 0, "champions": 0, "return": float("-inf")}
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
    # the acting policy: a separate copy read by the step graphs while the learner updates ``model``
    actor = copy.deepcopy(model) if pipelined else model
    try:
        apply_reward_config(cfg)  # (before the native library is built: its spec hash covers the weights)
    except ValueError as e:
        raise SystemExit(f"reward: {e}") from None
    print(f"reward: skill term {cfg.skill_reward:g} (menu events {cfg.skill_menu_reward:g}); trainer weight "
          f"{cfg.trainer_reward:g}; {R.summary()}"
          + (f"; wall-clock budget {cfg.max_minutes:g} min" if cfg.max_minutes > 0 else ""), flush=True)
    print(f"life ends: stagnation {cfg.stagnation_steps} (a party level rise and an opponent KO are progress), "
          f"length cap {cfg.max_episode_steps}; returns: gamma {cfg.ppo.gamma:g}, GAE lambda {cfg.ppo.gae_lambda:g}",
          flush=True)
    print(start_preset_summary(cfg), flush=True)
    print(f"relay scopes: activation {cfg.relay_activation_reward:g}, clear {cfg.relay_clear_reward:g}, ordered, "
          f"confined (the presses that leave the area are masked, any other way out terminates)", flush=True)
    ensure_native()  # builds rl/pokerl/native if missing or stale
    envs = make_envs(cfg, seed)
    collector = Collector(envs, actor, cfg.ppo, device, rollout=cfg.rollout, buffers=2 if pipelined else 1, seed=seed,
                          post_ahead=cfg.post_ahead)
    st = collector.store
    pix = st.fields["pixels"]
    print(f"rollout store: {st.nbytes / 2 ** 30:.2f} GiB on {device} ({st.buffers} x {st.T} x {st.N}; pixels "
          f"{pix.numel() * pix.element_size() / 2 ** 30:.2f} GiB as uint8)", flush=True)
    print_gpu_memory(device, "after building the collector")
    prepare_learner(model, cfg.ppo)
    if ckpt:
        collector.scaler.load_state_dict(ckpt["scaler"])
        envs[0].reward.load_state_dict(ckpt["reward"])  # in place: the table is shared by the groups
        update, global_step, best = ckpt["update"], ckpt["global_step"], ckpt["best"]
    logger = Logger(run_dir)
    event_log = EventLog(run_dir, logger.tb)
    fallback_log = FallbackLog(run_dir)
    steps_per_update = cfg.rollout * cfg.num_envs
    last_row: Dict = {}
    frames_total = 0
    frames_hist = np.zeros(0, dtype=np.int64)

    def checkpoint(name: Optional[str] = None):
        path = os.path.join(run_dir, name or f"ckpt_{update:06d}.pt")
        save_checkpoint(path, model, opt, collector.scaler, cfg, update, global_step, best, envs[0].reward)
        if name is None:
            for old in _checkpoints(run_dir)[:-cfg.keep_checkpoints]:
                os.remove(old)

    t_start = time.time()
    gpu_memory_lines = 3  # (the update graphs of the two rollout buffers are captured in the first updates)
    learner = Learner(model, opt, cfg.ppo, device, mode=cfg.learner) if pipelined else None
    synced_at = [None]  # env-steps of the current rollout collected before the actor got the new parameters

    def poll():
        """Called by the collector after every post: advances an interleaved update, and as soon as the
        update has finished copies the new parameters into the actor (mid-rollout: the rest of the
        rollout is collected with them, which keeps the policy lag of the next update small)."""
        if cfg.learner == "interleave":
            learner.poll()
        if synced_at[0] is None and learner.finished:
            with torch.cuda.stream(collector.stream) if collector.stream is not None else contextlib.nullcontext():
                if collector.stream is not None and learner.stream is not None:
                    collector.stream.wait_stream(learner.stream)
                sync_actor(actor, model)
            synced_at[0] = collector.posted_steps
    ros = collector.rollouts
    k = 0  # pipelined: the rollout buffer ready for the next update
    first_update = update
    lag_next = 0.0
    try:
        if pipelined and global_step < cfg.total_steps:
            collector.collect(ros[0], gae=False)  # (its statistics are not logged: no update consumed it yet)
            warm_up_learner(model, ros[0], cfg.ppo)
        t_last = time.perf_counter()
        t_budget = time.time()  # the wall-clock budget counts from the first update (start-up costs differ per config)
        while global_step < cfg.total_steps:
            if cfg.max_minutes > 0 and time.time() - t_budget >= cfg.max_minutes * 60:
                print(f"wall-clock budget of {cfg.max_minutes:g} min reached at update {update}: saving a checkpoint",
                      flush=True)
                break
            # learning rate of this update: the warm-up ramp at the start of a fresh run and after a resume
            lr = warmup_lr(cfg.ppo, update + 1 - first_update)
            if pipelined:
                learner.set_lr(lr)
                learner.start(ros[k])                 # update on rollout k ...
                synced_at[0] = None
                rs = collector.collect(ros[1 - k], gae=False, poll=poll)  # ... while rollout k+1 is collected
                tw = time.perf_counter()
                us = learner.wait()
                learner_wait = time.perf_counter() - tw
                if synced_at[0] is None:  # the update outlasted the rollout: copy at the boundary
                    sync_actor(actor, model)
                    synced_at[0] = steps_per_update
                k = 1 - k
                # policy lag of THIS update (rollout k): the fraction of it collected before the actor got
                # the parameters this update starts from (0 for the first update after a start/resume)
                lag = 0.0 if update == first_update else lag_next
                lag_next = synced_at[0] / steps_per_update
            else:
                rs = collector.collect(ros[0])
                for gr in opt.param_groups:
                    gr["lr"] = lr
                us = ppo_update(model, opt, ros[0], cfg.ppo)
                learner_wait, lag = 0.0, 0
            now = time.perf_counter()
            cycle, t_last = now - t_last, now
            update += 1
            global_step += steps_per_update
            fb = drain_fallback(envs)
            fallback_log.write(fb["events"], update, global_step)
            fr = drain_frames(envs)
            frames_total += fr["frames"]
            frames_hist = _add_hist(frames_hist, fr["frames_hist"])
            eps = rs["episodes"]
            badges = [e["badges"] for e in eps]
            improved = False
            if eps:
                if max(badges) > best["badges"]:
                    best["badges"], improved = max(badges), True
                best["flags"] = max(best["flags"], max(e["flags"] for e in eps))
                best["return"] = max(best["return"], max(e["return"] for e in eps))
                best["champions"] += sum(e["champion"] for e in eps)
            row = {"update": update, "global_step": global_step, "time": time.time() - t_start,
                   "sps": steps_per_update / cycle, "sps_rollout": steps_per_update / rs["seconds"],
                   "sps_env": steps_per_update / max(rs["env_seconds"], 1e-9),
                   "rollout_s": rs["seconds"], "update_s": us["seconds"], "learner_wait_s": learner_wait,
                   "policy_lag": lag, "lr": lr}
            row.update({k: us[k] for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "explained_var",
                                            "grad_norm", "minibatches", "kl_stop")})
            row["reward_raw"] = rs["reward_raw"]
            row["reward_std"] = collector.scaler.std
            row.update({f"r_{k}": rs["terms"][k] for k in TERMS})
            shares = collector.scaler.std_shares()
            if shares is not None:  # Cov(G_k, G) / Var(G): the terms' shares of the scaler's return variance
                row.update({f"std_share/{k}": float(s) for k, s in zip(TERMS, shares)})
            row.update({"ep_count": len(eps), "ep_return_mean": _nanmean([e["return"] for e in eps]),
                        "ep_return_max": _nanmax([e["return"] for e in eps]),
                        "ep_len_mean": _nanmean([e["length"] for e in eps]),
                        "ep_len_max": _nanmax([e["length"] for e in eps]),
                        "ep_badges_mean": _nanmean(badges), "ep_badges_max": _nanmax(badges),
                        "ep_maps_mean": _nanmean([e["maps"] for e in eps]),
                        "ep_flags_mean": _nanmean([e["flags"] for e in eps]),
                        "ep_stagnated": _nanmean([float(e["stagnated"]) for e in eps])})
            if cfg.rewind:
                row.update(_rewind_columns(_drain_sum(envs, "drain_rewind"), eps))
            for uk, ut in zip(UNIQUE_KEYS, UNIQUE_TERMS):  # (the life's pay of each unique term, ended lives)
                if getattr(cfg, uk) > 0:
                    row[f"ep_{ut}_mean"] = _nanmean([e[ut] for e in eps])
            rd = _drain_sum(envs, "drain_relay")
            row.update({"relay/activations": rd["activations"], "relay/exits": rd["exits"], "relay/masked": rd["masked"],
                        "relay/term_relay_mean": _nanmean([e["term_relay"] for e in eps]),
                        "ep_exit_frac": _nanmean([float(e["end"] == "relay_exit") for e in eps])})
            row["waiting_frac"] = rs["waiting_frac"]
            fh = fr["frames_hist"]
            fps_step = fr["frames"] / max(fr["steps"], 1)
            row.update({"frames_per_step": fps_step, "frames_total": frames_total,
                        "fps": fps_step * row["sps"],
                        "frames_p50": _hist_percentile(fh, 0.5), "frames_p90": _hist_percentile(fh, 0.9),
                        "frames_p99": _hist_percentile(fh, 0.99),
                        "frames_at_cap": float(fh[cfg.max_frames]) / max(fr["steps"], 1)
                        if fh.size > cfg.max_frames else 0.0})
            row.update({"mask_fallback_events": len(fb["events"]), "mask_fallback_envs": fb["active"],
                        "mask_fallback_frac": fb["fallback_steps"] / max(fb["steps"], 1),
                        "ep_fallback_mean": _nanmean([e["fallback_count"] for e in eps])})
            row.update({f"act_{a}": float(rs["actions"][i]) for i, a in enumerate(ACTIONS)})
            row.update({"best_badges": best["badges"], "best_flags": best["flags"], "champions": best["champions"]})
            row.update({f"t_{p}": v for p, v in rs["timing"].items()})
            if not all(math.isfinite(row[k]) for k in ("pg_loss", "v_loss", "entropy")):
                raise FloatingPointError(f"non-finite loss at update {update}: {row}")
            logger.log(row, global_step)
            event_log.log(envs, eps, update, global_step)
            last_row = row
            print(f"upd {update:5d} | step {global_step / 1e6:8.2f}M | sps {row['sps'] / 1e3:5.1f}k "
                  f"(roll {row['sps_rollout'] / 1e3:5.1f}k env {row['sps_env'] / 1e3:5.1f}k, upd {us['seconds']:.2f}s) | "
                  f"fps {row['fps'] / 1e6:5.2f}M ({row['frames_per_step']:5.1f}/step, wait {row['waiting_frac']:.3f}) | "
                  f"r {rs['reward_raw']:.4f} | ep {len(eps):3d} ret {row['ep_return_mean']:7.2f}/{row['ep_return_max']:7.2f} "
                  f"badges {row['ep_badges_mean']:.2f}/{row['ep_badges_max']:.0f} maps {row['ep_maps_mean']:5.1f} | "
                  f"ent {us['entropy']:.3f} kl {us['approx_kl']:.4f} clip {us['clipfrac']:.3f} "
                  f"ev {us['explained_var']:.3f} v {us['v_loss']:.4f} | best b{best['badges']} f{best['flags']}"
                  + (f" | rw {row['rw_rewinds']}/{row['rw_captures']} pal {row['rw_rewinds_per_life']:.2f}"
                     if cfg.rewind else ""),
                  flush=True)
            if gpu_memory_lines > 0:
                gpu_memory_lines -= 1
                print_gpu_memory(device, f"after update {update}")
            if len(fb["events"]) > max(1.0, 0.01 * len(eps)):
                print(f"WARNING mask fallback: {len(fb['events'])} activations this update "
                      f"({fb['active']} envs in fallback, {row['mask_fallback_frac']:.4f} of steps; "
                      f"see mask_fallback.csv)", flush=True)
            if improved:
                checkpoint("best.pt")
            if update % cfg.checkpoint_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        print("interrupted: saving a checkpoint", flush=True)
    finally:
        if learner is not None and learner.busy:  # let a running update finish (the model includes it)
            try:
                learner.wait()
                update += 1
                global_step += steps_per_update
            except Exception as e:  # noqa: BLE001
                print(f"learner failed while stopping: {e!r}", flush=True)
        if update > 0 and not os.path.exists(os.path.join(run_dir, f"ckpt_{update:06d}.pt")):
            checkpoint()
        logger.close()
        if frames_hist.size:
            with open(os.path.join(run_dir, "frames_hist.json"), "w") as f:
                json.dump({"frames_per_step_count": frames_hist.tolist(), "max_frames": cfg.max_frames}, f)
        for e in envs:
            e.close()
    return last_row


if __name__ == "__main__":
    main(sys.argv[1:])
