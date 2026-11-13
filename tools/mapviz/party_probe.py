#!/usr/bin/env python3
"""Party peek of a running training (the Party page, tools/mapviz/www/party.html; also usable from a shell).

Reads the trainer's worker shared memory read-only (like serve.py), with the trainer's own record layout and pokerl
tables (its PYTHONPATH), and prints one JSON document:

    python3 tools/mapviz/party_probe.py snapshot [--pid PID]          # every env's party, badges, map, scope
    python3 tools/mapviz/party_probe.py sample --secs 60 [--hz 10]    # behaviour counters over a sampling window

The trainer does no work for it. ``snapshot`` copies the records once (~2.4 MB for 512 envs). ``sample`` copies them
``--hz`` times a second for ``--secs`` seconds on one core and compares consecutive copies, so it sees only what
lasts at least one sampling interval (a party reorder undone within 0.1 s, or a prompt answered at once, can be
missed; counts are lower bounds).

Relay scope (record-only proxy): the first relay scope whose area holds the env's map, whose entry condition holds and
whose clear condition does not (pokerl.relay.scopes), else none. The trainer's own per-life scope status is not in
the records; this proxy differs from it only for a life inside an area it has not activated (it never reached the
scope's home with the entry condition) or one that has cleared a scope and rewound.

Snapshot extras per env (when the trainer's records have them): HMs as three 7-bit masks (bit i = HM0i+1: in the bag,
its move known by a party mon, a party mon can learn it), the consumable pockets (items, balls, berries) as [item, qty]
pairs, and game counters (game_stats_priv, cumulative over the save's whole lineage: a relay baton or a rewind keeps
the counts of the lives before it).

Sample groups are keyed by badge count, scope and "HM gap" (1: an HM in the bag whose move no party mon knows).
Sample counters (polls = env records sampled):
  polls, battle, trainer_battle, field (OVERWORLD outside battle), party_menu, bag, pc_box, tm_case, start_menu,
  shift_prompt (waiting on the battle YES/NO box), lead_weaker_field (outside battle, the lead's level below the
  party's best);
  battles (whose start was seen), per kind (trainer / wild): battles_k, lead_weaker_k (the first active mon below the
  party's best level), switched_up_k (a weaker lead, still alive, was switched out for a higher-level mon),
  switch_train_k (that, and the lead gained EXP by the end: the switch-training pattern), lead_exp_k (a weaker lead
  gained EXP), exp_split_k (2+ mons gained EXP), lost_trainer (every mon fainted), wild_ko / wild_caught / wild_lost
  / wild_ran (one outcome per wild battle; per map in each group's wild_maps "map:outcome"), won_k (trainer: its defeated flag got set, counted over the
  trainer_flag_trainer battles whose trainer was not beaten before; wild: some mon gained EXP, i.e. a KO), switch_vol_k / switch_forced_k (the active
  mon changed while the previous one had HP / had fainted);
  reorder (same party set, other order, outside battle), reorder_weak_lead (after it the lead is below the best
  level), pc_deposit / pc_withdraw (the party set changed on the PC box screen), added_outside_pc (catches, gifts),
  moves_learned / moves_learned_tm (a mon gained a move; the move is a TM/HM move), evolutions (same mon, new
  species); restarts (play time went back: a new life, a rewind or a relay handoff; that env's comparison of the
  two copies is skipped, so restored states never count as events); envs_* = distinct envs that did it;
  items (each group's "items" maps "kind:item" to units): buy (money went down and only bag counts went up, on the
  mart screen), sell (the reverse), throw (a ball count fell by 1 in battle), use_battle / use_field (one consumable
  fell by exactly 1 with money and party unchanged: in battle, or outside battle on the bag / party menu screen;
  a drop on the plain overworld is a restore, not a use), and outside battle the drops that are not uses: give (a
  party mon now holds one more of it: GIVE), store (the item PC holds one more), toss (the copy before waited on the
  bag's quantity / yes-no box; a toss answered within one copy interval counts as a use); buy_events, sell_events,
  throws, use_battle, use_field, give, store, toss,
  box_auto (box count rose outside the PC: a catch with a full party); sk_<event> = polls whose skill_events_priv bit
  was set (one-step pulses, so only ~hz / steps-per-second of them are seen: compare events, not totals);
  dwell / dwell_battle (each group's map -> polls, polls in battle).
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve as S  # noqa: E402

KINDS = ("trainer", "wild")
TRAINER_FLAGS_START = 0x500  # game/include/constants/flags.h: flag TRAINER_FLAGS_START + id = trainer id beaten
ITEM_HM01 = 339  # game/include/constants/items.h (HM01-HM07 = 339-345)
BALL_LAST = 12  # ITEM_MASTER_BALL (1) .. ITEM_PREMIER_BALL (12)
BAG_POCKETS = ("items", "balls", "berries")  # the consumable pockets
GAME_STATS = (("shopped", 38), ("pokecenter", 15), ("captures", 11), ("wild_battles", 8), ("trainer_battles", 9),
              ("evolved", 14), ("used_cut", 18))  # game/include/constants/game_stat.h


def item_names(rl):
    """item id -> name (without ITEM_) from the trainer's game/include/constants/items.h, else {}."""
    out = {}
    try:
        with open(os.path.join(rl, "..", "game", "include", "constants", "items.h")) as f:
            for ln in f:
                w = ln.split()
                if len(w) == 3 and w[0] == "#define" and w[1].startswith("ITEM_") and w[2].isdigit():
                    out.setdefault(int(w[2]), w[1][5:])
    except OSError:
        pass
    return out


def trainer_paths(pid):
    """(rl directory, python directory) of the trainer (its PYTHONPATH, relative to its cwd)."""
    cwd = os.readlink(f"/proc/{pid}/cwd")
    env = dict(kv.split(b"=", 1) for kv in open(f"/proc/{pid}/environ", "rb").read().split(b"\0") if b"=" in kv)
    entries = [""] + env.get(b"PYTHONPATH", b"").decode(errors="replace").split(":") + ["rl", "python"]
    rl = py = None
    for e in entries:
        d = os.path.normpath(os.path.join(cwd, e))
        if rl is None and os.path.isfile(os.path.join(d, "pokerl", "tables.py")):
            rl = d
        if py is None and os.path.isfile(os.path.join(d, "fireredsim", "obs_dtype.py")):
            py = d
    if rl is None or py is None:
        raise LookupError("the trainer's PYTHONPATH has no pokerl / fireredsim")
    return rl, py


class Probe:
    def __init__(self, pid):
        self.pid = pid
        rl, py = trainer_paths(pid)
        sys.path[:0] = [rl, py]
        from fireredsim import obs_dtype as OD
        from pokerl import events as EV
        from pokerl import reward as R
        from pokerl import tables as T
        from pokerl.relay import scopes as SC
        self.OD, self.EV, self.R, self.T = OD, EV, R, T
        self.scopes = SC.scope_set()
        self.area = self.scopes.area_table()  # bool [n_scopes, MAP_COUNT + 1]
        self.run = S.Run(pid)
        self.items = item_names(rl)
        fields = OD.OBS_DTYPE.names
        self.has_stats = "game_stats_priv" in fields
        self.skill_names = list(getattr(R, "SKILL_EVENTS", ())) if "skill_events_priv" in fields else []
        self.hm_moves = np.asarray(getattr(T, "HM_MOVES", []), np.int64)
        self.hm_learn = np.asarray(getattr(T, "HM_LEARN", [0]), np.int64)
        self.views = []
        for mm, _, _ in self.run.groups:
            h = np.frombuffer(mm, S.SHM_HEADER, 1)[0]
            if int(h["obs_size"]) != OD.OBS_DTYPE.itemsize:
                raise LookupError(f"record size {int(h['obs_size'])} != the trainer's obs_dtype "
                                  f"{OD.OBS_DTYPE.itemsize}")
            self.views.append(np.frombuffer(mm, OD.OBS_DTYPE, int(h["num_envs"]), int(h["off_obs"])))
        self.n = sum(len(v) for v in self.views)
        scr, wait = list(OD.SCREEN_NAMES), list(OD.WAIT_CTX_NAMES)
        self.SCR = {k: scr.index(k) for k in ("OVERWORLD", "PARTY_MENU", "BAG", "PC_BOX", "TM_CASE", "START_MENU",
                                               "MART")}
        self.W_YESNO = wait.index("BATTLE_YESNO")
        self.W_BAG_TOSS = [wait.index(w) for w in ("BAG_QUANTITY", "BAG_YESNO") if w in wait]
        self.is_tm_move = np.zeros(T.MOVES_COUNT + 1, bool)
        self.is_tm_move[np.asarray(T.TM_MOVES)] = True

    def read(self):
        return np.concatenate([v.copy() for v in self.views])

    def names(self):
        T = self.T
        return {"species": list(T.SPECIES_NAMES), "moves": list(T.MOVE_NAMES),
                "maps": ["?"] + [m[1][4:] for m in T.MAPS], "scopes": [s.name for s in self.scopes.scopes],
                "tm_moves": [int(m) for m in T.TM_MOVES], "hm_moves": self.hm_moves.tolist(),
                "items": {str(k): v for k, v in self.items.items()}, "stats": [k for k, _ in GAME_STATS]}

    def bag(self, a):
        """[n, 400] quantity of each item id in the consumable pockets."""
        ids = np.concatenate([a[p]["id"] for p in BAG_POCKETS], 1).astype(np.int64)
        qty = np.concatenate([a[p]["qty"] for p in BAG_POCKETS], 1).astype(np.int64)
        n = len(a)
        flat = (np.arange(n)[:, None] * 400 + np.minimum(ids, 399)).ravel()
        q = np.bincount(flat, qty.ravel(), n * 400).astype(np.int64).reshape(n, 400)
        q[:, 0] = 0
        return q

    def hms(self, a):
        """(in bag, taught, learnable) 7-bit masks per env (bit i = HM0i+1)."""
        n, k = len(a), len(self.hm_moves)
        if not k:
            z = np.zeros(n, np.int64)
            return z, z, z
        t = a["tmhm"]["id"].astype(np.int64)
        w = 1 << np.arange(k)
        bag = ((t[:, :, None] == ITEM_HM01 + np.arange(k)).any(1) * w).sum(1)
        p = a["party"]
        live = (np.arange(6)[None] < np.minimum(a["party_count"].astype(np.int64), 6)[:, None]) & (p["is_egg"] == 0)
        mv = np.where(live[:, :, None], p["moves"], 0).astype(np.int64)
        taught = ((mv[:, :, :, None] == self.hm_moves).any((1, 2)) * w).sum(1)
        sp = np.where(live, p["species"], 0).astype(np.int64)
        learn = np.bitwise_or.reduce(np.where(live, self.hm_learn[np.minimum(sp, len(self.hm_learn) - 1)], 0), 1)
        return bag, taught, learn

    def context(self, a):
        """(in game, badge count, dense map index, scope proxy) per env."""
        ing = (a["valid"] & 1) != 0
        b = np.unpackbits(a["badges"][:, None], axis=1).sum(1).astype(np.int64)
        mi = self.R.MAP_INDEX[a["map_group"].astype(np.int64) * 256 + a["map_num"]]
        ev = self.EV.observe(a)
        ok = self.area[:, mi].T & self.scopes.entry_holds(ev) & ~self.scopes.clear_holds(ev)  # [n, scopes]
        sc = np.where(ok.any(1), ok.argmax(1), -1)
        return ing, b, mi, sc

    def snapshot(self):
        a = self.read()
        ing, b, mi, sc = self.context(a)
        p = a["party"]
        hb, ht, hl = self.hms(a)
        q = self.bag(a)
        gs = a["game_stats_priv"][:, [i for _, i in GAME_STATS]].astype(np.int64) if self.has_stats else None
        envs = []
        for e in range(self.n):
            if not ing[e]:
                envs.append([e, -1])
                continue
            cnt = min(int(a["party_count"][e]), 6)
            party = []
            for j in range(cnt):
                m = p[e, j]
                party.append([int(m["species"]), int(m["level"]), int(m["hp"]), int(m["max_hp"]), int(m["status"]),
                              [int(x) for x in m["moves"]], int(m["is_egg"])])
            battle = 0 if not a["in_battle"][e] else (2 if a["battle_type_flags"][e] & 8 else 1)
            held = np.flatnonzero(q[e])
            envs.append([e, int(b[e]), int(a["badges"][e]), int(mi[e]), int(sc[e]), battle, int(a["money"][e]),
                         int(a["box_counts"][e].sum()), party, [int(hb[e]), int(ht[e]), int(hl[e])],
                         [[int(i), int(q[e, i])] for i in held], gs[e].tolist() if gs is not None else None])
        return {"kind": "snapshot", "run": self.run.name, "pid": self.pid, "t": time.time(), "n": self.n,
                "names": self.names(), "envs": envs}

    # ---- behaviour sample ----------------------------------------------------------------------------------

    def sample(self, secs, hz):
        C = collections.defaultdict(collections.Counter)  # (badges, scope) -> counters
        E = collections.defaultdict(lambda: collections.defaultdict(set))  # (badges, scope) -> name -> envs
        bat = {}  # env -> the battle being followed
        W = collections.defaultdict(collections.Counter)  # (badges, scope) -> "map:outcome" -> wild battles
        I = collections.defaultdict(collections.Counter)  # key -> "kind:item" -> units
        gid = {}  # key -> row of the per-group arrays below
        nmap = len(self.T.MAPS) + 1
        skn = self.skill_names
        D = np.zeros((64, 2, nmap), np.int64)  # group -> (polls, polls in battle) per map
        SK = np.zeros((64, max(len(skn), 1)), np.int64)  # group -> polls with each skill event bit set
        prev = None
        sweeps, t0, cpu0 = 0, time.time(), time.process_time()
        SCR = self.SCR
        while time.time() - t0 < secs:
            ts = time.time()
            a = self.read()
            sweeps += 1
            ing, b, mi, sc = self.context(a)
            hb, ht, _ = self.hms(a)
            key = list(zip(b.tolist(), sc.tolist(), ((hb & ~ht) != 0).astype(int).tolist()))
            p = a["party"]
            cnt = np.minimum(a["party_count"].astype(np.int64), 6)
            live = np.arange(6)[None, :] < cnt[:, None]
            pid = np.where(live, p["pid_priv"], 0).astype(np.int64)
            lvl = np.where(live & (p["is_egg"] == 0), p["level"], 0).astype(np.int64)
            inb = a["in_battle"] != 0
            scr = a["screen"]
            best = lvl.max(1)
            lead_weak = (cnt > 0) & (lvl[:, 0] < best)
            polls = {
                "polls": ing, "battle": ing & inb, "trainer_battle": ing & inb & ((a["battle_type_flags"] & 8) != 0),
                "field": ing & ~inb & (scr == SCR["OVERWORLD"]), "party_menu": ing & ~inb & (scr == SCR["PARTY_MENU"]),
                "bag": ing & (scr == SCR["BAG"]), "pc_box": ing & (scr == SCR["PC_BOX"]),
                "tm_case": ing & (scr == SCR["TM_CASE"]), "start_menu": ing & (scr == SCR["START_MENU"]),
                "shift_prompt": ing & inb & (a["waiting_for_input"] != 0) & (a["wait_ctx"] == self.W_YESNO),
                "lead_weaker_field": ing & ~inb & lead_weak,
            }
            for name, m in polls.items():
                for e in np.flatnonzero(m):
                    C[key[e]][name] += 1
            for k in set(key):
                if k not in gid:
                    gid[k] = len(gid)
                    if gid[k] >= len(D):
                        D = np.concatenate([D, np.zeros_like(D)])
                        SK = np.concatenate([SK, np.zeros_like(SK)])
            g = np.fromiter((gid[k] for k in key), np.int64, len(key))
            np.add.at(D, (g[ing], inb[ing].astype(np.int64), np.minimum(mi[ing], nmap - 1)), 1)
            if skn:
                bits = (a["skill_events_priv"][ing, None].astype(np.int64) >> np.arange(len(skn))) & 1
                np.add.at(SK, g[ing], bits)
            q = self.bag(a)
            money = a["money"].astype(np.int64)
            box = a["box_counts"].astype(np.int64).sum(1)
            pt = (a["play_hours"].astype(np.int64) * 3600 + a["play_minutes"].astype(np.int64) * 60
                  + a["play_seconds_priv"].astype(np.int64))
            if prev is not None:
                pa, ping, pinb, ppid, pcnt, pscr, ppt, pq, pmoney, pbox = prev
                # a new life, a rewind or a relay handoff restored the game state: play time went back, or a mon in the
                # same slot lost EXP
                same0 = (pid == ppid) & (pid != 0)
                jump = ping & ing & ((pt < ppt) | (same0 & (p["exp"] < pa["party"]["exp"])).any(1))
                for e in np.flatnonzero(jump):
                    C[key[e]]["restarts"] += 1
                    bat.pop(e, None)
                both = ing & ping & ~jump
                # party set / order changes outside battle
                ch = both & ~inb & ~pinb & ((pid != ppid).any(1) | (cnt != pcnt))
                for e in np.flatnonzero(ch):
                    k = key[e]
                    cur, old = [x for x in pid[e] if x], [x for x in ppid[e] if x]
                    if sorted(cur) == sorted(old):
                        C[k]["reorder"] += 1
                        E[k]["reorder"].add(e)
                        if lead_weak[e]:
                            C[k]["reorder_weak_lead"] += 1
                        continue
                    gone, new = set(old) - set(cur), set(cur) - set(old)
                    if scr[e] == SCR["PC_BOX"] or pscr[e] == SCR["PC_BOX"]:
                        C[k]["pc_deposit"] += len(gone)
                        C[k]["pc_withdraw"] += len(new)
                        E[k]["pc"].add(e)
                    elif len(new) == 1 and not gone:  # a catch (named after the battle) or a gift
                        C[k]["added_outside_pc"] += 1
                    else:  # a whole new party outside the PC: a relay handoff, not an action
                        C[k]["restarts"] += 1
                        bat.pop(e, None)
                self._items(both, a, pa, q, pq, money, pmoney, box, pbox, pid, ppid, inb, pinb, scr, pscr, key, C, E, I)
                # moves learned / evolutions (same slot, same mon)
                same = both[:, None] & (pid == ppid) & (pid != 0)
                mv, pmv = p["moves"].astype(np.int64), pa["party"]["moves"].astype(np.int64)
                newm = same[:, :, None] & (mv != 0) & ~(mv[:, :, :, None] == pmv[:, :, None, :]).any(-1)
                evo = same & (p["species"] != pa["party"]["species"]) & (p["is_egg"] == 0) & (pa["party"]["is_egg"] == 0)
                for e in np.flatnonzero(newm.any((1, 2)) | evo.any(1)):
                    k = key[e]
                    ms = mv[e][newm[e]]
                    C[k]["moves_learned"] += int(ms.size)
                    C[k]["moves_learned_tm"] += int(self.is_tm_move[np.minimum(ms, self.T.MOVES_COUNT)].sum())
                    C[k]["evolutions"] += int(evo[e].sum())
                # battles
                for e in np.flatnonzero(both & (inb | pinb)):
                    self._battle(e, a, pa, inb[e], pinb[e], key[e], bat, C, W, int(mi[e]))
            prev = (a, ing, inb, pid, cnt, scr, pt, q, money, box)
            time.sleep(max(0.0, 1.0 / hz - (time.time() - ts)))
        el = time.time() - t0
        groups = []
        for k, r in gid.items():
            for i, nm in enumerate(skn):
                if SK[r, i]:
                    C[k]["sk_" + nm] += int(SK[r, i])
        for k in sorted(set(C) | set(E) | set(W)):
            c = dict(C[k])
            r = gid.get(k)
            dw = D[r] if r is not None else np.zeros((2, nmap), np.int64)
            dwt = dw[0] + dw[1]
            for name, s in E[k].items():
                c["envs_" + name] = len(s)
            groups.append({"b": k[0], "s": k[1], "g": k[2], "c": c, "wild_maps": dict(W[k]), "items": dict(I[k]),
                           "dwell": {str(m): int(dwt[m]) for m in np.flatnonzero(dwt)},
                           "dwell_battle": {str(m): int(dw[1, m]) for m in np.flatnonzero(dw[1])}})
        return {"kind": "sample", "run": self.run.name, "pid": self.pid, "t": time.time(), "n": self.n, "secs": el,
                "sweeps": sweeps, "hz": sweeps / max(el, 1e-9), "cpu_s": time.process_time() - cpu0,
                "names": {"scopes": [s.name for s in self.scopes.scopes], "maps": ["?"] + [m[1][4:] for m in self.T.MAPS],
                          "items": {str(k): v for k, v in self.items.items()}, "skills": skn},
                "groups": groups}

    def _items(self, both, a, pa, q, pq, money, pmoney, box, pbox, pid, ppid, inb, pinb, scr, pscr, key, C, E, I):
        """Purchases, sales, ball throws, consumable uses and catches sent to the box between two copies."""
        SCR = self.SCR
        dq = q - pq
        dm = money - pmoney
        up, down = (dq > 0).sum(1), (dq < 0).sum(1)
        mart = (scr == SCR["MART"]) | (pscr == SCR["MART"])
        same_party = (pid == ppid).all(1)
        menu = np.isin(scr, (SCR["BAG"], SCR["PARTY_MENU"])) | np.isin(pscr, (SCR["BAG"], SCR["PARTY_MENU"]))
        btl = inb | pinb
        W = self.W_BAG_TOSS
        for e in np.flatnonzero(both & ((up + down) > 0)):
            k = key[e]
            if mart[e] and dm[e] < 0 and up[e] and not down[e]:
                kind = "buy"
            elif mart[e] and dm[e] > 0 and down[e] and not up[e]:
                kind = "sell"
            elif dm[e] == 0 and same_party[e] and down[e] == 1 and not up[e] and dq[e].sum() == -1:
                i = int(np.flatnonzero(dq[e])[0])
                if btl[e]:
                    kind = "throw" if i <= BALL_LAST else "use_battle"
                elif menu[e] and i > BALL_LAST:
                    held = lambda r: int((r["party"]["held_item"][:min(int(r["party_count"]), 6)] == i).sum())
                    pc = lambda r: int(r["pc_items"]["qty"][r["pc_items"]["id"] == i].sum())
                    kind = ("give" if held(a[e]) > held(pa[e]) else "store" if pc(a[e]) > pc(pa[e])
                            else "toss" if pa["waiting_for_input"][e] and pa["wait_ctx"][e] in W else "use_field")
                else:
                    continue
            else:
                continue
            C[k]["throws" if kind == "throw" else kind + "_events" if kind in ("buy", "sell") else kind] += 1
            E[k][kind].add(e)
            for i in np.flatnonzero(dq[e]):
                I[k][f"{kind}:{int(i)}"] += abs(int(dq[e, i]))
        auto = both & (box > pbox) & (scr != SCR["PC_BOX"]) & (pscr != SCR["PC_BOX"])
        for e in np.flatnonzero(auto):
            C[key[e]]["box_auto"] += int(box[e] - pbox[e])

    def _battle(self, e, a, pa, inb, pinb, k, bat, C, W, mi):
        r = a[e]
        cnt = min(int(r["party_count"]), 6)
        mons = {int(m["pid_priv"]): m for m in r["party"][:cnt] if m["pid_priv"]}
        if inb and not pinb:  # a battle starts: follow it
            bat[e] = {"kind": "trainer" if r["battle_type_flags"] & 8 else "wild", "lead": None, "act": None,
                      "exp0": None, "weak": False, "up": False, "tid": None, "pids0": set(mons),
                      "box0": int(r["box_counts"].sum()), "fainted": False, "map": mi}
        B = bat.get(e)
        if B is None:
            return
        if mons and all(int(m["hp"]) == 0 for m in mons.values() if not m["is_egg"]):
            B["fainted"] = True  # (every mon down: a loss; the whiteout heals soon after the battle)
        if inb:
            act = None
            if r["valid"] & 8:
                for bt in r["battlers"]:
                    if bt["is_player_side"] and bt["is_active"]:
                        act = int(bt["personality_priv"])
                        break
            if B["kind"] == "trainer" and B["tid"] is None and r["trainer_id_priv"]:
                f = TRAINER_FLAGS_START + int(r["trainer_id_priv"])
                if not r["flags_priv"][f >> 3] & (1 << (f & 7)):  # (a rematch of a beaten trainer: no win signal)
                    B["tid"] = f
            if act is None or act not in mons:
                return
            if B["lead"] is None:
                lv = [int(m["level"]) for m in mons.values() if not m["is_egg"]]
                B["lead"], B["act"] = act, act
                B["exp0"] = {q: int(m["exp"]) for q, m in mons.items()}
                B["lv"] = {q: int(m["level"]) for q, m in mons.items()}
                B["weak"] = int(mons[act]["level"]) < max(lv)
                return
            if act != B["act"]:
                old = mons.get(B["act"])
                forced = old is None or int(old["hp"]) == 0
                C[k][("switch_forced_" if forced else "switch_vol_") + B["kind"]] += 1
                if not forced and B["act"] == B["lead"] and B["weak"] and B["lv"].get(act, 0) > B["lv"][B["lead"]]:
                    B["up"] = True
                B["act"] = act
            return
        # the battle ended
        del bat[e]
        if B["lead"] is None:
            return
        t = B["kind"]
        gained = [q for q, x0 in B["exp0"].items() if q in mons and int(mons[q]["exp"]) > x0]
        lead_exp = B["lead"] in gained
        C[k]["battles_" + t] += 1
        C[k]["battles"] += 1
        # won (the enemy party is cleared when a battle ends and the outcome byte lasts only the end text, a few ms at
        # the trainer's speed): a trainer battle sets the trainer's defeated flag; a wild mon gives EXP only when KO'd
        if t == "trainer":
            f = B["tid"]
            C[k]["won_" + t] += f is not None and bool(r["flags_priv"][f >> 3] & (1 << (f & 7)))
            C[k]["trainer_flag_" + t] += f is not None
            C[k]["lost_" + t] += B["fainted"]
        else:
            # one outcome per wild battle: KO (EXP: only a KO gives EXP), caught (a new party mon or a box count
            # increase at the end), lost (every mon fainted), else ran (or another escape: Teleport, Roar, ...)
            caught = bool(set(mons) - B["pids0"]) or int(r["box_counts"].sum()) > B["box0"]
            out = "ko" if gained else "caught" if caught else "lost" if B["fainted"] else "ran"
            C[k]["won_" + t] += out == "ko"
            C[k][f"wild_{out}"] += 1
            W[k][f"{B['map']}:{out}"] += 1
        C[k]["exp_split_" + t] += len(gained) >= 2
        if B["weak"]:
            C[k]["lead_weaker_" + t] += 1
            C[k]["switched_up_" + t] += B["up"]
            C[k]["lead_exp_" + t] += lead_exp
            C[k]["switch_train_" + t] += B["up"] and lead_exp


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("snapshot", "sample"))
    ap.add_argument("--pid", type=int, default=None, help="trainer pid (default: the newest pokerl.train)")
    ap.add_argument("--secs", type=float, default=60.0, help="sample: seconds to sample")
    ap.add_argument("--hz", type=float, default=10.0, help="sample: record copies per second")
    a = ap.parse_args()
    try:
        pr = Probe(a.pid or S.find_trainer())
        out = pr.snapshot() if a.what == "snapshot" else pr.sample(max(1.0, min(a.secs, 900.0)),
                                                                    max(0.5, min(a.hz, 30.0)))
    except (LookupError, OSError, ValueError) as e:
        out = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
