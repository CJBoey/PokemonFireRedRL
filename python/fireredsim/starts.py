"""Start states made from input scripts (``PokeVecEnv(start=ScriptStart(...))``) and from
flash saves (``PokeVecEnv(start=SaveStart("game.sav"))``).

A script start is replayed from power-on by every worker (once per timer-1
sequence, then cached), so it does not depend on a snapshot file — snapshots are
only valid for the executable that wrote them, input scripts are valid for every
build. The built-in ``"bedroom"`` start is the script
worker/new_game_to_pallet.txt.

A save start powers the game on with a 128 KiB flash save and continues it: a fixed,
rule-based input driver in the worker taps START through the intro and title screen,
chooses CONTINUE on the main menu, ends the Quest Log recap ("Previously on your
quest...") with B if the save has one, and hands the env over at the first frame
boundary where the player can move in the overworld (worker/envpool.h). Like the
bedroom it is cached per worker for each (save contents, timer-1 values).
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .actions import buttons_to_str, parse_script

FLASH_SIZE = 128 * 1024


@dataclass
class ScriptStart:
    """Input script(s) (lines ``START END BUTTONS``, logical frames from power-on; several are
    merged) replayed from power-on.

    timer1: fixed timer-1 values (e.g. ``[0x9BF5, 0x6518]`` for the oracle's scripts); None =
        each episode's values from its seed (the RNG seed / trainer ID then differ per episode,
        as with ``"bedroom"``; one replay per new seed, cached per worker).
    frames: frames to run (default: up to the end of the last range).
    """
    scripts: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]
    timer1: Optional[Sequence[int]] = None
    frames: Optional[int] = None

    def paths(self) -> List[str]:
        if isinstance(self.scripts, (str, os.PathLike)):
            return [os.fspath(self.scripts)]
        return [os.fspath(p) for p in self.scripts]

    def render(self) -> str:
        """The merged script with the worker's directives (#! frames, #! timer1)."""
        ranges = []
        for p in self.paths():
            with open(p) as f:
                ranges += parse_script(f.read())
        end = max((b for _, b, _ in ranges), default=0)
        n = int(self.frames) if self.frames is not None else end
        buttons = np.zeros(max(n, end), dtype=np.uint16)
        for a, b, m in ranges:
            buttons[a:b] |= m
        buttons = buttons[:n]
        lines = [f"# merged by fireredsim.ScriptStart from {', '.join(self.paths())}", f"#! frames {n}"]
        if self.timer1 is not None:
            vals = [int(v) for v in self.timer1]
            if not 1 <= len(vals) <= 8 or any(not 0 <= v <= 0xFFFF for v in vals):
                raise ValueError("timer1: 1..8 values of 16 bits")
            lines.append("#! timer1 " + ",".join(hex(v) for v in vals))
        f = 0
        while f < n:
            if buttons[f] == 0:
                f += 1
                continue
            g = f
            while g < n and buttons[g] == buttons[f]:
                g += 1
            lines.append(f"{f} {g} {buttons_to_str(int(buttons[f]))}")
            f = g
        return "\n".join(lines) + "\n"


def _timer1_tuple(values) -> Tuple[int, ...]:
    vals = tuple(int(v) for v in np.atleast_1d(np.asarray(values)).reshape(-1))
    if not 1 <= len(vals) <= 8 or any(not 0 <= v <= 0xFFFF for v in vals):
        raise ValueError("timer1: 1..8 values of 16 bits")
    return vals


@dataclass(frozen=True)
class SaveStart:
    """A FireRed flash save (a 128 KiB ``.sav``, e.g. from mGBA)
    continued from power-on (worker/envpool.h, ``PN_TEMPLATE_SAVE``).

    Every reset powers the game on with the save in the flash chip (exactly as
    ``fireredsim-cli --sav``), and a fixed input driver taps START on the intro and title screens,
    selects CONTINUE on the main menu (A), ends the Quest Log recap with B if there is one, and
    stops at the first frame boundary at which the player is free to move in the overworld (no
    script, no fade, field controls unlocked, no button held). Each worker caches the result per
    (save contents, timer-1 values); a new pair costs one boot (~370-550 frames, a few ms).

    timer1: fixed timer-1 values (the first one seeds the RNG when START is pressed on the title
        screen; the trainer ID comes from the save); None = each episode's values from its seed,
        so the RNG differs per episode as with ``"bedroom"``.

    Usable as ``start=`` of :class:`~fireredsim.PokeVecEnv` and per
    reset: ``env.reset(options={"start": SaveStart(...)})`` (sticky: the env keeps that start for
    its later autoresets). Saves written by another executable (mGBA) are fine;
    see python/README.md for the limits.
    """
    path: Union[str, os.PathLike]
    timer1: Optional[Sequence[int]] = None

    def __post_init__(self):
        object.__setattr__(self, "path", os.path.abspath(os.fspath(self.path)))
        if self.timer1 is not None:
            object.__setattr__(self, "timer1", _timer1_tuple(self.timer1))

    def check(self) -> None:
        """Raises if the file is missing or shorter than 128 KiB (longer files are truncated to
        128 KiB, as fireredsim-cli --sav does: mGBA appends RTC data to some saves)."""
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"save file not found: {self.path}")
        size = os.path.getsize(self.path)
        if size < FLASH_SIZE:
            raise ValueError(f"{self.path}: a FireRed save is 128 KiB of flash, the file has {size} bytes")

    def record(self, timer1: Optional[Sequence[int]] = None, worker_path: Optional[str] = None) -> Dict[str, Any]:
        """Runs this save start once outside the env (``fireredsim-worker --save-start``) and returns
        what the driver did: ``script`` (its buttons from power-on as an input script,
        with a ``#! frames N`` directive), ``frames``, ``state_hash`` (equal to
        ``env.state_hash()`` of an env reset to this start with these timer-1 values, same worker
        executable), ``map`` (group, num), ``pos`` (x, y), ``timer1``, ``save_hash``, ``build_id``.

        ``timer1`` defaults to this start's fixed values, else ``(0,)``. For the values of an episode
        use ``fireredsim.seeding.timer1_for_seed(info["episode_seed"][i])``. The script replays with
        ``fireredsim-cli --sav SAVE --seed V1,V2 --input SCRIPT --frames N`` (or ``fireredsim-worker
        --replay`` with the same options, which prints a hash comparable with the worker's).
        """
        import tempfile
        from ._paths import find_worker
        self.check()
        t1 = _timer1_tuple(timer1) if timer1 is not None else (self.timer1 or (0,))
        worker = find_worker(worker_path)
        with tempfile.TemporaryDirectory(prefix="fireredsim-savestart-") as tmp:
            out = os.path.join(tmp, "start.txt")
            r = subprocess.run([worker, "--save-start", self.path, "--seed", ",".join(hex(v) for v in t1),
                                "--script-out", out, "--quiet-faults"], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"save start failed: {r.stderr.strip()[-2000:]}")
            info = json.loads(r.stdout.strip().splitlines()[-1])
            with open(out) as f:
                info["script"] = f.read()
        info["state_hash"] = int(info["state_hash"], 16)
        info["save_hash"] = int(info["save_hash"], 16)
        info["timer1"] = tuple(info["timer1"])
        info["map"] = tuple(info["map"])
        info["pos"] = tuple(info["pos"])
        return info
