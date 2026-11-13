"""GBA buttons and the action tables of the RL interface.

Buttons use the GBA KEYINPUT bit layout (active high here), the same values as
input scripts (lines ``START END BUTTONS``: frames [START, END) from power-on hold BUTTONS, e.g.
``5161 5162 START``)::

    A=0x001 B=0x002 SELECT=0x004 START=0x008 RIGHT=0x010 LEFT=0x020 UP=0x040 DOWN=0x080 R=0x100 L=0x200

An action is "the set of buttons held during the step" (all ``frameskip``
frames, or the first ``hold_frames`` of them). Three action types:

``discrete`` (default)
    an index into an action table; ``ACTION_SETS["basic"]`` =
    NOOP, A, B, START, SELECT, UP, DOWN, LEFT, RIGHT (9 actions);
    ``"full"`` adds L, R and B+direction (running once the player has the
    Running Shoes) = 15 actions. Any list of button masks can be passed instead.
``multibinary``
    a 0/1 vector of length 10 in ``BUTTON_ORDER`` (A, B, SELECT, START, RIGHT,
    LEFT, UP, DOWN, R, L), i.e. bit i of the KEYINPUT mask.
``buttons``
    the raw uint16 KEYINPUT mask.

Pressing A+B+START+SELECT together makes the game soft-reset (as on hardware);
the env reports it (``info["soft_reset"]``).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np

A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L = (1 << i for i in range(10))
BUTTON_ORDER = ("A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L")
BUTTONS: Dict[str, int] = {name: 1 << i for i, name in enumerate(BUTTON_ORDER)}
NOOP = 0

ACTION_SETS: Dict[str, List[int]] = {
    "basic": [NOOP, A, B, START, SELECT, UP, DOWN, LEFT, RIGHT],
    "full": [NOOP, A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, B | UP, B | DOWN, B | LEFT, B | RIGHT],
}


def buttons_to_str(mask: int) -> str:
    mask = int(mask)
    if mask == 0:
        return "NONE"
    return "+".join(n for i, n in enumerate(BUTTON_ORDER) if mask >> i & 1)


def parse_buttons(text: str) -> int:
    """``"A"``, ``"B+LEFT"``, ``"NONE"`` -> KEYINPUT mask."""
    text = text.strip().upper()
    if text in ("", "NONE", "NOOP"):
        return 0
    mask = 0
    for part in text.split("+"):
        if part not in BUTTONS:
            raise ValueError(f"unknown button {part!r}")
        mask |= BUTTONS[part]
    return mask


def action_table(action_set: Union[str, Sequence[Union[int, str]]]) -> np.ndarray:
    """The button mask of each discrete action (uint16 array)."""
    if isinstance(action_set, str):
        if action_set not in ACTION_SETS:
            raise ValueError(f"unknown action set {action_set!r} (known: {sorted(ACTION_SETS)})")
        table = ACTION_SETS[action_set]
    else:
        table = [parse_buttons(a) if isinstance(a, str) else int(a) for a in action_set]
    arr = np.asarray(table, dtype=np.uint16)
    if arr.ndim != 1 or arr.size == 0 or np.any(arr > 0x3FF):
        raise ValueError("an action table is a non-empty list of 10-bit button masks")
    return arr


def multibinary_to_buttons(bits: np.ndarray) -> np.ndarray:
    """(..., 10) 0/1 array in BUTTON_ORDER -> uint16 masks."""
    bits = np.asarray(bits)
    if bits.shape[-1] != 10:
        raise ValueError("multibinary actions have 10 entries (A B SELECT START RIGHT LEFT UP DOWN R L)")
    weights = (1 << np.arange(10)).astype(np.uint16)
    return ((bits != 0).astype(np.uint16) * weights).sum(axis=-1).astype(np.uint16)


def parse_script(text: str):
    """Input script text (lines ``START END BUTTONS``) -> list of (start, end, mask)."""
    ranges = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        a, b, btn = line.split()
        ranges.append((int(a), int(b), parse_buttons(btn)))
    return ranges
