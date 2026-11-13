"""The action mask.

Actions (``ACTIONS``, the PokeVecEnv ``action_set``): NOOP, A, B, START, UP, DOWN, LEFT, RIGHT.

* ``waiting_for_input == 0``: NOOP only.
* else every action whose button mask intersects ``input_keys``, NOOP excluded (a decision step offers no idle
  choice);
* A is forbidden in the start menu with the cursor on an entry that never advances the game (POKEDEX, the trainer
  card PLAYER / PLAYER2, OPTION, SAVE: ``wait_ctx == START_MENU`` and ``start_menu_cursor`` in ``START_MENU_NO_A``),
  in the save prompt (``SAVE_YESNO``, ``SAVE_TEXT``) and when the entry under the cursor is RELEASE (``menu_action ==
  ACTION_RELEASE``: the Pokemon Storage System's selected-mon menu and the YES of its "Release this POKeMON?" prompt);
  POKEMON, BAG, EXIT and the Safari Zone's RETIRE stay open;
* a row with no action left (waiting but no maskable key, or only a forbidden A) allows NOOP, so every row allows at
  least one action.

The *mask fallback*'s mask (env.py "Mask fallback") allows every action, NOOP included, with the A rule still
applied. Both masks are computed by pokerl.native (pn_action_mask, pn_fallback_mask); the env decides which one a row
gets. The wait-context values are those of ``enum PnWaitCtx`` in obs/observation.h (checked by rl/tests/test_mask.py).
"""
from __future__ import annotations

import numpy as np

ACTIONS = ["NOOP", "A", "B", "START", "UP", "DOWN", "LEFT", "RIGHT"]
KEY = {"A": 1, "B": 2, "SELECT": 4, "START": 8, "RIGHT": 16, "LEFT": 32, "UP": 64, "DOWN": 128}
ACTION_BUTTONS = np.array([0] + [KEY[a] for a in ACTIONS[1:]], dtype=np.uint16)
NUM_ACTIONS = len(ACTIONS)
A_INDEX = ACTIONS.index("A")

WAIT_START_MENU = 18   # PN_WAIT_START_MENU
WAIT_SAVE_YESNO = 19   # PN_WAIT_SAVE_YESNO
WAIT_SAVE_TEXT = 20    # PN_WAIT_SAVE_TEXT
# start_menu_cursor option ids (enum StartMenuOption, game/src/start_menu.c)
START_MENU_POKEDEX = 0
START_MENU_PLAYER = 3   # the trainer card
START_MENU_SAVE = 4
START_MENU_OPTION = 5
START_MENU_PLAYER2 = 8  # the trainer card entry of the link-room menu
# entries A may not open: they never advance the game (SAVE: 768 frames that persist nothing)
START_MENU_NO_A = (START_MENU_POKEDEX, START_MENU_PLAYER, START_MENU_SAVE, START_MENU_OPTION, START_MENU_PLAYER2)
START_MENU_NO_A_BITS = sum(1 << i for i in START_MENU_NO_A)  # the same set as a bit mask (native/pokerl_native.c)
MENU_ACTION_RELEASE = 23  # PN_ACTION_RELEASE (enum PnMenuAction, obs/observation.h; checked by rl/tests/test_mask.py)

