"""fireredsim: Pokemon FireRed (USA 1.0) recompiled natively, batched for reinforcement learning.

    import fireredsim
    env = fireredsim.PokeVecEnv(num_envs=64, frameskip=24)      # batched, shared memory
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(actions)

The simulator runs in 32-bit worker processes (``fireredsim-worker``, built by CMake); this package is pure Python +
NumPy (python/README.md).
"""
from ._ipc import WorkerError
from ._paths import find_worker
from .actions import ACTION_SETS, BUTTON_ORDER, BUTTONS, action_table, buttons_to_str, parse_buttons
from .starts import SaveStart, ScriptStart
from .vec_env import PokeVecEnv, SimulatorError

__all__ = [
    "PokeVecEnv", "SaveStart", "ScriptStart", "SimulatorError", "WorkerError", "find_worker",
    "ACTION_SETS", "BUTTONS", "BUTTON_ORDER", "action_table", "buttons_to_str", "parse_buttons",
]
