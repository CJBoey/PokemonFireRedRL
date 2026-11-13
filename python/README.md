# fireredsim — Python binding

Pokémon FireRed (USA 1.0), recompiled natively and stepped by 32-bit worker processes, as batched
reinforcement-learning environments. This package is pure Python + NumPy; the simulator is the
`fireredsim-worker` executable built by CMake. Observations are written by the workers straight into shared
memory that NumPy maps (no serialisation). Linux x86-64 only (32-bit multilib build of the worker, futexes,
`memfd_create`).

## Install

Build the worker (top-level README), then either `pip install -e python` or put `python/` on `PYTHONPATH`. The
package finds the newest `build*/worker/fireredsim-worker` of the repository; set
`FIREREDSIM_WORKER=/path/to/fireredsim-worker` (or pass `worker_path=`) to choose one.

```bash
python -m pytest python/tests    # needs the worker
```

## Minimal loop (random actions)

```python
import numpy as np
import fireredsim

env = fireredsim.PokeVecEnv(num_envs=16, frameskip=24, seed=0)   # workers selected from the CPU count
obs, info = env.reset()               # structured ndarray, one record per env (obs/observation.h)
rng = np.random.default_rng(0)
for t in range(1000):
    actions = rng.integers(0, len(env.actions), size=env.num_envs)   # Discrete(9)
    obs, reward, terminated, truncated, info = env.step(actions)
    # obs["map_group"], obs["x"], obs["party"]["hp"], obs["screen"], obs["waiting_for_input"], ...
env.close()
```

## What a step is

`step(a)` holds the buttons of action `a` for `frameskip` logical frames (or only the first `hold_frames` of
them, then releases; both may be per env) and returns the state at the end of the last frame. Frame skip never
changes the game: a step of 24 frames is bit for bit the same as 24 single-frame steps with the same buttons held;
button-press edges, key repeat, scripts, NPCs and the RNG run exactly as in the original game.
`auto_advance=True` makes every step a decision step: after the step's frames the worker keeps running NOOP chunks
while the game does not wait for input (`obs["waiting_for_input"]`), up to `max_frames` frames per step
(`info["frames_run"]`). `step_async(actions)` / `step_wait()` overlap a batch with your own work.

## Actions

GBA KEYINPUT bits (A=1 B=2 SELECT=4 START=8 RIGHT=16 LEFT=32 UP=64 DOWN=128 R=256 L=512):

| action_set | actions |
|---|---|
| `"basic"` (default) | NOOP, A, B, START, SELECT, UP, DOWN, LEFT, RIGHT |
| `"full"` | basic + L, R, B+UP, B+DOWN, B+LEFT, B+RIGHT (running) |
| a list | any button masks / names (`["A", "B+LEFT", 0x40]`) |

`action_type="multibinary"` takes a 0/1 vector of the 10 buttons (in `fireredsim.BUTTON_ORDER`),
`action_type="buttons"` raw masks. A+B+START+SELECT soft-resets the game, as on hardware (`info["soft_reset"]`).

## Observations

- `obs_mode="visible"` (default): what a player can see or reach through the menus. `obs_mode="privileged"` fills
  the `*_priv` fields too (RNG, exact enemy data, flags, vars, ...): for reward shaping, not for the policy.
- `PokeVecEnv` returns the raw structured records (`env.obs_dtype`, `fireredsim.obs_dtype.OBS_DTYPE`, generated
  from obs/observation.h; strings are game-encoded, decode with `fireredsim.obs.decode_text`).
- `pixels=True` adds the final frame of each step, `uint8[N, 160, 240, 3]`; `pixel_format="gray2"` (and `gray1`
  .. `gray4`, `gray2s` .. `gray4s`) renders it as downscaled luma instead, `uint8[N, H, W]` (80 x 120 for gray2:
  2 x 2 box averages); `env.pixel_shape` gives the shape, `env.pixels` the mapped view. `env.render(ids)` renders
  on demand.
- `copy=False` returns views of the shared memory (zero copy; overwritten by the next call).

## Rewards and episodes

Rewards are 0 unless you supply `reward_fn(prev_obs, obs, info)` (batched records in, `float[N]` out). Episodes
end when your `termination_fn(obs, info)` says so, after `max_episode_steps` (truncation), or when the simulator
ends an env itself (`truncated` with `info["sim_terminal"]`: the real game would hang or crash there, or an
unsupported feature such as the link cable was used). `PokeVecEnv` resets ended envs on the next `step`
(next-step autoreset: their action is ignored, `info["autoreset"]`); `reset(env_ids=[...])` resets envs at once.

## Starts, seeds, snapshots

- `start="bedroom"` (default): a new game, replayed from power-on through the intro and naming screens to the
  player's room (frame 5161; worker/new_game_to_pallet.txt, embedded in the worker). The game's RNG seed and
  trainer ID come from the hardware timer at two moments of that sequence; each episode gets its own values from
  its seed (`fireredsim.seeding`), so episodes differ by seed and are reproducible. A new seed costs one ~9 ms
  replay in its worker, then it is cached (`bedroom_cache` per worker).
- `start=fireredsim.ScriptStart([...])`: input scripts (lines `START END BUTTONS`: the buttons held during frames
  [START, END) from power-on; several files are merged) replayed from power-on per episode seed, e.g. the bedroom
  script followed by option-menu presses (rl/pokerl/starts/).
- `start=fireredsim.SaveStart("game.sav")`: continue a 128 KiB FireRed flash save (a fixed input driver in the
  worker taps through the title screen and CONTINUE; the env is handed over at the first frame the player can move).
- `preset_flags` / `preset_vars`: event flags set and game variables written into every start replayed from
  power-on.
- `reset(seed=s)` reseeds every env (env i gets a seed derived from `s` and `i`, independent of the number of
  workers); `reset(seed=[...])` sets per-env episode seeds.
- `snapshot(ids)` -> list of `bytes`, `restore(snaps, ids)`, `clone(src_ids, dst_ids)`, `state_hash(ids)`;
  `reserved_slots` keeps snapshot slots in the workers' shared memory (`snapshot_to_slots`, `restore_from_slots`,
  `slot_bytes`) for snapshots that never cross into this process.

### RNG reseeding (no RNG manipulation)

Within an episode the game's RNG is one LCG chain driven by the agent's own timing, as on hardware, so an agent
could learn speedrun-style RNG manipulation. `rng_reseed=True` prevents it: before every step's first frame the
game's PRNG state is replaced by fresh values from a per-env host entropy stream (splitmix64), seeded from the
episode seed at every reset (`seeding.rng_stream_seed`) and carried by snapshots. The game code, rules and
probabilities are unchanged; only the random sequence differs.

## Workers

The defaults use one env per worker up to 3/4 of the logical CPUs, then two envs per worker, then one env per worker
again up to 16 x CPUs; `num_workers` overrides it. A worker process costs ~12 MiB plus its start-state cache. Every
`step` is one synchronous batch over all envs, so the slowest worker sets the pace; workers busy-wait between batches
only while their count fits 3/4 of the CPUs. `send(actions, env_ids)` / `recv(min_envs=k)` remove the barrier.
