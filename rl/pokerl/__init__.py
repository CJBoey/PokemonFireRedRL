"""pokerl: PPO on the native FireRed simulator (fireredsim), as run for v76.

Modules: env (the training env: the PokeVecEnv wrapper, life ends, rewind, relay scopes, the unique and Indigo
Plateau terms), reward (the terms, weights and tables; the per-step code is native/), native (the C reward, masks and
trail window), mask, events (milestones), relay (scopes, exit tables), rewind (capture rings), model (the policy),
mingru, ppo (rollout, GAE, update), train (the CLI), tables (generated from the game data by gen_tables, with atlas),
records (record field helpers). Run with PYTHONPATH=rl:python.
"""
