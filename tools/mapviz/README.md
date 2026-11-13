# Map visualizer

A live map of a running training: every env is drawn as the FireRed player on a world map built from
`game/data`. It is adapted from [PWhiddy/pokerl-map-viz](https://github.com/PWhiddy/pokerl-map-viz) (MIT;
`www/LICENSE-pokerl-map-viz.txt`) and hosted locally.

```sh
python3 tools/mapviz/build_world.py   # once (~45 s): renders www/world/{story,full}/ from game/data
python3 tools/mapviz/serve.py         # then open http://localhost:8780/
```

The server finds the newest `python -m pokerl.train` itself (`--pid` picks another one) and follows
restarts. Nothing in the trainer changes.

## Cost to training

The server reads the env records (`x`, `y`, `map_group`, `map_num`, `badges`, `in_battle`) straight from
the workers' shared memory (the memfd of `python/fireredsim`), mapped read-only. The trainer does no
work for the stream.

* **No visible page** (no page open, tab hidden, paused, or idle for 10 minutes): no reads. `--linger`
  seconds later (default 10) the memory is unmapped, and the server waits in its event loop at 0 CPU.
* **A visible page**: 30 samples per second (`--hz`). The page receives one batch per env per second
  (`--flush`). Measured on a 512-env training (2 groups): 1.2–1.8 % of
  one core, out of the machine's 32.

`--envs 0-63` streams a subset. `--hz` trades smoothness for CPU. The page draws smooth movement
between samples, so 10–30 Hz looks the same.

## Page

* Wheel or pinch to zoom, drag to pan. **H** returns to Kanto.
* Zoomed out, each env is a dot coloured by its badge count. The panel shows how many envs have each
  badge count. Zoomed in, each env is the player sprite. It is dimmed while in battle, and jumps
  instead of walking across warps.
* **P** or the Story/Full buttons switch the map:
  * **story** (default): Kanto and Sevii Islands 1–3, everything reachable before the Hall of Fame.
  * **full**: adds the post-game (Sevii 4–7, Trainer Tower, Tanoby, Cerulean
    Cave, the Ruby Path) and the event islands.

  A position on a map the shown variant lacks is not drawn.
* **L** shows labels: `e<env> b<badges>`. **F** filters them with a regular expression; for example,
  `b[3-8]` shows envs with three or more badges.
* **M** shows the dwell heatmap (below), **B** counts battle time in it, **N** shows the novelty view
  (below), **E** hides the envs.
* **Space** pauses (the server stops reading). `?idle=0` turns the idle pause off; `?idle=N` sets it to
  N minutes. `?map=full` opens the full map.

## Dwell heatmap

Where the envs spend their time. The server counts every position sample it takes per (map, x, y) tile,
keeping field and battle time apart. Each sample of one env stands for the window's wall time divided by
its samples, so a count gives that env time. The counts come only from samples the server already takes,
so they cover only the time some page was visible. They don't depend on the heatmap being shown, and the
trainer does no extra work. Not counted: the warp window (`field_state` 2) and Quest Log playback (3).

* The table belongs to the run name (`--run-name` or `--resume`). It is saved in `heat/<run>.npz`
  (git-ignored) every minute and on detach or exit, so it survives server restarts and continues across
  resumes. **Reset** clears it for every viewer.
* The page draws one pixel per tile on a log colour scale. The scale tops out at the 99.5th percentile of
  visited tiles, so a few hot spots don't wash out the rest. The map beneath is dimmed. Hovering a tile
  shows its map, coordinates, env time and share of all sampled time.
* Pages that show it get the table as a binary frame every `--heat-every` seconds (default 5).

## Novelty view (tile-count bonus)

**N** shows how much of the tile-count bonus is left, and where. A first visit in an episode to a tile that
N earlier episodes first-visited pays W · min(1, √(S / (N + 1))), where W is `--tile-count-reward` and S is
`--tile-count-scale` (pokerl `reward.py`). The trainer keeps these run-long counts in its own memory, but every
checkpoint saves them. While a page shows the view, the server checks the run's directory every 5 s and reads
each new `ckpt_*.pt`. It reads only the checkpoint's `data.pkl` (about 250 KB, 3 ms, no torch, model weights
skipped), so the trainer does no extra work. The tile layout comes from the trainer's own `pokerl/tables.py`
and is checked against the checkpoint's `tile_layout`.

* **Left** (default): each reached tile is coloured by the share of W a first visit still pays (viridis:
  yellow is full pay, purple is spent).
* **Recent** (**R** toggles): first visits per tile between the newest checkpoint and the newest one at
  least `--novelty-window` seconds older (default 600). Tiles first reached in that window are white, and
  reached tiles with no first visits in it are slate. When the server attaches, the window starts at the
  oldest checkpoint still on disk. It grows while a page stays visible, because the server keeps the
  checkpoints it has read in memory (not saved).
* **Frontier** (**O**): passable tiles nobody has reached, next to reached ones. They are drawn as markers
  that keep a minimum screen size: pink for land, cyan for water (needs Surf). Other unreached passable
  tiles are a grey (land) or blue (water) checker. Walkability comes from `walk.bin` (`build_world.py`),
  built from the collision bits and the surfable behaviours. Objects (NPCs, item balls, boulders, cut
  trees) are not in it, so a frontier tile can be one that an object stands on.
* The panel shows the checkpoint and its age, the tiles reached, the tiles at full pay and the median share.
  In Recent mode it also shows first visits, new tiles and an estimate of the bonus paid per env-step,
  which you can compare with the trainer's `r` column. Hovering shows a tile's count, what its next first
  visit pays and its recent first visits.
* The dwell heatmap and the novelty view are exclusive; turning one on turns the other off.

## Party page (parties, moves, battle and party habits)

`http://localhost:8780/party.html` (the **Party** link on the map page) shows the live training's
envs: every env's party (species icon, level, HP, moves; three mons per row), badges, map and relay scope, with tiles (best level, party size,
lead not the strongest, PC mons, money), levels by rank, leads, and a species table (share of envs, levels, top moves;
TM/HM moves underlined; click a row for every move). Filters: badge counts (multi-select), relay scope, in / out of
battle, a species name. **Behaviour sample** (30 s to 5 min, on request) counts: time shares (battle, field, party
menu, bag, TM case, PC box), per trainer / wild battle the win rate, a lead below the party's best level, a switch to a
stronger mon while that lead lives, the switch-training pattern (that and the lead still gets EXP), EXP shared,
voluntary / forced switches, losses; wild battles: battles and KOs per env-minute and one outcome per battle (KO: a
mon gained EXP; caught: a new party mon or box mon; lost; ran: anything else), per map; outside battle: party reorders (per env-minute, envs, leaving a weaker lead), PC
deposits / withdrawals, catches / gifts, moves learned (TM/HM moves), evolutions; items and shopping (2026-10-29):
purchases, ball throws, consumables used in / outside battle, given to a mon to hold, put in the item PC or tossed,
sales, catches sent to the box, per item, and the
one-step skill event bits seen in the copies (sampled pulses); where the time goes (time and battle share per map).
The snapshot also has HMs (in the bag, taught, in the bag but not taught, and whether a party mon can learn it), the
bag's top consumables, and the game counters (shopped, Pokemon Center heals, captures, battles, evolutions, Cut uses;
cumulative over the save's lineage, since relay batons and rewinds keep them); the HM filter (in bag, not taught)
applies to both.

Cost: nothing between requests. A snapshot is one copy of the records (~0.4 s, cached 10 s); a sample copies them 10
times a second on one core (~30 % of a core at 1024 envs, mostly the copy) for its length. Both run `party_probe.py` as a subprocess with the
trainer's own record layout and pokerl tables (its PYTHONPATH), so it follows whichever checkout the trainer runs from; also from a shell:
`python3 -I tools/mapviz/party_probe.py snapshot` / `sample --secs 60` (JSON). The relay scope is a record-only proxy
(in the scope's area, entry condition met, not cleared), not the trainer's per-life status. The game runs ~66x real
time, so sampled counts are lower bounds; restored states (new lives, rewinds: play time going back, a
mon losing EXP, a whole new party) are skipped.

## World map (`build_world.py`)

Every map is drawn at 16 px per metatile from its layout's blocks and tilesets, using the built
`tiles.4bpp` and `.gbapal` files, or the PNG and `.pal` sources in a fresh checkout. Placement:

* Outdoor maps are placed by their map connections: Kanto from Pallet Town, the Sevii Islands in a
  band east of it.
* Indoor and cave maps are grouped by region map section (a city's buildings, a dungeon's floors). Each
  group goes in the free space nearest to the doors that lead into it, with each map's name above it.
* Left out: the link rooms, the prototype maps and the unused houses.

The party page's species icons come from the same run: `www/world/icons.png` (every species' party menu icon in
its icon palette, 32 per row by species id); `build_world.py --variants ""` writes only it and `player.png`.

Output per variant goes in `www/world/<variant>/` (git-ignored):

* `overview.png`: the whole world at ¼ scale, shown when zoomed out.
* 2048 px `chunk_R_C.png` tiles: loaded only when zoomed in and in view, at most 16 at a time.
* `world.json`: the regions, one per map, with id `map_group * 256 + map_num`.
* `walk.bin`: one byte per world tile: 0 outside every map, 1 passable, 2 surfable water, 3 blocked. The
  novelty view uses it to find the frontier.

## Files

* `serve.py`: the server (stdlib and NumPy only): static files, a WebSocket at `/receive`, the
  shared-memory reader, the dwell table and the checkpoint reader of the novelty view, and the party page's JSON API
  (`/api/party/snapshot`, `/api/party/sample`).
* `www/party.html`, `www/party.js`, `party_probe.py`: the party page and its record reader.
* `build_world.py`: the world maps.
* `www/index.html`, `www/viewer.js`: the page, using PixiJS 7.3.3 (`www/pixi.js`, vendored from
  pokerl-map-viz).

The WebSocket only accepts pages from its own host, and the server binds to `127.0.0.1` unless you pass
`--host 0.0.0.0`.
