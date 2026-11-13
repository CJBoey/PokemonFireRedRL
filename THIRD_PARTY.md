# Third-party code and licences

This repository's own code is under the MIT licence (`LICENSE`). This file lists everything else that is in the
repository or that its build uses, and the terms each part comes with.

## The game: pret/pokefirered (not included)

`tools/fetch_game.sh` fills `game/` from [pret/pokefirered](https://github.com/pret/pokefirered) at the commit in
`game/UPSTREAM.txt`. The decompilation (code, graphics, music and text) is not part of this repository. pret
publishes it without a licence, and the game it reconstructs belongs to Nintendo, Creatures Inc. and GAME FREAK inc.

These files are derived from the game's own code and are **not** covered by the MIT licence:

- `game/native/pokefirered.patch`: the simulator's changes to 58 of pret's files;
- `platform/sound.c`: a C port of the m4a sound engine (pret's `src/m4a_1.s`);
- `platform/gcn_multiboot.c`: a C port of the GameCube link library (pret's `src/libgcnmultiboot.s`).

`docs/v76_full_run.jpg`, the README's video thumbnail, is a frame of a recording of the game. It shows the game's
graphics and is not covered by the MIT licence either.

The repository's own files in `game/` (`Makefile.native`, `NATIVE_CHANGES.md`, `UPSTREAM.txt`, `native/*.py` and
`include/gba/portable_abi.h`) are under the MIT licence.

`rl/pokerl/tables.py` and `reference/rom_symbols*` hold facts about the game, generated from pret's sources and
build: names, base stats, learnsets, item pockets, and RAM addresses and sizes.

## mGBA (MPL-2.0)

- `render/src/pn_ppu.c` and `render/src/pn_ppu.h` are derived from mGBA 0.10.5's software renderer.
  `render/THIRD_PARTY.md` gives their origin and modifications.
- `platform/bios.c` is derived from mGBA 0.10.5's HLE BIOS (`src/gba/bios.c`, `src/gba/hle-bios.s`).

[mGBA](https://mgba.io) is Copyright (c) 2013-2019 Jeffrey Pfau. These files are under the Mozilla Public License
2.0 (<https://mozilla.org/MPL/2.0/>) and keep its header. Under the MPL, they and any changes to them must stay
available in source form under the MPL; the licence of the rest of the repository is not affected.

`platform/dma.c` and the DMA engine in `render/src/pn_render.c` were written with mGBA's `src/gba/dma.c` as a
reference for the hardware's behaviour.

## newlib (Cygnus Solutions)

`Port_memcpy`, `Port_strcmp` and `Port_strcpy` in `platform/libc_port.c` follow newlib's `string/memcpy.c`,
`strcmp.c` and `strcpy.c`, as built into the C library of agbcc (the compiler the retail game was built with).
This software was developed at Cygnus Solutions. Their notice, also kept in the file:

```
Copyright (c) 1994, 1997 Cygnus Solutions.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
at Cygnus Solutions.  Cygnus Solutions may not be used to
endorse or promote products derived from this software without
specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
```

## pokerl-map-viz and PixiJS (MIT)

`tools/mapviz/www/viewer.js` is adapted from [PWhiddy/pokerl-map-viz](https://github.com/PWhiddy/pokerl-map-viz),
Copyright (c) 2024 Peter Whidden, and `tools/mapviz/www/pixi.js` (PixiJS v7.3.3) is copied from it. Both are under
the MIT licence; `tools/mapviz/www/LICENSE-pokerl-map-viz.txt` has the notice.

## Trademarks

Pokémon and Pokémon FireRed are trademarks of Nintendo, Creatures Inc. and GAME FREAK inc. This project is not
affiliated with or endorsed by them. It is meant for education and research.
