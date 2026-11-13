# Third-party code in render/

## mGBA 0.10.5 — Mozilla Public License 2.0

`src/pn_ppu.c` and `src/pn_ppu.h` are derived from mGBA, Copyright (c) 2013-2019 Jeffrey Pfau,
tag 0.10.5, commit `26b7884bc25a5933960f3cdcd98bac1ae14d42e2`. They are distributed under the Mozilla Public License, v. 2.0
(<https://mozilla.org/MPL/2.0/>), keep the MPL file header and state their origin and
modifications at the top of the file. Origin:

| fireredsim file | mGBA files |
|---|---|
| `src/pn_ppu.c` | `src/gba/renderers/video-software.c` (register writes, windows, pre/post-processing, scanline driver), `software-private.h` (compositing, brighten/darken), `software-mode0.c` (text BGs), `software-bg.c` (affine and bitmap BGs), `software-obj.c` (OBJs), `common.c` (OAM cleaning), `include/mgba/core/interface.h` (`mColorFrom555`, `mColorMix5Bit`, 32-bit `color_t` branch), the OBJ size table of `src/gba/video.c` |
| `src/pn_ppu.h` | `include/mgba/internal/gba/renderers/video-software.h` (state structure, flags) |

Modifications: one translation unit; state rebuilt from a memory snapshot per frame instead of
bus-write callbacks; scanline dirty tracking and caching, highlighting, debug layer toggles,
cache-set hooks and output-buffer management removed; always the 32-bit `color_t` (0x00BBGGRR)
build; bit-field accessor macros expanded locally (keeping the "Is" = masked / "Get" = shifted
distinction); window list given headroom; the two mosaic coordinates of `BACKGROUND_BITMAP_INIT`
initialised to 0 (only read on the mosaic path, which assigns them). Pixel-affecting logic is
unchanged.

Under MPL-2.0 the covered files (and modifications to them) must stay available in source form
under the MPL; the rest of `render/` (`src/pn_render.c`, `include/pn_render.h`) is
fireredsim's own code and is not covered. `src/pn_render.c`'s DMA engine re-implements the
behaviour of mGBA's `src/gba/dma.c` (no code copied); it is documented as such.
