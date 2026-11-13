# render/ — GBA PPU renderer

`render/` turns the video state of a frame boundary (I/O registers, palette RAM, VRAM, OAM) into the
240×160 picture the GBA draws from it, scanline by scanline. Its compositor is derived from mGBA 0.10.5's
software renderer and was checked pixel-identical to mGBA's output in the project this repository was
extracted from. The simulator never needs it to step; the worker calls it only for the frames that become
pixel observations (`worker/render_adapter.cpp`).

| file | what |
|---|---|
| `include/pn_render.h` | public C API (C11, also usable from C++) |
| `src/pn_render.c` | frame driver: state rebuild, HBlank-DMA engine, per-line hook, output formats, `pn_resolve_addr_map` |
| `src/pn_ppu.c`, `src/pn_ppu.h` | scanline compositor derived from mGBA 0.10.5 (MPL-2.0, `THIRD_PARTY.md`) |
| `THIRD_PARTY.md` | provenance and licences |

```c
#include <pn_render.h>

PnVideoMem mem = { io, pltt, vram, oam, resolve, ctx };   /* 1 KiB, 1 KiB, 96 KiB, 1 KiB, as the GBA sees them */
uint8_t rgb[160 * 240 * 3];
pn_render_rgb(&mem, NULL, rgb);                            /* HWC RGB8, (0,0) top-left */
PnRenderOpts o = { PN_FMT_GRAY8, 2, 0, PN_ISA_AUTO };       /* gray, scale 2, box filter */
uint8_t gray[80 * 120];                                    /* pn_gray_dims(2, &h, &w): 80 x 120 */
pn_render_ex(&mem, NULL, &o, gray, &stats);                /* 0, or -1 for invalid options */
```

- **Gray formats** (`pn_render_ex`): `PN_FMT_GRAY8` is `uint8_t[ceil(160/k)][ceil(240/k)]` for scale k.
  The luma of a pixel is `Y = (77 R + 150 G + 29 B + 128) >> 8` of its RGB8 value, and an output pixel is
  the rounded mean `(sum + n/2) / n` of the lumas of its k×k box (`line_sub = 0`). The training
  observation (`gray2`) is scale 2: 80×120.
- **HBlank effects**: the native simulator never executes HBlank-timed DMA; the renderer does, per line.
  An optional `PnRenderHooks.hblank(ctx, line)` runs C HBlank callbacks (e.g. the trainer card flip)
  between lines.
- **Implementations**: `PN_ISA_AUTO` uses AVX2 kernels when the CPU supports them, else portable C; both
  produce identical bytes in every format. About 40 µs per full frame on i386.
- **Purity**: the output is a pure function of the views. There is no mutable global state, and all
  scratch lives in a per-call structure on the stack, so any number of threads may render at once. The
  caller's memory is never written.
