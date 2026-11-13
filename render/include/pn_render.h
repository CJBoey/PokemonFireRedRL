/* pn_render.h -- on-demand GBA PPU renderer for fireredsim (ARCHITECTURE.md section 7).
 *
 * A pure function of its inputs: it reads the video state (I/O registers, palette RAM, VRAM,
 * OAM) at a logical frame boundary and composes the 240x160 frame the PPU draws from it,
 * scanline by scanline, pixel-exact with mGBA 0.10.5's software renderer (see render/README.md).
 * No global state, no allocation on the normal path, reentrant (all scratch is on the stack,
 * about 16 KiB), usable from a 32-bit (-m32) or 64-bit process.
 *
 * The memory views are never written. Effects of HBlank-timed DMA (scanline effects) are
 * applied to the renderer's private copies only.
 */
#ifndef PN_RENDER_H
#define PN_RENDER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_SCREEN_W 240
#define PN_SCREEN_H 160

#define PN_IO_SIZE 0x400u     /* 0x04000000 */
#define PN_PLTT_SIZE 0x400u   /* 0x05000000 */
#define PN_VRAM_SIZE 0x18000u /* 0x06000000 */
#define PN_OAM_SIZE 0x400u    /* 0x07000000 */

/* Video state at a frame boundary. All four views are required and must be the full
 * hardware size (little-endian bytes exactly as the GBA sees them). */
typedef struct PnVideoMem {
    const uint8_t *io;   /* I/O registers, PN_IO_SIZE bytes */
    const uint8_t *pltt; /* palette RAM (BG 512 B, then OBJ 512 B), PN_PLTT_SIZE bytes */
    const uint8_t *vram; /* VRAM, PN_VRAM_SIZE bytes */
    const uint8_t *oam;  /* OAM, PN_OAM_SIZE bytes */
    /* Maps an address taken from a DMA source/destination register to at least `len`
     * readable bytes, or returns NULL. Used for HBlank-timed DMA only. The value is
     * whatever the game wrote to DMAxSAD/DMAxDAD: a GBA address for oracle dumps
     * (0x02xxxxxx EWRAM, 0x03xxxxxx IWRAM, ...), a native pointer value in the native
     * simulator. Returning a pointer into mem->io/pltt/vram/oam marks a video-memory
     * access (the renderer then uses its private, DMA-updated copy). If NULL or if it
     * returns NULL for 0x04000000-0x07FFFFFF, GBA hardware addresses are mapped to the
     * views directly. May be NULL. */
    const uint8_t *(*resolve)(void *ctx, uint32_t addr, uint32_t len);
    void *ctx;
} PnVideoMem;

/* A ready-made PnVideoMem::resolve (ctx = const PnAddrMap *): a table of address ranges, searched
 * in order. Range r maps an address a in [r.start, r.start + r.span) to r.host + (a - r.start) % r.size;
 * a span larger than the size folds mirrors (the GBA repeats EWRAM every 256 KiB in 0x02xxxxxx and
 * IWRAM every 32 KiB in 0x03xxxxxx). The access [a, a + len) must stay inside one period, else NULL.
 * Typical tables:
 *   oracle dumps    {0x02000000, 0x01000000, 0x40000, ewram}, {0x03000000, 0x01000000, 0x8000, iwram}
 *                   (I/O, palette, VRAM and OAM addresses then fall back to the views, see above)
 *   in-process      one entry per state region of the native simulator with start = the region's
 *                   address, span = size, host = the region itself: the DMA registers hold plain
 *                   pointers of this process, which are used after this range check (works for the
 *                   gstate layout, where REG_BASE/VRAM/... are arrays inside the region, and for the
 *                   mirror layout, where they sit at the GBA addresses).
 * Pure and reentrant; the table is only read. */
typedef struct PnAddrRange {
    uint32_t start;      /* first address */
    uint32_t span;       /* addresses covered, >= size (== size: no mirroring) */
    uint32_t size;       /* bytes behind host */
    const uint8_t *host; /* the bytes of [start, start + size) */
} PnAddrRange;

typedef struct PnAddrMap {
    const PnAddrRange *ranges;
    size_t count;
} PnAddrMap;

const uint8_t *pn_resolve_addr_map(void *ctx, uint32_t addr, uint32_t len);

/* Optional per-scanline hook, e.g. for the native simulator to run gMain.hblankCallback
 * (HBlank IRQ of line-1) or a VCount callback before `line` is composed.
 * Order per line y = 1..159, mirroring hardware/mGBA: line y-1 is composed, then the HBlank
 * DMA transfers of line y-1's HBlank run, then hblank(ctx, y), then line y is composed.
 * For y = 0 the hook runs first (no HBlank DMA happens during VBlank).
 * The hook may write the memory behind mem->io/pltt/vram/oam; after it returns the renderer
 * merges the changed I/O registers, palette entries and OAM into its state (VRAM is read
 * live). With a hook installed a frame costs a little more (diffs of IO/PLTT/OAM per line). */
typedef struct PnRenderHooks {
    void (*hblank)(void *ctx, int line);
    void *ctx;
} PnRenderHooks;

typedef enum PnPixelFormat {
    PN_FMT_RGB8 = 0,     /* uint8_t[160][240][3], R,G,B (HWC, row-major, (0,0) top-left) */
    PN_FMT_XBGR8888 = 1, /* uint32_t[160][240], 0x00BBGGRR = mGBA's native 32-bit color_t */
    PN_FMT_BGR555 = 2,   /* uint16_t[160][240], GBA layout; each 8-bit channel >> 3 (exact for
                            unblended pixels; blended pixels are mGBA's 8-bit result truncated) */
    PN_FMT_GRAY8 = 3     /* uint8_t[H][W] luma, downscaled by PnRenderOpts::scale (pn_render_ex only;
                            see PnRenderOpts for the exact arithmetic) */
} PnPixelFormat;

/* Diagnostics of one render call (all counters, zero when nothing unusual happened). */
typedef struct PnRenderStats {
    uint32_t hblank_dma_units;   /* units moved by HBlank-timed DMA */
    uint32_t dma_unresolved;     /* DMA units whose source or destination could not be mapped */
    uint32_t dma_dropped_writes; /* DMA units written outside IO/PLTT/VRAM/OAM (no visual effect) */
    uint32_t forced_blank_lines; /* lines drawn white because of DISPCNT forced blank */
    uint32_t bitmap_mode_lines;  /* lines in BG modes 3-5 (implemented, not covered by fixtures) */
    uint32_t invalid_mode_lines; /* lines in BG modes 6-7 (backdrop + OBJ tiles >= 512, as mGBA) */
    uint32_t hook_calls;         /* calls of hooks->hblank */
} PnRenderStats;

/* Renders one frame. `out` must hold 160*240 pixels of `fmt`. `hooks` and `stats` may be NULL. */
void pn_render(const PnVideoMem *mem, const PnRenderHooks *hooks, PnPixelFormat fmt, void *out,
               PnRenderStats *stats);

/* Implementation (instruction set) of the compositor. Every implementation produces exactly the
 * same pixels (render_dump cross-checks them on every frame it renders, mgba_fuzz compares each
 * with libmgba); they differ only in speed. pn_render uses PN_ISA_AUTO. */
typedef enum PnRenderIsa {
    PN_ISA_AUTO = 0,    /* the fastest implementation this CPU supports (decided per call) */
    PN_ISA_GENERIC = 1, /* portable C (SSE2 helpers where the compiler targets SSE2) */
    PN_ISA_AVX2 = 2     /* x86 AVX2 kernels (run-time dispatched; the build needs GCC/Clang on x86) */
} PnRenderIsa;

/* pn_render with an explicit implementation (testing, benchmarking). An implementation the CPU or
 * the build lacks falls back to PN_ISA_GENERIC. */
void pn_render_isa(const PnVideoMem *mem, const PnRenderHooks *hooks, PnPixelFormat fmt, void *out,
                   PnRenderStats *stats, PnRenderIsa isa);
/* The implementation PN_ISA_AUTO selects on this CPU (never PN_ISA_AUTO). */
PnRenderIsa pn_render_auto_isa(void);
/* Short name of an implementation ("auto", "generic", "avx2"), NULL if unknown. */
const char *pn_render_isa_name(PnRenderIsa isa);

/* Options of pn_render_ex.
 *
 * fmt       output format. For PN_FMT_RGB8 / XBGR8888 / BGR555, scale must be 1 and line_sub 0
 *           (the full-size frame, exactly what pn_render_isa writes).
 * scale     PN_FMT_GRAY8 only: downscale factor k, 1..PN_GRAY_MAX_SCALE. The output is
 *           uint8_t[H][W] with H = ceil(160 / k), W = ceil(240 / k) (pn_gray_dims), row-major.
 * line_sub  PN_FMT_GRAY8 only: 0 = box filter, output pixel (i, j) is the mean luma of input rows
 *           i*k .. min((i+1)*k, 160) - 1 and columns j*k .. min((j+1)*k, 240) - 1 (partial boxes at
 *           the bottom and right edges); 1 = sub-sampled lines, only the scanlines y = 0, k, 2k, ...
 *           are composed and output pixel (i, j) is the mean luma of row i*k, columns as above. The
 *           lines that are not composed still run their HBlank DMA and the hook, and every per-line
 *           side effect of the compositor (affine reference points, BG enable latency, ...), so the
 *           composed lines are byte-identical to the same lines of a full render.
 * isa       implementation, as for pn_render_isa.
 *
 * Gray arithmetic (render/README.md "Gray formats"; python/tests/test_pixels.py checks it in numpy):
 * luma of one pixel from the R, G, B bytes of the PN_FMT_RGB8 frame (green swap and blending
 * included): Y = (77*R + 150*G + 29*B + 128) >> 8; mean of n lumas = (sum + n/2) / n (integers). */
typedef struct PnRenderOpts {
    PnPixelFormat fmt;
    int scale;
    int line_sub;
    PnRenderIsa isa;
} PnRenderOpts;

#define PN_GRAY_MAX_SCALE 15

/* Output size of PN_FMT_GRAY8 at `scale` (any scale >= 1): h = ceil(160 / scale), w = ceil(240 / scale). */
void pn_gray_dims(int scale, int *h, int *w);

/* Renders one frame with options (NULL opts = PN_FMT_RGB8, scale 1, PN_ISA_AUTO). `out` must hold the
 * output of opts (160*240 pixels of fmt, or pn_gray_dims bytes for PN_FMT_GRAY8). Returns 0 on success,
 * -1 (nothing rendered, out untouched) if the options are invalid. pn_render_isa(mem, hooks, fmt, out,
 * stats, isa) == pn_render_ex with {fmt, 1, 0, isa}. */
int pn_render_ex(const PnVideoMem *mem, const PnRenderHooks *hooks, const PnRenderOpts *opts, void *out,
                 PnRenderStats *stats);

/* Convenience wrappers. */
void pn_render_rgb(const PnVideoMem *mem, const PnRenderHooks *hooks,
                   uint8_t out[PN_SCREEN_H * PN_SCREEN_W * 3]);
void pn_render_xbgr8888(const PnVideoMem *mem, const PnRenderHooks *hooks,
                        uint32_t out[PN_SCREEN_H * PN_SCREEN_W]);
void pn_render_bgr555(const PnVideoMem *mem, const PnRenderHooks *hooks,
                      uint16_t out[PN_SCREEN_H * PN_SCREEN_W]);

/* mGBA's 15-bit -> 8-bit channel expansion: (c << 3) | (c >> 2). */
static inline uint8_t pn_expand5(unsigned c5) {
    return (uint8_t) (((c5 & 0x1Fu) << 3) | ((c5 & 0x1Fu) >> 2));
}

#ifdef __cplusplus
}
#endif

#endif /* PN_RENDER_H */
