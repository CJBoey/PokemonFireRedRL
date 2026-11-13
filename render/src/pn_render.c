/* pn_render.c -- frame driver of fireredsim's renderer (ARCHITECTURE.md section 7).
 *
 * Builds the compositor state (pn_ppu.c) from a video-memory snapshot taken at a logical
 * frame boundary, then composes the 160 scanlines in hardware order:
 *
 *   for y in 0..159:
 *       if y > 0: HBlank-timed DMA transfers of line y-1's HBlank   (scanline effects)
 *       hooks->hblank(ctx, y)                                       (optional, e.g. HBlank IRQ)
 *       compose line y
 *
 * which is the order of mGBA's _startHblank (draw line, then GBADMARunHblank, then the HBlank
 * IRQ). VBlank-timed and immediate DMAs are already reflected in the snapshot; DMA1/DMA2
 * sound-FIFO ("special" timing) and DMA3 video capture are not video effects and are ignored.
 *
 * The caller's memory is never written: the I/O registers, palette RAM and OAM are copied
 * (3 KiB) and VRAM is copied on the first DMA write to it only.
 *
 * pn_render_ex's gray formats: the emit stage turns each composed line into lumas (box filter:
 * accumulated per output row); with sub-sampled lines only y = 0, k, 2k, ... are composed and the
 * other lines run everything above except the compositing (pn_ppu_skip_scanline).
 */
#include "pn_render.h"

#include "pn_ppu.h"

#include <stdlib.h>
#include <string.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#define PN_HAVE_SSE2 1
#endif
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && !defined(PNRENDER_NO_AVX2)
#include <immintrin.h>
#define PN_HAVE_AVX2 1
#define PN_AVX2_FN __attribute__((target("avx2"))) static
#endif

enum { DMA_TIMING_NOW = 0, DMA_TIMING_VBLANK = 1, DMA_TIMING_HBLANK = 2, DMA_TIMING_CUSTOM = 3 };

typedef struct PnDmaChannel {
    uint16_t reg;      /* CNT_H as mGBA keeps it (masked) */
    uint32_t count;    /* units per burst (CNT_L; 0 means 0x4000, or 0x10000 for DMA3) */
    uint32_t nextSource;
    uint32_t nextDest;
} PnDmaChannel;

typedef struct PnFrame {
    const PnVideoMem *mem;
    PnRenderStats stats;
    PnPpu ppu;

    uint8_t io[PN_IO_SIZE];     /* working I/O registers (HBlank DMA and hooks write here) */
    uint8_t pltt[PN_PLTT_SIZE]; /* working palette RAM */
    uint8_t oam[PN_OAM_SIZE];   /* working OAM */
    const uint8_t *vram;        /* mem->vram, or vramCopy after a DMA write to VRAM */
    uint8_t *vramCopy;
    uint8_t *vramSeen;          /* with a hook: the caller's VRAM as of the last merge (after a copy) */

    PnDmaChannel dma[4];
    uint32_t dmaTransferRegister; /* last value moved by DMA (used when a source is unreadable) */

    /* Hook support: what the caller's memory looked like after the previous sync. */
    int hooked;
    uint8_t ioSeen[PN_IO_SIZE];
    uint8_t plttSeen[PN_PLTT_SIZE];
    uint8_t oamSeen[PN_OAM_SIZE];
} PnFrame;

static inline uint16_t rd16(const uint8_t *p, uint32_t off) {
    uint16_t v;
    memcpy(&v, p + off, 2);
    return v;
}

static inline uint32_t rd32(const uint8_t *p, uint32_t off) {
    uint32_t v;
    memcpy(&v, p + off, 4);
    return v;
}

static inline void wr16(uint8_t *p, uint32_t off, uint16_t v) {
    memcpy(p + off, &v, 2);
}

/* ------------------------------------------------------------------ address classification */

enum PnRegion { RGN_NONE = 0, RGN_IO, RGN_PLTT, RGN_VRAM, RGN_OAM, RGN_OTHER };

typedef struct PnTarget {
    int region;
    uint32_t offset;         /* for IO/PLTT/VRAM/OAM */
    const uint8_t *ptr;      /* for RGN_OTHER (readable host memory) */
} PnTarget;

static int in_view(const uint8_t *p, const uint8_t *base, uint32_t size, uint32_t len, uint32_t *off) {
    if (!base) {
        return 0;
    }
    uintptr_t d = (uintptr_t) p - (uintptr_t) base;
    if ((uintptr_t) p >= (uintptr_t) base && d < size && d + len <= size) {
        *off = (uint32_t) d;
        return 1;
    }
    return 0;
}

static PnTarget classify(const PnFrame *f, uint32_t addr, uint32_t len) {
    PnTarget t = {RGN_NONE, 0, NULL};
    const PnVideoMem *mem = f->mem;
    if (mem->resolve) {
        const uint8_t *p = mem->resolve(mem->ctx, addr, len);
        if (p) {
            uint32_t off;
            if (in_view(p, mem->io, PN_IO_SIZE, len, &off)) {
                t.region = RGN_IO;
                t.offset = off;
            } else if (in_view(p, mem->pltt, PN_PLTT_SIZE, len, &off)) {
                t.region = RGN_PLTT;
                t.offset = off;
            } else if (in_view(p, mem->vram, PN_VRAM_SIZE, len, &off)) {
                t.region = RGN_VRAM;
                t.offset = off;
            } else if (in_view(p, mem->oam, PN_OAM_SIZE, len, &off)) {
                t.region = RGN_OAM;
                t.offset = off;
            } else {
                t.region = RGN_OTHER;
                t.ptr = p;
            }
            return t;
        }
    }
    /* GBA hardware addresses (mirrors folded like the GBA bus). */
    switch (addr >> 24) {
    case 0x04:
        if ((addr & 0x00FFFFFFu) < PN_IO_SIZE) {
            t.region = RGN_IO;
            t.offset = addr & (PN_IO_SIZE - 1);
        }
        break;
    case 0x05:
        t.region = RGN_PLTT;
        t.offset = addr & (PN_PLTT_SIZE - 1);
        break;
    case 0x06: {
        uint32_t off = addr & 0x1FFFFu;
        if (off >= PN_VRAM_SIZE) {
            off -= 0x8000u;
        }
        t.region = RGN_VRAM;
        t.offset = off;
        break;
    }
    case 0x07:
        t.region = RGN_OAM;
        t.offset = addr & (PN_OAM_SIZE - 1);
        break;
    default:
        break;
    }
    if (t.region != RGN_NONE && t.offset + len > (t.region == RGN_IO ? PN_IO_SIZE
                                                  : t.region == RGN_PLTT ? PN_PLTT_SIZE
                                                  : t.region == RGN_VRAM ? PN_VRAM_SIZE
                                                                         : PN_OAM_SIZE)) {
        t.region = RGN_NONE;
    }
    return t;
}

/* ------------------------------------------------------------------ address-map resolver */

const uint8_t *pn_resolve_addr_map(void *ctx, uint32_t addr, uint32_t len) {
    const PnAddrMap *map = (const PnAddrMap *) ctx;
    size_t i;
    if (!map) {
        return NULL;
    }
    for (i = 0; i < map->count; ++i) {
        const PnAddrRange *r = &map->ranges[i];
        if (!r->host || !r->size || addr < r->start || (uint64_t) (addr - r->start) >= r->span) {
            continue;
        }
        uint32_t off = (addr - r->start) % r->size;
        if ((uint64_t) off + len > r->size) {
            return NULL; /* crosses the end of the backing memory (or of a mirror period) */
        }
        return r->host + off;
    }
    return NULL;
}

/* ------------------------------------------------------------------ register / memory writes */

static void dma_write_cnt_hi(PnFrame *f, int ch, uint16_t control);

/* A 16-bit write to the I/O registers at `off`, with the side effects mGBA's GBAIOWrite has
 * on the video unit and the DMA controller. */
static void io_write16(PnFrame *f, uint32_t off, uint16_t value) {
    off &= PN_IO_SIZE - 2;
    if (off < 0x060 && (off > 0x006 || off < 0x004)) {
        value = pn_ppu_write_video_register(&f->ppu, off, value);
        wr16(f->io, off, value);
        return;
    }
    if (off >= 0x0B0 && off < 0x0E0) {
        int ch = (int) ((off - 0x0B0) / 12);
        uint32_t reg = (off - 0x0B0) % 12;
        if (reg == 8) { /* CNT_L */
            wr16(f->io, off, value);
            uint32_t count = ch < 3 ? (value & 0x3FFFu) : value;
            f->dma[ch].count = count ? count : (ch == 3 ? 0x10000u : 0x4000u);
            return;
        }
        if (reg == 10) { /* CNT_H */
            dma_write_cnt_hi(f, ch, value);
            return;
        }
        wr16(f->io, off, value); /* SAD / DAD halves: latched when the channel is enabled */
        return;
    }
    wr16(f->io, off, value);
}

static void pltt_write16(PnFrame *f, uint32_t off, uint16_t value) {
    off &= PN_PLTT_SIZE - 2;
    wr16(f->pltt, off, value);
    pn_ppu_write_palette(&f->ppu, off, value);
}

static void oam_write16(PnFrame *f, uint32_t off, uint16_t value) {
    off &= PN_OAM_SIZE - 2;
    wr16(f->oam, off, value);
    pn_ppu_oam_changed(&f->ppu, f->oam);
}

static int vram_write16(PnFrame *f, uint32_t off, uint16_t value) {
    if (!f->vramCopy) {
        /* First DMA write to VRAM: from now on the renderer reads a private copy. With a hook,
         * also remember the caller's VRAM so later hook writes can be merged into the copy. */
        f->vramCopy = (uint8_t *) malloc(f->hooked ? 2 * PN_VRAM_SIZE : PN_VRAM_SIZE);
        if (!f->vramCopy) {
            return 0;
        }
        memcpy(f->vramCopy, f->mem->vram, PN_VRAM_SIZE);
        if (f->hooked) {
            f->vramSeen = f->vramCopy + PN_VRAM_SIZE;
            memcpy(f->vramSeen, f->mem->vram, PN_VRAM_SIZE);
        }
        f->vram = f->vramCopy;
    }
    wr16(f->vramCopy, off & ~1u, value);
    pn_ppu_vram_changed(&f->ppu, f->vram);
    return 1;
}

/* ------------------------------------------------------------------ DMA (mGBA src/gba/dma.c semantics) */

static const int DMA_OFFSET[] = {1, -1, 0, 1};

static void dma_arm(PnFrame *f, int ch) {
    PnDmaChannel *d = &f->dma[ch];
    uint32_t base = 0x0B0u + (uint32_t) ch * 12u;
    uint32_t width = 2u << ((d->reg >> 10) & 1);
    d->nextSource = rd32(f->io, base) & ~(width - 1);
    d->nextDest = rd32(f->io, base + 4) & ~(width - 1);
}

static void dma_write_cnt_hi(PnFrame *f, int ch, uint16_t control) {
    PnDmaChannel *d = &f->dma[ch];
    int wasEnabled = (d->reg & 0x8000) != 0;
    control &= ch < 3 ? 0xF7E0 : 0xFFE0;
    d->reg = control;
    wr16(f->io, 0x0BAu + (uint32_t) ch * 12u, control);
    if (!wasEnabled && (control & 0x8000)) {
        dma_arm(f, ch);
    }
}

static int dma_timing(uint16_t reg) {
    return (reg >> 12) & 3;
}

static uint32_t dma_load(PnFrame *f, uint32_t addr, uint32_t width) {
    PnTarget t = classify(f, addr & ~(width - 1), width); /* the bus aligns to the access width */
    switch (t.region) {
    case RGN_IO:
        return width == 4 ? rd32(f->io, t.offset) : rd16(f->io, t.offset);
    case RGN_PLTT:
        return width == 4 ? rd32(f->pltt, t.offset) : rd16(f->pltt, t.offset);
    case RGN_VRAM:
        return width == 4 ? rd32(f->vram, t.offset) : rd16(f->vram, t.offset);
    case RGN_OAM:
        return width == 4 ? rd32(f->oam, t.offset) : rd16(f->oam, t.offset);
    case RGN_OTHER:
        return width == 4 ? rd32(t.ptr, 0) : rd16(t.ptr, 0);
    default:
        ++f->stats.dma_unresolved;
        return width == 4 ? f->dmaTransferRegister : (f->dmaTransferRegister & 0xFFFFu);
    }
}

static void dma_store(PnFrame *f, uint32_t addr, uint32_t value, uint32_t width) {
    PnTarget t = classify(f, addr & ~(width - 1), width); /* the bus aligns to the access width */
    uint32_t half;
    for (half = 0; half < width; half += 2) {
        uint16_t v = (uint16_t) (value >> (half * 8));
        switch (t.region) {
        case RGN_IO:
            io_write16(f, t.offset + half, v);
            break;
        case RGN_PLTT:
            pltt_write16(f, t.offset + half, v);
            break;
        case RGN_VRAM:
            if (!vram_write16(f, t.offset + half, v)) {
                ++f->stats.dma_unresolved;
            }
            break;
        case RGN_OAM:
            oam_write16(f, t.offset + half, v);
            break;
        case RGN_OTHER:
            if (half == 0) {
                ++f->stats.dma_dropped_writes;
            }
            break;
        default:
            if (half == 0) {
                ++f->stats.dma_unresolved;
            }
            break;
        }
    }
}

/* The transfers mGBA's GBADMARunHblank triggers at the HBlank of a visible line, serviced in
 * channel priority order (0 first), each moving `count` units, then reload/disable. */
static void dma_run_hblank(PnFrame *f) {
    int ch;
    for (ch = 0; ch < 4; ++ch) {
        PnDmaChannel *d = &f->dma[ch];
        if (!(d->reg & 0x8000) || dma_timing(d->reg) != DMA_TIMING_HBLANK) {
            continue;
        }
        uint32_t width = 2u << ((d->reg >> 10) & 1);
        int srcControl = (d->reg >> 7) & 3;
        int destControl = (d->reg >> 5) & 3;
        uint32_t n;
        for (n = 0; n < d->count; ++n) {
            uint32_t source = d->nextSource;
            uint32_t dest = d->nextDest;
            if (width == 4) {
                if (source) {
                    f->dmaTransferRegister = dma_load(f, source, 4);
                }
                dma_store(f, dest, f->dmaTransferRegister, 4);
            } else {
                if (source) {
                    f->dmaTransferRegister = dma_load(f, source, 2);
                    f->dmaTransferRegister |= f->dmaTransferRegister << 16;
                }
                dma_store(f, dest, f->dmaTransferRegister & 0xFFFFu, 2);
            }
            int sourceOffset;
            if (source >= 0x08000000u && source < 0x0E000000u && srcControl < 3) {
                sourceOffset = (int) width;
            } else {
                sourceOffset = DMA_OFFSET[srcControl] * (int) width;
            }
            if (source) {
                d->nextSource = source + (uint32_t) sourceOffset;
            }
            d->nextDest = dest + (uint32_t) (DMA_OFFSET[destControl] * (int) width);
            ++f->stats.hblank_dma_units;
        }
        if (!(d->reg & 0x0200)) { /* no repeat: the enable bit clears */
            d->reg &= 0x7FFF;
            wr16(f->io, 0x0BAu + (uint32_t) ch * 12u, (uint16_t) (rd16(f->io, 0x0BAu + (uint32_t) ch * 12u) & 0x7FE0));
        }
        if (destControl == 3) { /* increment/reload: mGBA reloads the DAD value as written */
            d->nextDest = rd32(f->io, 0x0B4u + (uint32_t) ch * 12u);
        }
    }
}

/* ------------------------------------------------------------------ hooks */

static void sync_after_hook(PnFrame *f) {
    const PnVideoMem *mem = f->mem;
    uint32_t off;
    /* I/O: video registers and DMA registers, in address order (SAD, DAD, CNT_L, CNT_H). */
    static const uint32_t ranges[2][2] = {{0x000, 0x060}, {0x0B0, 0x0E0}};
    int r;
    for (r = 0; r < 2; ++r) {
        for (off = ranges[r][0]; off < ranges[r][1]; off += 2) {
            uint16_t now = rd16(mem->io, off);
            if (now != rd16(f->ioSeen, off)) {
                wr16(f->ioSeen, off, now);
                if (off == 0x004 || off == 0x006) {
                    continue; /* DISPSTAT / VCOUNT */
                }
                io_write16(f, off, now);
            }
        }
    }
    if (memcmp(mem->pltt, f->plttSeen, PN_PLTT_SIZE) != 0) {
        for (off = 0; off < PN_PLTT_SIZE; off += 2) {
            uint16_t now = rd16(mem->pltt, off);
            if (now != rd16(f->plttSeen, off)) {
                wr16(f->plttSeen, off, now);
                pltt_write16(f, off, now);
            }
        }
    }
    if (memcmp(mem->oam, f->oamSeen, PN_OAM_SIZE) != 0) {
        for (off = 0; off < PN_OAM_SIZE; off += 2) {
            uint16_t now = rd16(mem->oam, off);
            if (now != rd16(f->oamSeen, off)) {
                wr16(f->oamSeen, off, now);
                wr16(f->oam, off, now);
            }
        }
        pn_ppu_oam_changed(&f->ppu, f->oam);
    }
    /* VRAM is read live from the caller's memory, unless a DMA write forced a private copy: then
     * the hook's VRAM writes are merged into it (last writer wins, like the bus). Either way the
     * per-row tile-map caches are dropped. */
    if (f->vramSeen && memcmp(mem->vram, f->vramSeen, PN_VRAM_SIZE) != 0) {
        for (off = 0; off < PN_VRAM_SIZE; ++off) {
            if (mem->vram[off] != f->vramSeen[off]) {
                f->vramSeen[off] = mem->vram[off];
                f->vramCopy[off] = mem->vram[off];
            }
        }
    }
    pn_ppu_vram_changed(&f->ppu, f->vram);
}

/* ------------------------------------------------------------------ output */

/* A composed row (0x00BBGGRR | flags) to the output formats. All variants write the same bytes. */

static void emit_rgb8(const uint32_t *row, uint8_t *o) {
    /* 4 pixels -> 3 little-endian words: R0 G0 B0 R1 | G1 B1 R2 G2 | B2 R3 G3 B3 (the flag byte
     * of each pixel is shifted out). */
    int x;
    for (x = 0; x < PN_SCREEN_W; x += 4, o += 12) {
        uint32_t p0 = row[x], p1 = row[x + 1], p2 = row[x + 2], p3 = row[x + 3];
        uint32_t w0 = (p0 & 0x00FFFFFFu) | (p1 << 24);
        uint32_t w1 = ((p1 >> 8) & 0x0000FFFFu) | (p2 << 16);
        uint32_t w2 = ((p2 >> 16) & 0x000000FFu) | (p3 << 8);
        memcpy(o, &w0, 4);
        memcpy(o + 4, &w1, 4);
        memcpy(o + 8, &w2, 4);
    }
}

static void emit_xbgr8888(const uint32_t *row, uint32_t *o) {
    int x;
#ifdef PN_HAVE_SSE2
    const __m128i mask = _mm_set1_epi32(0x00FFFFFF);
    for (x = 0; x < PN_SCREEN_W; x += 4) {
        _mm_storeu_si128((__m128i *) &o[x], _mm_and_si128(_mm_loadu_si128((const __m128i *) &row[x]), mask));
    }
#else
    for (x = 0; x < PN_SCREEN_W; ++x) {
        o[x] = row[x] & 0x00FFFFFFu;
    }
#endif
}

static void emit_bgr555(const uint32_t *row, uint16_t *o) {
    int x;
#ifdef PN_HAVE_SSE2
    const __m128i r = _mm_set1_epi32(0xF8), g = _mm_set1_epi32(0xF800), b = _mm_set1_epi32(0xF80000);
    for (x = 0; x < PN_SCREEN_W; x += 8) {
        __m128i c0 = _mm_loadu_si128((const __m128i *) &row[x]);
        __m128i c1 = _mm_loadu_si128((const __m128i *) &row[x + 4]);
        __m128i v0 = _mm_or_si128(_mm_or_si128(_mm_srli_epi32(_mm_and_si128(c0, r), 3), _mm_srli_epi32(_mm_and_si128(c0, g), 6)),
                                  _mm_srli_epi32(_mm_and_si128(c0, b), 9));
        __m128i v1 = _mm_or_si128(_mm_or_si128(_mm_srli_epi32(_mm_and_si128(c1, r), 3), _mm_srli_epi32(_mm_and_si128(c1, g), 6)),
                                  _mm_srli_epi32(_mm_and_si128(c1, b), 9));
        /* every value is < 0x8000: the signed saturating pack is exact */
        _mm_storeu_si128((__m128i *) &o[x], _mm_packs_epi32(v0, v1));
    }
#else
    for (x = 0; x < PN_SCREEN_W; ++x) {
        uint32_t c = row[x];
        o[x] = (uint16_t) (((c & 0xF8u) >> 3) | ((c & 0xF800u) >> 6) | ((c & 0xF80000u) >> 9));
    }
#endif
}

#ifdef PN_HAVE_AVX2
PN_AVX2_FN void emit_rgb8_avx2(const uint32_t *row, uint8_t *o) {
    /* 8 pixels -> 24 bytes: drop the flag byte inside each 128-bit lane, then close the gap. */
    const __m256i shuf = _mm256_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
                                          0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    const __m256i perm = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);
    int x;
    /* 32-byte stores whose last 8 bytes the next group overwrites; the last group of the row
     * stores exactly 24 bytes, so nothing outside the row is touched. */
    for (x = 0; x < PN_SCREEN_W - 8; x += 8, o += 24) {
        __m256i v = _mm256_loadu_si256((const __m256i *) &row[x]);
        _mm256_storeu_si256((__m256i *) o, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v, shuf), perm));
    }
    __m256i v = _mm256_loadu_si256((const __m256i *) &row[x]);
    v = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v, shuf), perm);
    _mm_storeu_si128((__m128i *) o, _mm256_castsi256_si128(v));
    _mm_storel_epi64((__m128i *) (o + 16), _mm256_extracti128_si256(v, 1));
}
#endif

/* PN_FMT_GRAY8 state of one render (pn_render_ex): scale, mode, output width and the per-output-row
 * accumulator of the box filter (sum of the lumas of the box rows seen so far, per output column). */
typedef struct PnGray {
    int k;
    int sub;
    int w;
    uint32_t acc[PN_SCREEN_W];
    /* AVX2 path: per input column, the sum of the lumas of the box rows seen so far (vacc), and the
     * lumas of the current line (lum, sub-sampled formats). 16 entries of padding, kept 0: the k = 4
     * reduction reads 256 entries. */
    uint16_t vacc[PN_SCREEN_W + 16];
    uint16_t lum[PN_SCREEN_W + 16];
} PnGray;

void pn_gray_dims(int scale, int *h, int *w) {
    if (scale < 1) {
        scale = 1;
    }
    if (h) {
        *h = (PN_SCREEN_H + scale - 1) / scale;
    }
    if (w) {
        *w = (PN_SCREEN_W + scale - 1) / scale;
    }
}

/* Luma of a composed row (0x00BBGGRR | flags, green swap applied), row y of the frame, into the gray
 * output (see PnRenderOpts for the arithmetic). Scalar on purpose: 240 multiply-adds per line. */
static void emit_gray8(const uint32_t *row, int y, PnGray *g, uint8_t *out) {
    uint32_t lum[PN_SCREEN_W];
    int x, j;
    const int k = g->k;
    for (x = 0; x < PN_SCREEN_W; ++x) {
        uint32_t c = row[x];
        lum[x] = (77u * (c & 0xFFu) + 150u * ((c >> 8) & 0xFFu) + 29u * ((c >> 16) & 0xFFu) + 128u) >> 8;
    }
    if (k == 1) {
        uint8_t *o = out + (size_t) y * PN_SCREEN_W;
        for (x = 0; x < PN_SCREEN_W; ++x) {
            o[x] = (uint8_t) lum[x];
        }
        return;
    }
    const int i = y / k;
    const int r0 = i * k;
    uint8_t *o = out + (size_t) i * (size_t) g->w;
    if (g->sub) { /* only the composed lines y = i*k get here */
        for (j = 0, x = 0; j < g->w; ++j) {
            int x1 = x + k < PN_SCREEN_W ? x + k : PN_SCREEN_W;
            uint32_t sum = 0, n = (uint32_t) (x1 - x);
            for (; x < x1; ++x) {
                sum += lum[x];
            }
            o[j] = (uint8_t) ((sum + n / 2) / n);
        }
        return;
    }
    for (j = 0, x = 0; j < g->w; ++j) {
        int x1 = x + k < PN_SCREEN_W ? x + k : PN_SCREEN_W;
        uint32_t sum = 0;
        for (; x < x1; ++x) {
            sum += lum[x];
        }
        g->acc[j] = (y == r0 ? 0u : g->acc[j]) + sum;
    }
    if (y - r0 == k - 1 || y == PN_SCREEN_H - 1) {
        const uint32_t rows = (uint32_t) (y - r0 + 1);
        for (j = 0; j < g->w; ++j) {
            int x0 = j * k;
            uint32_t cols = (uint32_t) ((x0 + k < PN_SCREEN_W ? x0 + k : PN_SCREEN_W) - x0);
            uint32_t n = rows * cols;
            o[j] = (uint8_t) ((g->acc[j] + n / 2) / n);
        }
    }
}

#ifdef PN_HAVE_AVX2
/* Luma of 8 pixels (0x00BBGGRR | flags) as 8 int32: (77 R + 150 G + 29 B + 128) >> 8, exactly the scalar
 * formula (every product and sum is a non-negative int32 < 65536). */
PN_AVX2_FN inline __m256i gray_luma8_avx2(const uint32_t *p) {
    const __m256i v = _mm256_loadu_si256((const __m256i *) p);
    const __m256i rb = _mm256_and_si256(v, _mm256_set1_epi32(0x00FF00FF));                  /* R | B << 16 */
    const __m256i g = _mm256_and_si256(_mm256_srli_epi32(v, 8), _mm256_set1_epi32(0xFF));   /* G | 0 << 16 */
    __m256i y = _mm256_add_epi32(_mm256_madd_epi16(rb, _mm256_set1_epi32(77 | (29 << 16))),
                                 _mm256_madd_epi16(g, _mm256_set1_epi32(150)));
    return _mm256_srli_epi32(_mm256_add_epi32(y, _mm256_set1_epi32(128)), 8);
}

/* Lumas of 16 pixels as 16 uint16, in pixel order. */
PN_AVX2_FN inline __m256i gray_luma16_avx2(const uint32_t *p) {
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(gray_luma8_avx2(p), gray_luma8_avx2(p + 8)), 0xD8);
}

/* 8 int32 means (each < 256) -> 8 bytes at o. */
PN_AVX2_FN inline void gray_store8_avx2(__m256i m, uint8_t *o) {
    __m256i t = _mm256_packus_epi32(m, m); /* lane: m0..m3 m0..m3 | m4..m7 m4..m7 (uint16) */
    t = _mm256_packus_epi16(t, t);         /* lane: m0..m3 x4 | m4..m7 x4 (bytes) */
    t = _mm256_permutevar8x32_epi32(t, _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4));
    _mm_storel_epi64((__m128i *) o, _mm256_castsi256_si128(t));
}

/* Output row from per-column sums v[0..239] (each the sum of `rows` lumas): output j = the rounded mean
 * (sum + n/2) / n of v[j*k .. j*k + k - 1], n = rows * k (k divides 240 for k = 2, 3, 4). */
PN_AVX2_FN void gray_reduce_avx2(const uint16_t *v, int k, int rows, uint8_t *o, int w) {
    const __m256i ones = _mm256_set1_epi16(1);
    int x, j;
    if (k == 2) { /* n = 2 or 4 */
        const __m128i shift = _mm_cvtsi32_si128(rows == 1 ? 1 : 2);
        const __m256i half = _mm256_set1_epi32(rows);
        for (x = 0, j = 0; x < PN_SCREEN_W; x += 16, j += 8) {
            __m256i m = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) &v[x]), ones);
            gray_store8_avx2(_mm256_srl_epi32(_mm256_add_epi32(m, half), shift), o + j);
        }
        return;
    }
    if (k == 4) { /* n = 4 or 16 */
        uint8_t tmp[64];
        const __m128i shift = _mm_cvtsi32_si128(rows == 1 ? 2 : 4);
        const __m256i half = _mm256_set1_epi32(rows == 1 ? 2 : 8);
        const __m256i order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        for (x = 0, j = 0; x < PN_SCREEN_W; x += 32, j += 8) { /* the last step reads the zero padding */
            __m256i a = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) &v[x]), ones);
            __m256i b = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) &v[x + 16]), ones);
            __m256i m = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(a, b), order);
            gray_store8_avx2(_mm256_srl_epi32(_mm256_add_epi32(m, half), shift), tmp + j);
        }
        memcpy(o, tmp, (size_t) w);
        return;
    }
    /* k = 3: n = 3 (one row) or 9; constant divisors */
    for (j = 0, x = 0; j < w; ++j, x += 3) {
        uint32_t sum = (uint32_t) v[x] + v[x + 1] + v[x + 2];
        o[j] = (uint8_t) (rows == 1 ? (sum + 1) / 3 : (sum + 4) / 9);
    }
}

/* emit_gray8 for k = 1..4 (the same bytes; the other scales use emit_gray8). */
PN_AVX2_FN void emit_gray8_avx2(const uint32_t *row, int y, PnGray *g, uint8_t *out) {
    const int k = g->k;
    int x;
    if (k == 1) {
        uint8_t *o = out + (size_t) y * PN_SCREEN_W;
        for (x = 0; x < PN_SCREEN_W; x += 16) {
            __m256i l = gray_luma16_avx2(row + x);
            l = _mm256_permute4x64_epi64(_mm256_packus_epi16(l, l), 0x08);
            _mm_storeu_si128((__m128i *) (o + x), _mm256_castsi256_si128(l));
        }
        return;
    }
    const int i = y / k;
    const int r0 = i * k;
    uint8_t *o = out + (size_t) i * (size_t) g->w;
    if (g->sub) { /* only the composed lines y = i*k get here */
        for (x = 0; x < PN_SCREEN_W; x += 16) {
            _mm256_storeu_si256((__m256i *) &g->lum[x], gray_luma16_avx2(row + x));
        }
        gray_reduce_avx2(g->lum, k, 1, o, g->w);
        return;
    }
    if (y == r0) {
        for (x = 0; x < PN_SCREEN_W; x += 16) {
            _mm256_storeu_si256((__m256i *) &g->vacc[x], gray_luma16_avx2(row + x));
        }
    } else {
        for (x = 0; x < PN_SCREEN_W; x += 16) {
            __m256i a = _mm256_loadu_si256((const __m256i *) &g->vacc[x]);
            _mm256_storeu_si256((__m256i *) &g->vacc[x], _mm256_add_epi16(a, gray_luma16_avx2(row + x)));
        }
    }
    if (y - r0 == k - 1 || y == PN_SCREEN_H - 1) {
        gray_reduce_avx2(g->vacc, k, y - r0 + 1, o, g->w);
    }
}
#endif

static void emit_line(const PnFrame *f, int y, int normal, PnPixelFormat fmt, void *out, PnGray *gray) {
    const uint32_t *row = f->ppu.row;
    uint32_t swapped[PN_SCREEN_W];
    int x;
    if (normal && f->ppu.greenswap) {
        for (x = 0; x < PN_SCREEN_W; x += 2) {
            swapped[x] = (row[x] & 0x00FF00FFu) | (row[x + 1] & 0x0000FF00u);
            swapped[x + 1] = (row[x + 1] & 0x00FF00FFu) | (row[x] & 0x0000FF00u);
        }
        row = swapped;
    }
    switch (fmt) {
    case PN_FMT_RGB8: {
        uint8_t *o = (uint8_t *) out + (size_t) y * PN_SCREEN_W * 3;
#ifdef PN_HAVE_AVX2
        if (f->ppu.simd == PN_PPU_SIMD_AVX2) {
            emit_rgb8_avx2(row, o);
            break;
        }
#endif
        emit_rgb8(row, o);
        break;
    }
    case PN_FMT_XBGR8888:
        emit_xbgr8888(row, (uint32_t *) out + (size_t) y * PN_SCREEN_W);
        break;
    case PN_FMT_BGR555:
        emit_bgr555(row, (uint16_t *) out + (size_t) y * PN_SCREEN_W);
        break;
    case PN_FMT_GRAY8:
#ifdef PN_HAVE_AVX2
        if (f->ppu.simd == PN_PPU_SIMD_AVX2 && gray->k <= 4) {
            emit_gray8_avx2(row, y, gray, (uint8_t *) out);
            break;
        }
#endif
        emit_gray8(row, y, gray, (uint8_t *) out);
        break;
    }
}

/* ------------------------------------------------------------------ entry points */

PnRenderIsa pn_render_auto_isa(void) {
    return pn_ppu_avx2_usable() ? PN_ISA_AVX2 : PN_ISA_GENERIC;
}

const char *pn_render_isa_name(PnRenderIsa isa) {
    switch (isa) {
    case PN_ISA_AUTO:
        return "auto";
    case PN_ISA_GENERIC:
        return "generic";
    case PN_ISA_AVX2:
        return "avx2";
    }
    return NULL;
}

void pn_render(const PnVideoMem *mem, const PnRenderHooks *hooks, PnPixelFormat fmt, void *out,
               PnRenderStats *stats) {
    pn_render_isa(mem, hooks, fmt, out, stats, PN_ISA_AUTO);
}

/* The frame driver. gray: PN_FMT_GRAY8 state (NULL for the other formats). With gray->sub, only the
 * lines y = 0, k, 2k, ... are composed; the others run everything but the compositing. */
static void render_frame(const PnVideoMem *mem, const PnRenderHooks *hooks, PnPixelFormat fmt, void *out,
                         PnRenderStats *stats, PnRenderIsa isa, PnGray *gray) {
    PnFrame f;
    uint32_t off;
    int ch, y;

    memset(&f.stats, 0, sizeof(f.stats));
    f.mem = mem;
    memcpy(f.io, mem->io, PN_IO_SIZE);
    memcpy(f.pltt, mem->pltt, PN_PLTT_SIZE);
    memcpy(f.oam, mem->oam, PN_OAM_SIZE);
    f.vram = mem->vram;
    f.vramCopy = NULL;
    f.vramSeen = NULL;
    f.dmaTransferRegister = 0;
    f.hooked = hooks && hooks->hblank;

    /* Video unit: replay every video register and palette entry as writes made during VBlank
     * (nextY == 0), which is the state mGBA's renderer has when it starts line 0. */
    pn_ppu_reset(&f.ppu, f.vram, f.oam);
    f.ppu.simd = (isa == PN_ISA_AUTO || isa == PN_ISA_AVX2) && pn_ppu_avx2_usable() ? PN_PPU_SIMD_AVX2
                                                                                    : PN_PPU_SIMD_NONE;
    f.ppu.nextY = 0;
    for (off = 0; off < 0x058; off += 2) {
        if (off == 0x004 || off == 0x006) {
            continue;
        }
        wr16(f.io, off, pn_ppu_write_video_register(&f.ppu, off, rd16(f.io, off)));
    }
    for (off = 0; off < PN_PLTT_SIZE; off += 2) {
        pn_ppu_write_palette(&f.ppu, off, rd16(f.pltt, off));
    }
    pn_ppu_latch_frame_start(&f.ppu);

    /* DMA controller: a channel enabled at the boundary is taken as freshly armed from its
     * SAD/DAD registers (FireRed re-arms its HBlank DMA in every VBlankIntr). */
    for (ch = 0; ch < 4; ++ch) {
        uint32_t base = 0x0B0u + (uint32_t) ch * 12u;
        uint16_t cntl = rd16(f.io, base + 8);
        uint32_t count = ch < 3 ? (cntl & 0x3FFFu) : cntl;
        f.dma[ch].count = count ? count : (ch == 3 ? 0x10000u : 0x4000u);
        f.dma[ch].reg = (uint16_t) (rd16(f.io, base + 10) & (ch < 3 ? 0xF7E0 : 0xFFE0));
        f.dma[ch].nextSource = 0;
        f.dma[ch].nextDest = 0;
        if (f.dma[ch].reg & 0x8000) {
            dma_arm(&f, ch);
        }
    }

    if (f.hooked) {
        memcpy(f.ioSeen, mem->io, PN_IO_SIZE);
        memcpy(f.plttSeen, mem->pltt, PN_PLTT_SIZE);
        memcpy(f.oamSeen, mem->oam, PN_OAM_SIZE);
    }

    for (y = 0; y < PN_SCREEN_H; ++y) {
        if (y > 0) {
            dma_run_hblank(&f);
        }
        if (f.hooked) {
            hooks->hblank(hooks->ctx, y);
            ++f.stats.hook_calls;
            sync_after_hook(&f);
        }
        unsigned mode = f.ppu.dispcnt & 7u;
        int compose = !gray || !gray->sub || y % gray->k == 0;
        int normal = compose ? pn_ppu_draw_scanline(&f.ppu, y) : pn_ppu_skip_scanline(&f.ppu, y);
        if (!normal) {
            ++f.stats.forced_blank_lines;
        } else if (mode >= 6) {
            ++f.stats.invalid_mode_lines;
        } else if (mode >= 3) {
            ++f.stats.bitmap_mode_lines;
        }
        if (compose) {
            emit_line(&f, y, normal, fmt, out, gray);
        }
    }

    free(f.vramCopy); /* vramSeen lives in the same block */
    if (stats) {
        *stats = f.stats;
    }
}

void pn_render_isa(const PnVideoMem *mem, const PnRenderHooks *hooks, PnPixelFormat fmt, void *out,
                   PnRenderStats *stats, PnRenderIsa isa) {
    render_frame(mem, hooks, fmt, out, stats, isa, NULL);
}

int pn_render_ex(const PnVideoMem *mem, const PnRenderHooks *hooks, const PnRenderOpts *opts, void *out,
                 PnRenderStats *stats) {
    PnRenderOpts o = {PN_FMT_RGB8, 1, 0, PN_ISA_AUTO};
    if (opts) {
        o = *opts;
    }
    if ((int) o.isa < (int) PN_ISA_AUTO || (int) o.isa > (int) PN_ISA_AVX2 || (o.line_sub != 0 && o.line_sub != 1)) {
        return -1;
    }
    switch (o.fmt) {
    case PN_FMT_RGB8:
    case PN_FMT_XBGR8888:
    case PN_FMT_BGR555:
        if (o.scale != 1 || o.line_sub) {
            return -1;
        }
        render_frame(mem, hooks, o.fmt, out, stats, o.isa, NULL);
        return 0;
    case PN_FMT_GRAY8: {
        if (o.scale < 1 || o.scale > PN_GRAY_MAX_SCALE) {
            return -1;
        }
        PnGray g;
        g.k = o.scale;
        g.sub = o.line_sub && o.scale > 1;
        pn_gray_dims(o.scale, NULL, &g.w);
        memset(g.vacc + PN_SCREEN_W, 0, 16 * sizeof(uint16_t));
        memset(g.lum + PN_SCREEN_W, 0, 16 * sizeof(uint16_t));
        render_frame(mem, hooks, o.fmt, out, stats, o.isa, &g);
        return 0;
    }
    }
    return -1;
}

void pn_render_rgb(const PnVideoMem *mem, const PnRenderHooks *hooks, uint8_t out[PN_SCREEN_H * PN_SCREEN_W * 3]) {
    pn_render(mem, hooks, PN_FMT_RGB8, out, NULL);
}

void pn_render_xbgr8888(const PnVideoMem *mem, const PnRenderHooks *hooks, uint32_t out[PN_SCREEN_H * PN_SCREEN_W]) {
    pn_render(mem, hooks, PN_FMT_XBGR8888, out, NULL);
}

void pn_render_bgr555(const PnVideoMem *mem, const PnRenderHooks *hooks, uint16_t out[PN_SCREEN_H * PN_SCREEN_W]) {
    pn_render(mem, hooks, PN_FMT_BGR555, out, NULL);
}
