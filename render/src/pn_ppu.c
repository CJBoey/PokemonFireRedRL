/* pn_ppu.c -- scanline compositor of fireredsim's renderer.
 *
 * Derived from mGBA 0.10.5 (commit 26b7884bc25a5933960f3cdcd98bac1ae14d42e2):
 *   src/gba/renderers/video-software.c, software-private.h, software-mode0.c,
 *   software-bg.c, software-obj.c, common.c and include/mgba/core/interface.h (colour helpers),
 * Copyright (c) 2013-2019 Jeffrey Pfau.
 *
 * Modifications for fireredsim (render/THIRD_PARTY.md): one translation unit; the state is
 * rebuilt from a memory snapshot each frame (pn_render.c) instead of being fed by bus writes;
 * scanline dirty-tracking/caching, highlighting, debug layer toggles, cache-set hooks and the
 * output-buffer handling are removed; colour is always mGBA's 32-bit color_t (0x00BBGGRR) with
 * 8-bit-per-channel blending exactly as mGBA's default build; window list sized defensively.
 * Every pixel-affecting branch, including mGBA's quirks (partial text tiles reading OBJ VRAM,
 * the Is/Get bit-field comparison in the OBJ window path, mosaic edge behaviour, the OBJ cycle
 * budget, the BG enable latency), is kept as in mGBA so the output matches its screenshots.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#include "pn_ppu.h"

#include <string.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "pn_ppu.c assumes a little-endian host (as mGBA's LOAD_16/LOAD_32 on x86)"
#endif

/* SIMD (fireredsim performance work, render/README.md "Performance"): SSE2 helpers where the
 * compiler targets SSE2 (x86-64; the i386 builds pass -msse2), AVX2 kernels compiled with a
 * function-level target attribute and selected at run time (PnPpu.simd). Every kernel is a
 * lane-wise transcription of the scalar mGBA code next to it; the scalar code stays the
 * reference and the fallback. PNRENDER_NO_AVX2 compiles the AVX2 kernels out. */
#if defined(__SSE2__)
#include <emmintrin.h>
#define PN_HAVE_SSE2 1
#endif
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && !defined(PNRENDER_NO_AVX2)
#include <immintrin.h>
#define PN_HAVE_AVX2 1
#define PN_AVX2_FN __attribute__((target("avx2"))) static
#define PN_AVX2_INLINE __attribute__((always_inline, target("avx2"))) static inline
#endif

int pn_ppu_avx2_usable(void) {
#ifdef PN_HAVE_AVX2
    /* Reads libgcc's CPU model (filled once by its constructor; checks OS YMM support too). */
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------ constants */

#define GBA_VIDEO_HORIZONTAL_PIXELS 240
#define GBA_VIDEO_VERTICAL_PIXELS 160
#define VIDEO_VERTICAL_TOTAL_PIXELS 228
#define OBJ_HBLANK_FREE_LENGTH 954
#define OBJ_LENGTH 1210
#define BASE_TILE 0x00010000
#define ENABLED_MAX 4

#define M_COLOR_RED 0x000000FFu
#define M_COLOR_GREEN 0x0000FF00u
#define M_COLOR_BLUE 0x00FF0000u
#define M_COLOR_WHITE 0x00FFFFFFu

#define OBJ_MODE_NORMAL 0
#define OBJ_MODE_SEMITRANSPARENT 1
#define OBJ_MODE_OBJWIN 2

enum { OFFSET_PRIORITY = 30, OFFSET_INDEX = 28 };

#define FLAG_PRIORITY 0xC0000000u
#define FLAG_INDEX 0x30000000u
#define FLAG_IS_BACKGROUND 0x08000000u
#define FLAG_UNWRITTEN 0xFC000000u
#define FLAG_REBLEND 0x04000000u
#define FLAG_TARGET_1 0x02000000u
#define FLAG_TARGET_2 0x01000000u
#define FLAG_OBJWIN 0x01000000u
#define FLAG_ORDER_MASK 0xF8000000u

#define IS_WRITABLE(PIXEL) ((PIXEL) &0xFE000000u)

#define LIKELY(X) __builtin_expect(!!(X), 1)
#define UNLIKELY(X) __builtin_expect(!!(X), 0)

/* mGBA I/O register offsets (video). */
#define REG_DISPCNT 0x000
#define REG_GREENSWP 0x002
#define REG_BG0CNT 0x008
#define REG_BG1CNT 0x00A
#define REG_BG2CNT 0x00C
#define REG_BG3CNT 0x00E
#define REG_BG0HOFS 0x010
#define REG_BG0VOFS 0x012
#define REG_BG1HOFS 0x014
#define REG_BG1VOFS 0x016
#define REG_BG2HOFS 0x018
#define REG_BG2VOFS 0x01A
#define REG_BG3HOFS 0x01C
#define REG_BG3VOFS 0x01E
#define REG_BG2PA 0x020
#define REG_BG2PB 0x022
#define REG_BG2PC 0x024
#define REG_BG2PD 0x026
#define REG_BG2X_LO 0x028
#define REG_BG2X_HI 0x02A
#define REG_BG2Y_LO 0x02C
#define REG_BG2Y_HI 0x02E
#define REG_BG3PA 0x030
#define REG_BG3PB 0x032
#define REG_BG3PC 0x034
#define REG_BG3PD 0x036
#define REG_BG3X_LO 0x038
#define REG_BG3X_HI 0x03A
#define REG_BG3Y_LO 0x03C
#define REG_BG3Y_HI 0x03E
#define REG_WIN0H 0x040
#define REG_WIN1H 0x042
#define REG_WIN0V 0x044
#define REG_WIN1V 0x046
#define REG_WININ 0x048
#define REG_WINOUT 0x04A
#define REG_MOSAIC 0x04C
#define REG_BLDCNT 0x050
#define REG_BLDALPHA 0x052
#define REG_BLDY 0x054

/* mGBA's DECL_BITS accessors: "Is" returns the masked (unshifted) bits, "Get" the field value.
 * The distinction matters (see the OBJ-window comparison in pn_preprocess_sprite). */
#define GBARegisterDISPCNTGetMode(v) ((v) &0x7)
#define GBARegisterDISPCNTIsFrameSelect(v) ((v) &0x10)
#define GBARegisterDISPCNTIsHblankIntervalFree(v) ((v) &0x20)
#define GBARegisterDISPCNTIsObjCharacterMapping(v) ((v) &0x40)
#define GBARegisterDISPCNTIsForcedBlank(v) ((v) &0x80)
#define GBARegisterDISPCNTGetBg0Enable(v) (((v) >> 8) & 1)
#define GBARegisterDISPCNTGetBg1Enable(v) (((v) >> 9) & 1)
#define GBARegisterDISPCNTGetBg2Enable(v) (((v) >> 10) & 1)
#define GBARegisterDISPCNTGetBg3Enable(v) (((v) >> 11) & 1)
#define GBARegisterDISPCNTIsObjEnable(v) ((v) &0x1000)
#define GBARegisterDISPCNTIsWin0Enable(v) ((v) &0x2000)
#define GBARegisterDISPCNTIsWin1Enable(v) ((v) &0x4000)
#define GBARegisterDISPCNTIsObjwinEnable(v) ((v) &0x8000)

#define GBAWindowControlIsBg0Enable(v) ((v) &0x01)
#define GBAWindowControlIsBg1Enable(v) ((v) &0x02)
#define GBAWindowControlIsBg2Enable(v) ((v) &0x04)
#define GBAWindowControlIsBg3Enable(v) ((v) &0x08)
#define GBAWindowControlIsObjEnable(v) ((v) &0x10)
#define GBAWindowControlIsBlendEnable(v) ((v) &0x20)
#define GBAWindowControlGetBlendEnable(v) (((v) >> 5) & 1)

#define GBAObjAttributesAGetY(a) ((a) &0xFF)
#define GBAObjAttributesAIsTransformed(a) ((a) &0x100)
#define GBAObjAttributesAIsDisable(a) ((a) &0x200)
#define GBAObjAttributesAGetDoubleSize(a) (((a) >> 9) & 1)
#define GBAObjAttributesAGetMode(a) (((a) >> 10) & 3)
#define GBAObjAttributesAIsMosaic(a) ((a) &0x1000)
#define GBAObjAttributesAIs256Color(a) ((a) &0x2000)
#define GBAObjAttributesAGetShape(a) (((a) >> 14) & 3)
#define GBAObjAttributesBGetX(b) ((b) &0x1FF)
#define GBAObjAttributesBGetMatIndex(b) (((b) >> 9) & 0x1F)
#define GBAObjAttributesBIsHFlip(b) ((b) &0x1000)
#define GBAObjAttributesBIsVFlip(b) ((b) &0x2000)
#define GBAObjAttributesBGetSize(b) (((b) >> 14) & 3)
#define GBAObjAttributesCGetTile(c) ((c) &0x3FF)
#define GBAObjAttributesCGetPriority(c) (((c) >> 10) & 3)
#define GBAObjAttributesCGetPalette(c) (((c) >> 12) & 0xF)

#define GBARegisterBGCNTGetPriority(v) ((v) &3)
#define GBARegisterBGCNTGetCharBase(v) (((v) >> 2) & 3)
#define GBARegisterBGCNTGetMosaic(v) (((v) >> 6) & 1)
#define GBARegisterBGCNTGet256Color(v) (((v) >> 7) & 1)
#define GBARegisterBGCNTGetScreenBase(v) (((v) >> 8) & 0x1F)
#define GBARegisterBGCNTGetOverflow(v) (((v) >> 13) & 1)
#define GBARegisterBGCNTGetSize(v) (((v) >> 14) & 3)

#define GBARegisterBLDCNTGetTarget1Bg0(v) ((v) &1)
#define GBARegisterBLDCNTGetTarget1Bg1(v) (((v) >> 1) & 1)
#define GBARegisterBLDCNTGetTarget1Bg2(v) (((v) >> 2) & 1)
#define GBARegisterBLDCNTGetTarget1Bg3(v) (((v) >> 3) & 1)
#define GBARegisterBLDCNTGetTarget1Obj(v) (((v) >> 4) & 1)
#define GBARegisterBLDCNTGetTarget1Bd(v) (((v) >> 5) & 1)
#define GBARegisterBLDCNTGetEffect(v) (((v) >> 6) & 3)
#define GBARegisterBLDCNTGetTarget2Bg0(v) (((v) >> 8) & 1)
#define GBARegisterBLDCNTGetTarget2Bg1(v) (((v) >> 9) & 1)
#define GBARegisterBLDCNTGetTarget2Bg2(v) (((v) >> 10) & 1)
#define GBARegisterBLDCNTGetTarget2Bg3(v) (((v) >> 11) & 1)
#define GBARegisterBLDCNTGetTarget2Obj(v) (((v) >> 12) & 1)
#define GBARegisterBLDCNTGetTarget2Bd(v) (((v) >> 13) & 1)

#define GBAMosaicControlGetBgH(m) ((m) &0xF)
#define GBAMosaicControlGetBgV(m) (((m) >> 4) & 0xF)
#define GBAMosaicControlGetObjH(m) (((m) >> 8) & 0xF)
#define GBAMosaicControlGetObjV(m) (((m) >> 12) & 0xF)

#define GBA_TEXT_MAP_TILE(MAP) ((MAP) &0x03FF)
#define GBA_TEXT_MAP_HFLIP(MAP) ((MAP) &0x0400)
#define GBA_TEXT_MAP_VFLIP(MAP) ((MAP) &0x0800)
#define GBA_TEXT_MAP_PALETTE(MAP) (((MAP) &0xF000) >> 12)

static const int GBAVideoObjSizes[16][2] = {
    {8, 8},   {16, 16}, {32, 32}, {64, 64}, {16, 8}, {32, 8}, {32, 16}, {64, 32},
    {8, 16},  {8, 32},  {16, 32}, {32, 64}, {0, 0},  {0, 0}, {0, 0},   {0, 0},
};

/* ------------------------------------------------------------------ memory loads */

static inline uint32_t ld16(const uint8_t *base, uint32_t offset) {
    uint16_t v;
    memcpy(&v, base + offset, 2);
    return v;
}

static inline uint32_t ld32(const uint8_t *base, uint32_t offset) {
    uint32_t v;
    memcpy(&v, base + offset, 4);
    return v;
}

#define LOAD_16(DEST, ADDR, BASE) (DEST) = ld16((const uint8_t *) (BASE), (uint32_t) (ADDR))
#define LOAD_32(DEST, ADDR, BASE) (DEST) = ld32((const uint8_t *) (BASE), (uint32_t) (ADDR))

/* ------------------------------------------------------------------ colour (mGBA 32-bit color_t) */

static inline uint32_t mColorFrom555(uint16_t value) {
    uint32_t color = ((uint32_t) (value & 0x001F) << 3) | ((uint32_t) (value & 0x03E0) << 6) |
                     ((uint32_t) (value & 0x7C00) << 9);
    color |= (color >> 5) & 0x070707;
    return color;
}

static inline unsigned mColorMix5Bit(int weightA, unsigned colorA, int weightB, unsigned colorB) {
    unsigned c = 0;
    unsigned a, b;
    a = colorA & 0xFF;
    b = colorB & 0xFF;
    c |= ((a * weightA + b * weightB) / 16) & 0x1FF;
    if (c & 0x00000100) {
        c = 0x000000FF;
    }

    a = colorA & 0xFF00;
    b = colorB & 0xFF00;
    c |= ((a * weightA + b * weightB) / 16) & 0x1FF00;
    if (c & 0x00010000) {
        c = (c & 0x000000FF) | 0x0000FF00;
    }

    a = colorA & 0xFF0000;
    b = colorB & 0xFF0000;
    c |= ((a * weightA + b * weightB) / 16) & 0x1FF0000;
    if (c & 0x01000000) {
        c = (c & 0x0000FFFF) | 0x00FF0000;
    }
    return c;
}

static inline unsigned _brighten(unsigned color, int y) {
    unsigned c = 0;
    unsigned a;
    a = color & 0xFF;
    c |= (a + ((0xFF - a) * y) / 16) & 0xFF;

    a = color & 0xFF00;
    c |= (a + ((0xFF00 - a) * y) / 16) & 0xFF00;

    a = color & 0xFF0000;
    c |= (a + ((0xFF0000 - a) * y) / 16) & 0xFF0000;
    return c;
}

static inline unsigned _darken(unsigned color, int y) {
    unsigned c = 0;
    unsigned a;
    a = color & 0xFF;
    c |= (a - (a * y) / 16) & 0xFF;

    a = color & 0xFF00;
    c |= (a - (a * y) / 16) & 0xFF00;

    a = color & 0xFF0000;
    c |= (a - (a * y) / 16) & 0xFF0000;
    return c;
}

/* ------------------------------------------------------------------ compositing (software-private.h) */

/* The row buffer holds colour | flags; flags sit in the top byte so that a single unsigned
 * comparison orders pixels front to back (lower = in front). Layers are drawn front to back. */

static inline void _compositeBlendObjwin(PnPpu *renderer, uint32_t *pixel, uint32_t color, uint32_t current) {
    if (color >= current) {
        if (current & FLAG_TARGET_1 && color & FLAG_TARGET_2) {
            color = mColorMix5Bit(renderer->blda, current, renderer->bldb, color);
        } else {
            color = current & (0x00FFFFFF | FLAG_REBLEND | FLAG_OBJWIN);
        }
    } else {
        color = (color & ~FLAG_TARGET_2) | (current & FLAG_OBJWIN);
    }
    *pixel = color;
}

static inline void _compositeBlendNoObjwin(PnPpu *renderer, uint32_t *pixel, uint32_t color, uint32_t current) {
    if (color >= current) {
        if (current & FLAG_TARGET_1 && color & FLAG_TARGET_2) {
            color = mColorMix5Bit(renderer->blda, current, renderer->bldb, color);
        } else {
            color = current & (0x00FFFFFF | FLAG_REBLEND | FLAG_OBJWIN);
        }
    } else {
        color = color & ~FLAG_TARGET_2;
    }
    *pixel = color;
}

static inline void _compositeNoBlendObjwin(PnPpu *renderer, uint32_t *pixel, uint32_t color, uint32_t current) {
    (void) renderer;
    if (color < current) {
        color |= (current & FLAG_OBJWIN);
    } else {
        color = current & (0x00FFFFFF | FLAG_REBLEND | FLAG_OBJWIN);
    }
    *pixel = color;
}

static inline void _compositeNoBlendNoObjwin(PnPpu *renderer, uint32_t *pixel, uint32_t color, uint32_t current) {
    (void) renderer;
    if (color >= current) {
        color = current & (0x00FFFFFF | FLAG_REBLEND | FLAG_OBJWIN);
    }
    *pixel = color;
}

/* ------------------------------------------------------------------ AVX2 compositing (8 pixels per op) */

#ifdef PN_HAVE_AVX2

/* table[idx] for 8 lanes, idx in 0..15 (a 4bpp sub-palette): two in-register permutes, the
 * bit 3 of each index picks the half. Reads table[0..15]. */
PN_AVX2_INLINE __m256i pn_lut16_avx2(const uint32_t *table, __m256i idx) {
    __m256i lo = _mm256_loadu_si256((const __m256i *) table);
    __m256i hi = _mm256_loadu_si256((const __m256i *) (table + 8));
    __m256 a = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(lo, idx));
    __m256 b = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(hi, idx));
    return _mm256_castps_si256(_mm256_blendv_ps(a, b, _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28))));
}

/* mColorMix5Bit(weightA, colorA, weightB, colorB) on 8 lanes: every channel
 * ((a * weightA + b * weightB) / 16) & 0x1FF, saturated to 0xFF when bit 8 is set (the sum is
 * at most 510, so that is min(255, ...)), flag byte cleared. 16-bit lanes cannot overflow
 * (weights <= 16: at most 8160); packus saturates exactly like the scalar code. */
PN_AVX2_INLINE __m256i pn_mix_avx2(__m256i weightA, __m256i colorA, __m256i weightB, __m256i colorB) {
    const __m256i z = _mm256_setzero_si256();
    __m256i lo = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_unpacklo_epi8(colorA, z), weightA),
                                  _mm256_mullo_epi16(_mm256_unpacklo_epi8(colorB, z), weightB));
    __m256i hi = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_unpackhi_epi8(colorA, z), weightA),
                                  _mm256_mullo_epi16(_mm256_unpackhi_epi8(colorB, z), weightB));
    lo = _mm256_srli_epi16(lo, 4);
    hi = _mm256_srli_epi16(hi, 4);
    return _mm256_and_si256(_mm256_packus_epi16(lo, hi), _mm256_set1_epi32(0x00FFFFFF));
}

/* a < b, unsigned, per lane (all ones / zero). */
PN_AVX2_INLINE __m256i pn_ltu_avx2(__m256i a, __m256i b) {
    const __m256i sign = _mm256_set1_epi32((int) 0x80000000u);
    return _mm256_cmpgt_epi32(_mm256_xor_si256(b, sign), _mm256_xor_si256(a, sign));
}

/* The four _composite{Blend,NoBlend}{Objwin,NoObjwin} functions above, lane-wise: lanes in `keep`
 * keep `current`, the others get the composited value of `color` over `current`. */
PN_AVX2_INLINE __m256i pn_composite_keep_avx2(const PnPpu *renderer, __m256i current, __m256i color, __m256i keep,
                                              const int blend, const int objwin) {
    /* color >= current (unsigned) */
    __m256i ge = _mm256_cmpeq_epi32(_mm256_max_epu32(color, current), color);
    __m256i covered = _mm256_and_si256(current, _mm256_set1_epi32(0x00FFFFFF | FLAG_REBLEND | FLAG_OBJWIN));
    __m256i front;
    __m256i ow = _mm256_setzero_si256();
    if (objwin) {
        ow = _mm256_and_si256(current, _mm256_set1_epi32(FLAG_OBJWIN));
    }
    if (blend) {
        const __m256i t1 = _mm256_set1_epi32(FLAG_TARGET_1);
        const __m256i t2 = _mm256_set1_epi32(FLAG_TARGET_2);
        front = _mm256_and_si256(color, _mm256_set1_epi32((int) ~FLAG_TARGET_2));
        if (objwin) {
            front = _mm256_or_si256(front, ow);
        }
        __m256i mixm = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(current, t1), t1),
                                        _mm256_cmpeq_epi32(_mm256_and_si256(color, t2), t2));
        mixm = _mm256_andnot_si256(keep, _mm256_and_si256(ge, mixm));
        if (!_mm256_testz_si256(mixm, mixm)) {
            __m256i mixed = pn_mix_avx2(_mm256_set1_epi16((short) renderer->blda), current,
                                        _mm256_set1_epi16((short) renderer->bldb), color);
            covered = _mm256_blendv_epi8(covered, mixed, mixm);
        }
    } else {
        front = objwin ? _mm256_or_si256(color, ow) : color;
    }
    return _mm256_blendv_epi8(_mm256_blendv_epi8(front, covered, ge), current, keep);
}

/* Same with the lanes to composite given (`m`) instead of the lanes to keep. */
PN_AVX2_INLINE __m256i pn_composite_avx2(const PnPpu *renderer, __m256i current, __m256i color, __m256i m,
                                         const int blend, const int objwin) {
    return pn_composite_keep_avx2(renderer, current, color, _mm256_xor_si256(m, _mm256_set1_epi32(-1)), blend, objwin);
}

#endif /* PN_HAVE_AVX2 */

#define COMPOSITE_16_OBJWIN(BLEND, IDX)                                                                     \
    if (objwinForceEnable || (!(current & FLAG_OBJWIN)) == objwinOnly) {                                    \
        unsigned color = (current & FLAG_OBJWIN) ? objwinPalette[paletteData | pixelData] : palette[pixelData]; \
        unsigned mergedFlags = flags;                                                                       \
        if (current & FLAG_OBJWIN) {                                                                        \
            mergedFlags = objwinFlags;                                                                      \
        }                                                                                                   \
        _composite##BLEND##Objwin(renderer, &pixel[IDX], color | mergedFlags, current);                    \
    }

#define COMPOSITE_16_NO_OBJWIN(BLEND, IDX) \
    _composite##BLEND##NoObjwin(renderer, &pixel[IDX], palette[pixelData] | flags, current);

#define COMPOSITE_256_OBJWIN(BLEND, IDX)                                                       \
    if (objwinForceEnable || (!(current & FLAG_OBJWIN)) == objwinOnly) {                       \
        unsigned color = (current & FLAG_OBJWIN) ? objwinPalette[pixelData] : palette[pixelData]; \
        unsigned mergedFlags = flags;                                                          \
        if (current & FLAG_OBJWIN) {                                                           \
            mergedFlags = objwinFlags;                                                         \
        }                                                                                      \
        _composite##BLEND##Objwin(renderer, &pixel[IDX], color | mergedFlags, current);       \
    }

#define COMPOSITE_256_NO_OBJWIN COMPOSITE_16_NO_OBJWIN

#define BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, IDX) \
    pixelData = tileData & 0xF;                      \
    current = pixel[IDX];                            \
    if (pixelData && IS_WRITABLE(current)) {         \
        COMPOSITE_16_##OBJWIN(BLEND, IDX);           \
    }                                                \
    tileData >>= 4;

#define BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, IDX) \
    pixelData = tileData & 0xFF;                      \
    current = pixel[IDX];                             \
    if (pixelData && IS_WRITABLE(current)) {          \
        COMPOSITE_256_##OBJWIN(BLEND, IDX);           \
    }                                                 \
    tileData >>= 8;

#define PREPARE_OBJWIN                                                                               \
    int objwinSlowPath = GBARegisterDISPCNTIsObjwinEnable(renderer->dispcnt);                       \
    int objwinOnly = 0;                                                                              \
    int objwinForceEnable = 0;                                                                       \
    (void) objwinForceEnable;                                                                        \
    const uint32_t *objwinPalette = renderer->normalPalette;                                         \
    (void) objwinPalette;                                                                            \
    if (objwinSlowPath) {                                                                            \
        if (background->target1 && GBAWindowControlIsBlendEnable(renderer->objwin.packed) &&         \
            (renderer->blendEffect == BLEND_BRIGHTEN || renderer->blendEffect == BLEND_DARKEN)) {    \
            objwinPalette = renderer->variantPalette;                                                \
        }                                                                                            \
        switch (background->index) {                                                                 \
        case 0:                                                                                      \
            objwinForceEnable = GBAWindowControlIsBg0Enable(renderer->objwin.packed) &&              \
                                GBAWindowControlIsBg0Enable(renderer->currentWindow.packed);         \
            objwinOnly = !GBAWindowControlIsBg0Enable(renderer->objwin.packed);                      \
            break;                                                                                   \
        case 1:                                                                                      \
            objwinForceEnable = GBAWindowControlIsBg1Enable(renderer->objwin.packed) &&              \
                                GBAWindowControlIsBg1Enable(renderer->currentWindow.packed);         \
            objwinOnly = !GBAWindowControlIsBg1Enable(renderer->objwin.packed);                      \
            break;                                                                                   \
        case 2:                                                                                      \
            objwinForceEnable = GBAWindowControlIsBg2Enable(renderer->objwin.packed) &&              \
                                GBAWindowControlIsBg2Enable(renderer->currentWindow.packed);         \
            objwinOnly = !GBAWindowControlIsBg2Enable(renderer->objwin.packed);                      \
            break;                                                                                   \
        case 3:                                                                                      \
            objwinForceEnable = GBAWindowControlIsBg3Enable(renderer->objwin.packed) &&              \
                                GBAWindowControlIsBg3Enable(renderer->currentWindow.packed);         \
            objwinOnly = !GBAWindowControlIsBg3Enable(renderer->objwin.packed);                      \
            break;                                                                                   \
        }                                                                                            \
    }

#define BACKGROUND_BITMAP_INIT                                                                     \
    int32_t x = background->sx + (renderer->start - 1) * background->dx;                           \
    int32_t y = background->sy + (renderer->start - 1) * background->dy;                           \
    int mosaicH = 0;                                                                               \
    int mosaicWait = 0;                                                                            \
    int32_t localX = 0; /* only read on the mosaic path (mGBA leaves them uninitialised) */      \
    int32_t localY = 0;                                                                            \
    if (background->mosaic) {                                                                      \
        int mosaicV = GBAMosaicControlGetBgV(renderer->mosaic) + 1;                                \
        mosaicH = GBAMosaicControlGetBgH(renderer->mosaic) + 1;                                    \
        mosaicWait = (mosaicH - renderer->start + GBA_VIDEO_HORIZONTAL_PIXELS * mosaicH) % mosaicH; \
        int32_t startX = renderer->start - (renderer->start % mosaicH);                            \
        --mosaicH;                                                                                 \
        localX = -(inY % mosaicV) * background->dmx;                                               \
        localY = -(inY % mosaicV) * background->dmy;                                               \
        x += localX;                                                                               \
        y += localY;                                                                               \
        localX += background->sx + startX * background->dx;                                       \
        localY += background->sy + startX * background->dy;                                        \
    }                                                                                              \
                                                                                                   \
    uint32_t flags = background->flags;                                                            \
    uint32_t objwinFlags = background->objwinFlags;                                                \
    bool variant = background->variant;                                                            \
    const uint32_t *palette = renderer->normalPalette;                                             \
    if (variant) {                                                                                 \
        palette = renderer->variantPalette;                                                        \
    }                                                                                              \
    (void) palette;                                                                                \
    PREPARE_OBJWIN;

/* mGBA's TEST_LAYER_ENABLED(X) (BG X enabled, visible in the current window or the OBJ window, at
 * `priority`) is evaluated per window segment in pn_ppu_draw_scanline (layersAt). */

/* ================================================================== BG mode 0 (software-mode0.c) */

#define BACKGROUND_TEXT_SELECT_CHARACTER       \
    xBase = localX & 0xF8;                     \
    if (background->size & 1) {                \
        xBase += (localX & 0x100) << 5;        \
    }                                          \
    screenBase = yBase + (xBase >> 3);         \
    LOAD_16(mapData, screenBase << 1, vram);

#define DRAW_BACKGROUND_MODE_0_TILE_SUFFIX_16(BLEND, OBJWIN)                                       \
    paletteData = GBA_TEXT_MAP_PALETTE(mapData) << 4;                                              \
    palette = &mainPalette[paletteData];                                                           \
    charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 5)) + (localY << 2);         \
    LOAD_32(tileData, charBase, vram);                                                             \
    if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                            \
        tileData >>= 4 * mod8;                                                                     \
        for (; outX < end; ++outX, ++pixel) {                                                      \
            BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 0);                                            \
        }                                                                                          \
    } else {                                                                                       \
        for (outX = end - 1; outX >= renderer->start; --outX) {                                    \
            uint32_t *pixel = &renderer->row[outX];                                                \
            BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 0);                                            \
        }                                                                                          \
    }

#define DRAW_BACKGROUND_MODE_0_TILE_PREFIX_16(BLEND, OBJWIN)                                       \
    charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 5)) + (localY << 2);         \
    LOAD_32(tileData, charBase, vram);                                                             \
    paletteData = GBA_TEXT_MAP_PALETTE(mapData) << 4;                                              \
    palette = &mainPalette[paletteData];                                                           \
    pixel = &renderer->row[outX];                                                                  \
    if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                            \
        if (outX < renderer->start) {                                                              \
            tileData >>= 4 * (renderer->start - outX);                                             \
            outX = renderer->start;                                                                \
            pixel = &renderer->row[outX];                                                          \
        }                                                                                          \
        for (; outX < renderer->end; ++outX, ++pixel) {                                            \
            BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 0);                                            \
        }                                                                                          \
    } else {                                                                                       \
        tileData >>= 4 * (0x8 - mod8);                                                             \
        int end = renderer->end - 8;                                                               \
        if (end < -1) {                                                                            \
            end = -1;                                                                              \
        }                                                                                          \
        outX = renderer->end - 1;                                                                  \
        pixel = &renderer->row[outX];                                                              \
        for (; outX > end; --outX, --pixel) {                                                      \
            BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 0);                                            \
        }                                                                                          \
    }

#define DRAW_BACKGROUND_MODE_0_MOSAIC_16(BLEND, OBJWIN)                                            \
    x = inX & 7;                                                                                   \
    if (mosaicWait) {                                                                              \
        int baseX = x - (mosaicH - mosaicWait);                                                    \
        if (baseX < 0) {                                                                           \
            int disturbX = (16 + baseX) >> 3;                                                      \
            inX -= disturbX << 3;                                                                  \
            localX = tileX * 8 + inX;                                                              \
            BACKGROUND_TEXT_SELECT_CHARACTER;                                                      \
            localY = inY & 0x7;                                                                    \
            if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                     \
                localY = 7 - localY;                                                               \
            }                                                                                      \
            baseX -= disturbX << 3;                                                                \
            inX += disturbX << 3;                                                                  \
        } else {                                                                                   \
            localX = tileX * 8 + inX;                                                              \
            BACKGROUND_TEXT_SELECT_CHARACTER;                                                      \
            localY = inY & 0x7;                                                                    \
            if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                     \
                localY = 7 - localY;                                                               \
            }                                                                                      \
        }                                                                                          \
        charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 5)) + (localY << 2);     \
        if (UNLIKELY(charBase >= 0x10000)) {                                                       \
            carryData = 0;                                                                         \
        } else {                                                                                   \
            paletteData = GBA_TEXT_MAP_PALETTE(mapData) << 4;                                      \
            palette = &mainPalette[paletteData];                                                   \
            LOAD_32(tileData, charBase, vram);                                                     \
            if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                    \
                tileData >>= 4 * baseX;                                                            \
            } else {                                                                               \
                tileData >>= 4 * (7 - baseX);                                                      \
            }                                                                                      \
            tileData &= 0xF;                                                                       \
            tileData |= tileData << 4;                                                             \
            tileData |= tileData << 8;                                                             \
            tileData |= tileData << 16;                                                            \
            carryData = tileData;                                                                  \
        }                                                                                          \
    }                                                                                              \
    localX = tileX * 8 + inX;                                                                      \
    for (; length; ++tileX) {                                                                      \
        mapData = background->mapCache[(localX >> 3) & 0x3F];                                      \
        localX += 8;                                                                               \
        localY = inY & 0x7;                                                                        \
        if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                         \
            localY = 7 - localY;                                                                   \
        }                                                                                          \
        charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 5)) + (localY << 2);     \
        tileData = carryData;                                                                      \
        for (; x < 8 && length; ++x, --length) {                                                   \
            if (!mosaicWait) {                                                                     \
                if (UNLIKELY(charBase >= 0x10000)) {                                               \
                    carryData = 0;                                                                 \
                } else {                                                                           \
                    paletteData = GBA_TEXT_MAP_PALETTE(mapData) << 4;                              \
                    palette = &mainPalette[paletteData];                                           \
                    LOAD_32(tileData, charBase, vram);                                             \
                    if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                            \
                        tileData >>= x * 4;                                                        \
                    } else {                                                                       \
                        tileData >>= (7 - x) * 4;                                                  \
                    }                                                                              \
                    tileData &= 0xF;                                                               \
                    tileData |= tileData << 4;                                                     \
                    tileData |= tileData << 8;                                                     \
                    tileData |= tileData << 16;                                                    \
                    carryData = tileData;                                                          \
                }                                                                                  \
                mosaicWait = mosaicH;                                                              \
            }                                                                                      \
            --mosaicWait;                                                                          \
            BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 0);                                            \
            ++pixel;                                                                               \
        }                                                                                          \
        x = 0;                                                                                     \
    }

#define DRAW_BACKGROUND_MODE_0_TILES_16(BLEND, OBJWIN)                                             \
    for (; tileX < tileEnd; ++tileX) {                                                             \
        mapData = background->mapCache[(localX >> 3) & 0x3F];                                      \
        localX += 8;                                                                               \
        localY = inY & 0x7;                                                                        \
        if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                         \
            localY = 7 - localY;                                                                   \
        }                                                                                          \
        paletteData = GBA_TEXT_MAP_PALETTE(mapData) << 4;                                          \
        palette = &mainPalette[paletteData];                                                       \
        charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 5)) + (localY << 2);     \
        if (UNLIKELY(charBase >= 0x10000)) {                                                       \
            pixel += 8;                                                                            \
            continue;                                                                              \
        }                                                                                          \
        LOAD_32(tileData, charBase, vram);                                                         \
        if (tileData) {                                                                            \
            if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                    \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 0);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 1);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 2);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 3);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 4);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 5);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 6);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 7);                                        \
            } else {                                                                               \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 7);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 6);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 5);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 4);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 3);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 2);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 1);                                        \
                BACKGROUND_DRAW_PIXEL_16(BLEND, OBJWIN, 0);                                        \
            }                                                                                      \
        }                                                                                          \
        pixel += 8;                                                                                \
    }

#define DRAW_BACKGROUND_MODE_0_TILE_SUFFIX_256(BLEND, OBJWIN)                                      \
    charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 6)) + (localY << 3);         \
    int end2 = end - 4;                                                                            \
    if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                            \
        int shift = inX & 0x3;                                                                     \
        if (LIKELY(charBase < 0x10000)) {                                                          \
            if (end2 > outX) {                                                                     \
                LOAD_32(tileData, charBase, vram);                                                 \
                tileData >>= 8 * shift;                                                            \
                shift = 0;                                                                         \
                for (; outX < end2; ++outX, ++pixel) {                                             \
                    BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                   \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (LIKELY(charBase < 0x10000)) {                                                          \
            LOAD_32(tileData, charBase + 4, vram);                                                 \
            tileData >>= 8 * shift;                                                                \
            for (; outX < end; ++outX, ++pixel) {                                                  \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
            }                                                                                      \
        }                                                                                          \
    } else {                                                                                       \
        int start = outX;                                                                          \
        outX = end - 1;                                                                            \
        pixel = &renderer->row[outX];                                                              \
        if (LIKELY(charBase < 0x10000)) {                                                          \
            if (end2 > start) {                                                                    \
                LOAD_32(tileData, charBase, vram);                                                 \
                for (; outX >= end2; --outX, --pixel) {                                            \
                    BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                   \
                }                                                                                  \
                charBase += 4;                                                                     \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (LIKELY(charBase < 0x10000)) {                                                          \
            LOAD_32(tileData, charBase, vram);                                                     \
            for (; outX >= renderer->start; --outX, --pixel) {                                     \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
            }                                                                                      \
        }                                                                                          \
        outX = end;                                                                                \
        pixel = &renderer->row[outX];                                                              \
    }

#define DRAW_BACKGROUND_MODE_0_TILE_PREFIX_256(BLEND, OBJWIN)                                      \
    charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 6)) + (localY << 3);         \
    if (UNLIKELY(charBase >= 0x10000)) {                                                           \
        return;                                                                                    \
    }                                                                                              \
    int end = mod8 - 4;                                                                            \
    pixel = &renderer->row[outX];                                                                  \
    if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                            \
        if (end > 0) {                                                                             \
            LOAD_32(tileData, charBase, vram);                                                     \
            for (; outX < renderer->end - end; ++outX, ++pixel) {                                  \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
            }                                                                                      \
            charBase += 4;                                                                         \
        }                                                                                          \
                                                                                                   \
        LOAD_32(tileData, charBase, vram);                                                         \
        for (; outX < renderer->end; ++outX, ++pixel) {                                            \
            BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                           \
        }                                                                                          \
    } else {                                                                                       \
        int shift = (8 - mod8) & 0x3;                                                              \
        int start = outX;                                                                          \
        outX = renderer->end - 1;                                                                  \
        pixel = &renderer->row[outX];                                                              \
        if (end > 0) {                                                                             \
            LOAD_32(tileData, charBase, vram);                                                     \
            tileData >>= 8 * shift;                                                                \
            for (; outX >= start + 4; --outX, --pixel) {                                           \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
            }                                                                                      \
            shift = 0;                                                                             \
        }                                                                                          \
                                                                                                   \
        LOAD_32(tileData, charBase + 4, vram);                                                     \
        tileData >>= 8 * shift;                                                                    \
        for (; outX >= start; --outX, --pixel) {                                                   \
            BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                           \
        }                                                                                          \
    }

#define DRAW_BACKGROUND_MODE_0_TILES_256(BLEND, OBJWIN)                                            \
    for (; tileX < tileEnd; ++tileX) {                                                             \
        mapData = background->mapCache[(localX >> 3) & 0x3F];                                      \
        localX += 8;                                                                               \
        localY = inY & 0x7;                                                                        \
        if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                         \
            localY = 7 - localY;                                                                   \
        }                                                                                          \
        charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 6)) + (localY << 3);     \
        if (UNLIKELY(charBase >= 0x10000)) {                                                       \
            pixel += 8;                                                                            \
            continue;                                                                              \
        }                                                                                          \
        if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                        \
            LOAD_32(tileData, charBase, vram);                                                     \
            if (tileData) {                                                                        \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 1);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 2);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 3);                                       \
            }                                                                                      \
            pixel += 4;                                                                            \
            LOAD_32(tileData, charBase + 4, vram);                                                 \
            if (tileData) {                                                                        \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 1);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 2);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 3);                                       \
            }                                                                                      \
            pixel += 4;                                                                            \
        } else {                                                                                   \
            LOAD_32(tileData, charBase + 4, vram);                                                 \
            if (tileData) {                                                                        \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 3);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 2);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 1);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
            }                                                                                      \
            pixel += 4;                                                                            \
            LOAD_32(tileData, charBase, vram);                                                     \
            if (tileData) {                                                                        \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 3);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 2);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 1);                                       \
                BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                       \
            }                                                                                      \
            pixel += 4;                                                                            \
        }                                                                                          \
    }

#define DRAW_BACKGROUND_MODE_0_MOSAIC_256(BLEND, OBJWIN)                                           \
    x = inX & 7;                                                                                   \
    if (mosaicWait) {                                                                              \
        int baseX = x - (mosaicH - mosaicWait);                                                    \
        if (baseX < 0) {                                                                           \
            int disturbX = (16 + baseX) >> 3;                                                      \
            inX -= disturbX << 3;                                                                  \
            localX = tileX * 8 + inX;                                                              \
            BACKGROUND_TEXT_SELECT_CHARACTER;                                                      \
            localY = inY & 0x7;                                                                    \
            if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                     \
                localY = 7 - localY;                                                               \
            }                                                                                      \
            baseX -= disturbX << 3;                                                                \
            inX += disturbX << 3;                                                                  \
        } else {                                                                                   \
            localX = tileX * 8 + inX;                                                              \
            BACKGROUND_TEXT_SELECT_CHARACTER;                                                      \
            localY = inY & 0x7;                                                                    \
            if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                     \
                localY = 7 - localY;                                                               \
            }                                                                                      \
        }                                                                                          \
        charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 6)) + (localY << 3);     \
        if (UNLIKELY(charBase >= 0x10000)) {                                                       \
            carryData = 0;                                                                         \
        } else {                                                                                   \
            LOAD_32(tileData, charBase, vram);                                                     \
            if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                                    \
                if (x >= 4) {                                                                      \
                    LOAD_32(tileData, charBase + 4, vram);                                         \
                    tileData >>= (x - 4) * 8;                                                      \
                } else {                                                                           \
                    LOAD_32(tileData, charBase, vram);                                             \
                    tileData >>= x * 8;                                                            \
                }                                                                                  \
            } else {                                                                               \
                if (x >= 4) {                                                                      \
                    LOAD_32(tileData, charBase, vram);                                             \
                    tileData >>= (7 - x) * 8;                                                      \
                } else {                                                                           \
                    LOAD_32(tileData, charBase + 4, vram);                                         \
                    tileData >>= (3 - x) * 8;                                                      \
                }                                                                                  \
            }                                                                                      \
            tileData &= 0xFF;                                                                      \
            carryData = tileData;                                                                  \
        }                                                                                          \
    }                                                                                              \
    localX = tileX * 8 + inX;                                                                      \
    for (; length; ++tileX) {                                                                      \
        mapData = background->mapCache[(localX >> 3) & 0x3F];                                      \
        localX += 8;                                                                               \
        localY = inY & 0x7;                                                                        \
        if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                         \
            localY = 7 - localY;                                                                   \
        }                                                                                          \
        charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 6)) + (localY << 3);     \
        tileData = carryData;                                                                      \
        for (; x < 8 && length; ++x, --length) {                                                   \
            if (!mosaicWait) {                                                                     \
                if (UNLIKELY(charBase >= 0x10000)) {                                               \
                    carryData = 0;                                                                 \
                } else {                                                                           \
                    if (!GBA_TEXT_MAP_HFLIP(mapData)) {                                            \
                        if (x >= 4) {                                                              \
                            LOAD_32(tileData, charBase + 4, vram);                                 \
                            tileData >>= (x - 4) * 8;                                              \
                        } else {                                                                   \
                            LOAD_32(tileData, charBase, vram);                                     \
                            tileData >>= x * 8;                                                    \
                        }                                                                          \
                    } else {                                                                       \
                        if (x >= 4) {                                                              \
                            LOAD_32(tileData, charBase, vram);                                     \
                            tileData >>= (7 - x) * 8;                                              \
                        } else {                                                                   \
                            LOAD_32(tileData, charBase + 4, vram);                                 \
                            tileData >>= (3 - x) * 8;                                              \
                        }                                                                          \
                    }                                                                              \
                    tileData &= 0xFF;                                                              \
                    carryData = tileData;                                                          \
                }                                                                                  \
                mosaicWait = mosaicH;                                                              \
            }                                                                                      \
            tileData |= tileData << 8;                                                             \
            --mosaicWait;                                                                          \
            BACKGROUND_DRAW_PIXEL_256(BLEND, OBJWIN, 0);                                           \
            ++pixel;                                                                               \
        }                                                                                          \
        x = 0;                                                                                     \
    }

#ifdef PN_HAVE_AVX2
/* DRAW_BACKGROUND_MODE_0_TILES_16/256 (the whole tiles of a text BG segment), 8 pixels per tile
 * row at once. Every pixel gets the value BACKGROUND_DRAW_PIXEL_16/256 + COMPOSITE_* give it: left
 * alone when the palette index is 0 (and, with the OBJ window, when the objwinForceEnable /
 * objwinOnly test fails), else composited with the colour and flags of the OBJ-window palette/flags
 * where the row pixel carries FLAG_OBJWIN. H-flip reverses the pixel order of the tile row; a tile
 * whose character address is >= 0x10000 is skipped as in the scalar loop. Tiles write disjoint
 * pixels. */

/* A BG pixel `color` always carries FLAG_IS_BACKGROUND (>= 0x08000000), so against a pixel that is
 * not IS_WRITABLE (< 0x02000000) every _composite* variant leaves the pixel unchanged: color is not
 * in front, the covered value current & (0x00FFFFFF | FLAG_REBLEND | FLAG_OBJWIN) is current itself,
 * and current has no FLAG_TARGET_1 to blend with. mGBA's IS_WRITABLE test is therefore only an
 * early-out for BG layers, and the kernels below omit it (sprites in pn_postprocess_sprite are not
 * tested for it in mGBA either). */

/* One tile row: 8 pixels at `pixel`, palette indices `idx`, colours from the sub-palettes (with the OBJ
 * window: `ocolor` where the row pixel carries FLAG_OBJWIN, unless sameObjwin says the OBJ-window
 * palette and flags are the normal ones, as in battle). */
PN_AVX2_INLINE void pn_bg_text_pixels_avx2(PnPpu *renderer, uint32_t *pixel, __m256i idx, __m256i color,
                                           __m256i ocolor, int sameObjwin, int objwinForceEnable, int objwinOnly,
                                           const int blend, const int objwin) {
    const __m256i owbit = _mm256_set1_epi32(FLAG_OBJWIN);
    __m256i current = _mm256_loadu_si256((const __m256i *) pixel);
    __m256i keep = _mm256_cmpeq_epi32(idx, _mm256_setzero_si256()); /* palette index 0: transparent */
    if (objwin) {
        __m256i sel = _mm256_cmpeq_epi32(_mm256_and_si256(current, owbit), owbit);
        if (!sameObjwin) {
            color = _mm256_blendv_epi8(color, ocolor, sel);
        }
        if (!objwinForceEnable) {
            /* drawn iff (!(current & FLAG_OBJWIN)) == objwinOnly */
            keep = _mm256_or_si256(keep, objwinOnly ? sel : _mm256_xor_si256(sel, _mm256_set1_epi32(-1)));
            if (_mm256_testc_si256(keep, _mm256_set1_epi32(-1))) {
                return;
            }
        }
    }
    _mm256_storeu_si256((__m256i *) pixel, pn_composite_keep_avx2(renderer, current, color, keep, blend, objwin));
}

PN_AVX2_INLINE void pn_bg_text_tiles_avx2(PnPpu *renderer, const PnBackground *background, int inY, int tileX,
                                          int tileEnd, int localX, uint32_t *pixel, const uint32_t *mainPalette,
                                          const uint32_t *objwinPalette, uint32_t flags, uint32_t objwinFlags,
                                          int objwinForceEnable, int objwinOnly, const int bpp256, const int blend,
                                          const int objwin) {
    const uint8_t *vram = renderer->vram;
    const __m256i vflags = _mm256_set1_epi32((int) flags);
    const __m256i voflags = _mm256_set1_epi32((int) objwinFlags);
    const __m256i zero = _mm256_setzero_si256();
    /* OBJ-window colour == normal colour for every pixel: one lookup is enough */
    const int sameObjwin = objwinPalette == mainPalette && objwinFlags == flags;
    /* Tile t of the run uses mapCache[((localX >> 3) + t) & 0x3F] = maps[t]. The run is at most 31
     * tiles ((length + inX) / 8 - inX / 8 with length <= 240). */
    const uint16_t *maps = &renderer->mapRow[background->index & 3][((unsigned) localX >> 3) & 0x3F];
    const int n = tileEnd - tileX;
    /* per tile of the run: charBases[t] = character address of its row; for 4bpp also
     * paletteOffsets[t] = GBA_TEXT_MAP_PALETTE(mapData) << 4 and flips[t] = H-flip * 32 (byte
     * offset into `shifts`). One array (in the per-call state), so that the loop below needs one
     * base (i386). */
    uint32_t *charBases = &renderer->tileInfo[0], *paletteOffsets = &renderer->tileInfo[32],
             *flips = &renderer->tileInfo[64];
    int t;
    if (n <= 0) {
        return;
    }
    {
        /* Character addresses of the run, 8 tiles per step:
         * charBase + (tile << 5 or 6) + (localY << 2 or 3), localY = inY & 7 (7 - localY = localY ^ 7
         * when V-flipped). */
        const __m256i vcharBase = _mm256_set1_epi32((int) background->charBase);
        const __m256i vlocalY = _mm256_set1_epi32(inY & 0x7);
        const __m256i vflipBit = _mm256_set1_epi32(0x0800);
        for (t = 0; t < n; t += 8) {
            __m256i mapData = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) &maps[t]));
            __m256i localY = _mm256_xor_si256(
                vlocalY, _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(mapData, vflipBit), vflipBit),
                                          _mm256_set1_epi32(7)));
            __m256i tile = _mm256_and_si256(mapData, _mm256_set1_epi32(0x3FF));
            __m256i charBase = bpp256 ? _mm256_add_epi32(_mm256_slli_epi32(tile, 6), _mm256_slli_epi32(localY, 3))
                                      : _mm256_add_epi32(_mm256_slli_epi32(tile, 5), _mm256_slli_epi32(localY, 2));
            _mm256_storeu_si256((__m256i *) &charBases[t], _mm256_add_epi32(vcharBase, charBase));
            if (!bpp256) {
                _mm256_storeu_si256((__m256i *) &paletteOffsets[t], _mm256_slli_epi32(_mm256_srli_epi32(mapData, 12), 4));
                _mm256_storeu_si256((__m256i *) &flips[t],
                                    _mm256_and_si256(_mm256_srli_epi32(mapData, 5), _mm256_set1_epi32(32)));
            }
        }
    }
    if (!bpp256) {
        /* lane j takes nibble j, or 7 - j when H-flipped */
        const __m256i shifts[2] = {_mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28),
                                   _mm256_setr_epi32(28, 24, 20, 16, 12, 8, 4, 0)};
        const __m256i nibble = _mm256_set1_epi32(0xF);
        for (t = 0; t < n; ++t, pixel += 8) {
            uint32_t charBase = charBases[t];
            if (UNLIKELY(charBase >= 0x10000)) {
                continue;
            }
            if (!ld32(vram, charBase)) {
                continue;
            }
            __m256i shift = _mm256_loadu_si256((const __m256i *) ((const uint8_t *) shifts + flips[t]));
            /* the tile row, broadcast straight from VRAM */
            __m256i row = _mm256_castps_si256(_mm256_broadcast_ss((const float *) (const void *) (vram + charBase)));
            __m256i idx = _mm256_and_si256(_mm256_srlv_epi32(row, shift), nibble);
            unsigned paletteData = paletteOffsets[t];
            __m256i color = _mm256_or_si256(pn_lut16_avx2(&mainPalette[paletteData], idx), vflags);
            __m256i ocolor = zero;
            if (objwin && !sameObjwin) {
                ocolor = _mm256_or_si256(pn_lut16_avx2(&objwinPalette[paletteData], idx), voflags);
            }
            pn_bg_text_pixels_avx2(renderer, pixel, idx, color, ocolor, sameObjwin, objwinForceEnable, objwinOnly,
                                   blend, objwin);
        }
        return;
    }
    const __m128i reverse8 = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15);
    for (t = 0; t < n; ++t, pixel += 8) {
        uint32_t charBase = charBases[t];
        if (UNLIKELY(charBase >= 0x10000)) {
            continue;
        }
        __m128i bytes = _mm_loadl_epi64((const __m128i *) (vram + charBase));
        if (_mm_testz_si128(bytes, bytes)) {
            continue;
        }
        if (GBA_TEXT_MAP_HFLIP(maps[t])) {
            bytes = _mm_shuffle_epi8(bytes, reverse8);
        }
        __m256i idx = _mm256_cvtepu8_epi32(bytes);
        __m256i color = _mm256_or_si256(_mm256_i32gather_epi32((const int *) mainPalette, idx, 4), vflags);
        __m256i ocolor = zero;
        if (objwin && !sameObjwin) {
            ocolor = _mm256_or_si256(_mm256_i32gather_epi32((const int *) objwinPalette, idx, 4), voflags);
        }
        pn_bg_text_pixels_avx2(renderer, pixel, idx, color, ocolor, sameObjwin, objwinForceEnable, objwinOnly, blend,
                               objwin);
    }
}

/* A partial 4bpp tile row (the first or last tile of a segment, DRAW_BACKGROUND_MODE_0_TILE_SUFFIX_16 /
 * _PREFIX_16): lane j is pixel x0 + j and takes nibble nibbleIndex[j] of tileData (an index outside
 * 0..7 reads as 0, like a nibble shifted out of the scalar tileData); only `active` lanes (all inside
 * the row) change. */
PN_AVX2_INLINE void pn_bg_text_partial_avx2(PnPpu *renderer, int x0, __m256i nibbleIndex, __m256i active,
                                            uint32_t tileData, const uint32_t *palette, const uint32_t *objwinPalette,
                                            uint32_t flags, uint32_t objwinFlags, int objwinForceEnable,
                                            int objwinOnly, const int blend, const int objwin) {
    const __m256i owbit = _mm256_set1_epi32(FLAG_OBJWIN);
    const __m256i ones = _mm256_set1_epi32(-1);
    /* vpsrlvd gives 0 for shift counts >= 32 (and for negative indices, as unsigned) */
    __m256i idx = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) tileData), _mm256_slli_epi32(nibbleIndex, 2)),
                                   _mm256_set1_epi32(0xF));
    __m256i color = _mm256_or_si256(pn_lut16_avx2(palette, idx), _mm256_set1_epi32((int) flags));
    uint32_t *pixel = &renderer->row[x0];
    /* inside the row, inactive lanes are stored back unchanged (masked moves are slow on some CPUs) */
    const int masked = x0 + 8 > GBA_VIDEO_HORIZONTAL_PIXELS;
    __m256i current = masked ? _mm256_maskload_epi32((const int *) pixel, active) : _mm256_loadu_si256((const __m256i *) pixel);
    __m256i keep = _mm256_or_si256(_mm256_cmpeq_epi32(idx, _mm256_setzero_si256()), _mm256_xor_si256(active, ones));
    if (objwin) {
        __m256i sel = _mm256_cmpeq_epi32(_mm256_and_si256(current, owbit), owbit);
        if (objwinPalette != palette || objwinFlags != flags) {
            __m256i ocolor = _mm256_or_si256(pn_lut16_avx2(objwinPalette, idx), _mm256_set1_epi32((int) objwinFlags));
            color = _mm256_blendv_epi8(color, ocolor, sel);
        }
        if (!objwinForceEnable) {
            keep = _mm256_or_si256(keep, objwinOnly ? sel : _mm256_xor_si256(sel, ones));
        }
    }
    if (_mm256_testc_si256(keep, ones)) {
        return;
    }
    __m256i out = pn_composite_keep_avx2(renderer, current, color, keep, blend, objwin);
    if (masked) {
        _mm256_maskstore_epi32((int *) pixel, _mm256_xor_si256(keep, ones), out);
    } else {
        _mm256_storeu_si256((__m256i *) pixel, out);
    }
}

/* DRAW_BACKGROUND_MODE_0_TILE_SUFFIX_16 (the segment starts inside a tile, at outX == start):
 * pixel outX + j takes nibble mod8 + j, or nibble (end - 1 - outX) - j when H-flipped (the scalar
 * flipped loop runs from end - 1 down to start), j < end - outX. The character address is not
 * range-checked (mGBA reads OBJ VRAM there; it stays below 0x14000). */
PN_AVX2_INLINE void pn_bg_text_suffix_avx2(PnPpu *renderer, const PnBackground *background, uint16_t mapData,
                                           int localY, int mod8, int outX, int end, const uint32_t *mainPalette,
                                           const uint32_t *objwinPalette, uint32_t flags, uint32_t objwinFlags,
                                           int objwinForceEnable, int objwinOnly, const int blend, const int objwin) {
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    uint32_t charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 5)) + (localY << 2);
    unsigned paletteData = GBA_TEXT_MAP_PALETTE(mapData) << 4;
    int count = end - outX;
    __m256i nibbleIndex = GBA_TEXT_MAP_HFLIP(mapData) ? _mm256_sub_epi32(_mm256_set1_epi32(count - 1), lanes)
                                                      : _mm256_add_epi32(_mm256_set1_epi32(mod8), lanes);
    pn_bg_text_partial_avx2(renderer, outX, nibbleIndex, _mm256_cmpgt_epi32(_mm256_set1_epi32(count), lanes),
                            ld32(renderer->vram, charBase), &mainPalette[paletteData], &objwinPalette[paletteData],
                            flags, objwinFlags, objwinForceEnable, objwinOnly, blend, objwin);
}

/* DRAW_BACKGROUND_MODE_0_TILE_PREFIX_16 (the segment ends inside a tile; mod8 = length & 7). Not
 * flipped: pixel outX + j takes nibble j of tileData >> (4 * (start - outX)) if outX < start (shift
 * count masked to 5 bits as on x86; outX then starts at start), j < end - outX (nibbles past 7 are
 * 0). Flipped: the scalar loop visits end - 1 down to end - 7 (not below 0), pixel end - 1 - k
 * taking nibble 8 - mod8 + k: lane j = pixel end - 8 + j, nibble 15 - mod8 - j, active for
 * j >= 8 - mod8. Returns 0 when the case is left to the scalar code (flipped, end < 8). */
PN_AVX2_INLINE int pn_bg_text_prefix_avx2(PnPpu *renderer, const PnBackground *background, uint16_t mapData,
                                          int localY, int mod8, int outX, const uint32_t *mainPalette,
                                          const uint32_t *objwinPalette, uint32_t flags, uint32_t objwinFlags,
                                          int objwinForceEnable, int objwinOnly, const int blend, const int objwin) {
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    uint32_t charBase = (background->charBase + (GBA_TEXT_MAP_TILE(mapData) << 5)) + (localY << 2);
    unsigned paletteData = GBA_TEXT_MAP_PALETTE(mapData) << 4;
    uint32_t tileData = ld32(renderer->vram, charBase);
    __m256i nibbleIndex, active;
    int x0;
    if (!GBA_TEXT_MAP_HFLIP(mapData)) {
        int skip = 0;
        x0 = outX;
        if (x0 < renderer->start) {
            skip = ((4 * (renderer->start - x0)) & 31) >> 2;
            x0 = renderer->start;
        }
        if (x0 >= renderer->end) {
            return 1;
        }
        nibbleIndex = _mm256_add_epi32(_mm256_set1_epi32(skip), lanes);
        active = _mm256_cmpgt_epi32(_mm256_set1_epi32(renderer->end - x0), lanes);
    } else {
        if (renderer->end < 8) {
            return 0;
        }
        x0 = renderer->end - 8;
        nibbleIndex = _mm256_sub_epi32(_mm256_set1_epi32(15 - mod8), lanes);
        active = _mm256_cmpgt_epi32(lanes, _mm256_set1_epi32(7 - mod8));
    }
    pn_bg_text_partial_avx2(renderer, x0, nibbleIndex, active, tileData, &mainPalette[paletteData],
                            &objwinPalette[paletteData], flags, objwinFlags, objwinForceEnable, objwinOnly, blend,
                            objwin);
    return 1;
}

#define PN_BG_TEXT_PARTIAL_AVX2_FNS(BLEND, OBJWIN, B, O)                                                          \
    PN_AVX2_FN void pn_bg_suffix_avx2_##BLEND##_##OBJWIN(                                                          \
        PnPpu *renderer, const PnBackground *background, uint16_t mapData, int localY, int mod8, int outX, int end, \
        const uint32_t *mainPalette, const uint32_t *objwinPalette, uint32_t flags, uint32_t objwinFlags,         \
        int objwinForceEnable, int objwinOnly) {                                                                 \
        pn_bg_text_suffix_avx2(renderer, background, mapData, localY, mod8, outX, end, mainPalette, objwinPalette, \
                               flags, objwinFlags, objwinForceEnable, objwinOnly, B, O);                         \
    }                                                                                                            \
    PN_AVX2_FN int pn_bg_prefix_avx2_##BLEND##_##OBJWIN(                                                           \
        PnPpu *renderer, const PnBackground *background, uint16_t mapData, int localY, int mod8, int outX,        \
        const uint32_t *mainPalette, const uint32_t *objwinPalette, uint32_t flags, uint32_t objwinFlags,         \
        int objwinForceEnable, int objwinOnly) {                                                                 \
        return pn_bg_text_prefix_avx2(renderer, background, mapData, localY, mod8, outX, mainPalette,             \
                                      objwinPalette, flags, objwinFlags, objwinForceEnable, objwinOnly, B, O);   \
    }
PN_BG_TEXT_PARTIAL_AVX2_FNS(NoBlend, NO_OBJWIN, 0, 0)
PN_BG_TEXT_PARTIAL_AVX2_FNS(Blend, NO_OBJWIN, 1, 0)
PN_BG_TEXT_PARTIAL_AVX2_FNS(NoBlend, OBJWIN, 0, 1)
PN_BG_TEXT_PARTIAL_AVX2_FNS(Blend, OBJWIN, 1, 1)

#define PN_BG_TEXT_TILES_AVX2_FN(NAME, BPP256, BLEND, OBJWIN)                                                     \
    PN_AVX2_FN void NAME(PnPpu *renderer, const PnBackground *background, int inY, int tileX, int tileEnd,       \
                         int localX, uint32_t *pixel, const uint32_t *mainPalette, const uint32_t *objwinPalette, \
                         uint32_t flags, uint32_t objwinFlags, int objwinForceEnable, int objwinOnly) {          \
        pn_bg_text_tiles_avx2(renderer, background, inY, tileX, tileEnd, localX, pixel, mainPalette,             \
                              objwinPalette, flags, objwinFlags, objwinForceEnable, objwinOnly, BPP256, BLEND,  \
                              OBJWIN);                                                                          \
    }
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_16_NoBlend_NO_OBJWIN, 0, 0, 0)
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_16_Blend_NO_OBJWIN, 0, 1, 0)
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_16_NoBlend_OBJWIN, 0, 0, 1)
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_16_Blend_OBJWIN, 0, 1, 1)
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_256_NoBlend_NO_OBJWIN, 1, 0, 0)
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_256_Blend_NO_OBJWIN, 1, 1, 0)
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_256_NoBlend_OBJWIN, 1, 0, 1)
PN_BG_TEXT_TILES_AVX2_FN(pn_bg_tiles_avx2_256_Blend_OBJWIN, 1, 1, 1)

/* The whole-tile part of DRAW_BACKGROUND_MODE_0: the AVX2 kernel when selected, else mGBA's loop.
 * Either way tileX ends at max(tileX, tileEnd) (what the code after the loop reads). A run has at
 * most 31 tiles (window edges are <= 240); longer runs would take mGBA's loop. */
#define DRAW_BACKGROUND_MODE_0_TILES(BPP, BLEND, OBJWIN)                                                \
    if (renderer->simd == PN_PPU_SIMD_AVX2 && tileEnd - tileX <= 32) {                                  \
        pn_bg_tiles_avx2_##BPP##_##BLEND##_##OBJWIN(renderer, background, inY, tileX, tileEnd, localX,   \
                                                    pixel, mainPalette, objwinPalette, flags, objwinFlags, \
                                                    objwinForceEnable, objwinOnly);                        \
        if (tileX < tileEnd) {                                                                          \
            tileX = tileEnd;                                                                            \
        }                                                                                               \
    } else {                                                                                            \
        DRAW_BACKGROUND_MODE_0_TILES_##BPP(BLEND, OBJWIN)                                               \
    }
/* The partial tiles at the ends of a segment: AVX2 for 4bpp, mGBA's loops otherwise. */
#define DRAW_BACKGROUND_MODE_0_SUFFIX_16_FAST(BLEND, OBJWIN)                                                   \
    if (renderer->simd == PN_PPU_SIMD_AVX2) {                                                                  \
        pn_bg_suffix_avx2_##BLEND##_##OBJWIN(renderer, background, mapData, localY, mod8, outX, end, mainPalette, \
                                             objwinPalette, flags, objwinFlags, objwinForceEnable, objwinOnly);  \
    } else {                                                                                                   \
        DRAW_BACKGROUND_MODE_0_TILE_SUFFIX_16(BLEND, OBJWIN)                                                   \
    }
#define DRAW_BACKGROUND_MODE_0_PREFIX_16_FAST(BLEND, OBJWIN)                                                   \
    if (renderer->simd != PN_PPU_SIMD_AVX2 ||                                                                  \
        !pn_bg_prefix_avx2_##BLEND##_##OBJWIN(renderer, background, mapData, localY, mod8, outX, mainPalette,   \
                                              objwinPalette, flags, objwinFlags, objwinForceEnable, objwinOnly)) { \
        DRAW_BACKGROUND_MODE_0_TILE_PREFIX_16(BLEND, OBJWIN)                                                   \
    }
#else
#define DRAW_BACKGROUND_MODE_0_TILES(BPP, BLEND, OBJWIN) DRAW_BACKGROUND_MODE_0_TILES_##BPP(BLEND, OBJWIN)
#define DRAW_BACKGROUND_MODE_0_SUFFIX_16_FAST(BLEND, OBJWIN) DRAW_BACKGROUND_MODE_0_TILE_SUFFIX_16(BLEND, OBJWIN)
#define DRAW_BACKGROUND_MODE_0_PREFIX_16_FAST(BLEND, OBJWIN) DRAW_BACKGROUND_MODE_0_TILE_PREFIX_16(BLEND, OBJWIN)
#endif
#define DRAW_BACKGROUND_MODE_0_SUFFIX_256_FAST(BLEND, OBJWIN) DRAW_BACKGROUND_MODE_0_TILE_SUFFIX_256(BLEND, OBJWIN)
#define DRAW_BACKGROUND_MODE_0_PREFIX_256_FAST(BLEND, OBJWIN) DRAW_BACKGROUND_MODE_0_TILE_PREFIX_256(BLEND, OBJWIN)

#define DRAW_BACKGROUND_MODE_0(BPP, BLEND, OBJWIN)                                                 \
    uint32_t *pixel = &renderer->row[outX];                                                        \
    if (background->mosaic && GBAMosaicControlGetBgH(renderer->mosaic)) {                          \
        int mosaicH = GBAMosaicControlGetBgH(renderer->mosaic) + 1;                                \
        int x;                                                                                     \
        int mosaicWait = (mosaicH - outX + GBA_VIDEO_HORIZONTAL_PIXELS * mosaicH) % mosaicH;       \
        int carryData = 0;                                                                         \
        paletteData = 0; /* Quiets compiler warning */                                             \
        DRAW_BACKGROUND_MODE_0_MOSAIC_##BPP(BLEND, OBJWIN) return;                                 \
    }                                                                                              \
                                                                                                   \
    if (inX & 0x7) {                                                                               \
        localX = tileX * 8 + inX;                                                                  \
        BACKGROUND_TEXT_SELECT_CHARACTER;                                                          \
        localY = inY & 0x7;                                                                        \
        if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                         \
            localY = 7 - localY;                                                                   \
        }                                                                                          \
        int mod8 = inX & 0x7;                                                                      \
        int end = outX + 0x8 - mod8;                                                               \
        if (end > renderer->end) {                                                                 \
            end = renderer->end;                                                                   \
        }                                                                                          \
        if (UNLIKELY(end == outX)) {                                                               \
            return;                                                                                \
        }                                                                                          \
        if (UNLIKELY(end < outX)) {                                                                \
            return;                                                                                \
        }                                                                                          \
        DRAW_BACKGROUND_MODE_0_SUFFIX_##BPP##_FAST(BLEND, OBJWIN) outX = end;                      \
        if (tileX < tileEnd) {                                                                     \
            ++tileX;                                                                               \
        }                                                                                          \
        length -= end - renderer->start;                                                           \
    }                                                                                              \
    pixel = &renderer->row[outX];                                                                  \
    outX += (tileEnd - tileX) * 8;                                                                 \
    localX = (tileX * 8 + inX) & 0x1FF;                                                            \
    DRAW_BACKGROUND_MODE_0_TILES(BPP, BLEND, OBJWIN) if (length & 0x7) {                           \
        localX = tileX * 8 + inX;                                                                  \
        BACKGROUND_TEXT_SELECT_CHARACTER;                                                          \
        localY = inY & 0x7;                                                                        \
        if (GBA_TEXT_MAP_VFLIP(mapData)) {                                                         \
            localY = 7 - localY;                                                                   \
        }                                                                                          \
        int mod8 = length & 0x7;                                                                   \
        DRAW_BACKGROUND_MODE_0_PREFIX_##BPP##_FAST(BLEND, OBJWIN)                                  \
    }

static void pn_draw_background_mode0(PnPpu *renderer, PnBackground *background, int y) {
    int inX = (renderer->start + background->x) & 0x1FF;
    int length = renderer->end - renderer->start;
    if (background->mosaic) {
        int mosaicV = GBAMosaicControlGetBgV(renderer->mosaic) + 1;
        if (mosaicV > 1) { /* y % 1 == 0: skip the division (overworld BGs have the bit set, size 1) */
            y -= y % mosaicV;
        }
    }
    int inY = y + background->y;
    uint16_t mapData;

    unsigned yBase = inY & 0xF8;
    if (background->size == 2) {
        yBase += inY & 0x100;
    } else if (background->size == 3) {
        yBase += (inY & 0x100) << 1;
    }
    yBase = (background->screenBase >> 1) + (yBase << 2);

    int localX;
    int localY;

    unsigned xBase;

    uint32_t flags = background->flags;
    uint32_t objwinFlags = background->objwinFlags;
    bool variant = background->variant;

    uint32_t screenBase;
    uint32_t charBase;
    const uint32_t *mainPalette = renderer->normalPalette;
    if (variant) {
        mainPalette = renderer->variantPalette;
    }
    const uint32_t *palette = mainPalette;
    PREPARE_OBJWIN;

    int outX = renderer->start;

    uint32_t tileData;
    uint32_t current;
    int pixelData;
    int paletteData;
    int tileX;
    int tileEnd = ((length + inX) >> 3) - (inX >> 3);
    const uint8_t *vram = renderer->vram;

    if (background->yCache != inY >> 3) {
        /* mGBA: for tileX in 0..63 { localX = tileX * 8; BACKGROUND_TEXT_SELECT_CHARACTER; mapCache[tileX] =
         * mapData; }, i.e. the 32 map entries at yBase, then the 32 at yBase + 0x400 (the right
         * screen block) for 512-wide maps and again those at yBase for 256-wide ones. yBase <= 0x87E0,
         * so both runs lie inside VRAM. */
        memcpy(&background->mapCache[0], vram + (yBase << 1), 32 * sizeof(uint16_t));
        memcpy(&background->mapCache[32], vram + ((yBase + ((background->size & 1) ? 0x400u : 0u)) << 1),
               32 * sizeof(uint16_t));
        memcpy(&renderer->mapRow[background->index & 3][0], background->mapCache, sizeof(background->mapCache));
        memcpy(&renderer->mapRow[background->index & 3][64], background->mapCache, sizeof(background->mapCache));
        background->yCache = inY >> 3;
    }

    tileX = 0;
    if (!objwinSlowPath) {
        if (!(flags & FLAG_TARGET_2)) {
            if (!background->multipalette) {
                DRAW_BACKGROUND_MODE_0(16, NoBlend, NO_OBJWIN);
            } else {
                DRAW_BACKGROUND_MODE_0(256, NoBlend, NO_OBJWIN);
            }
        } else {
            if (!background->multipalette) {
                DRAW_BACKGROUND_MODE_0(16, Blend, NO_OBJWIN);
            } else {
                DRAW_BACKGROUND_MODE_0(256, Blend, NO_OBJWIN);
            }
        }
    } else {
        if (!(flags & FLAG_TARGET_2)) {
            if (!background->multipalette) {
                DRAW_BACKGROUND_MODE_0(16, NoBlend, OBJWIN);
            } else {
                DRAW_BACKGROUND_MODE_0(256, NoBlend, OBJWIN);
            }
        } else {
            if (!background->multipalette) {
                DRAW_BACKGROUND_MODE_0(16, Blend, OBJWIN);
            } else {
                DRAW_BACKGROUND_MODE_0(256, Blend, OBJWIN);
            }
        }
    }
}

/* ================================================================== BG modes 1-5 (software-bg.c) */

#define BACKGROUND_BITMAP_ITERATE(W, H)                                                     \
    x += background->dx;                                                                    \
    y += background->dy;                                                                    \
    if ((x < 0 || y < 0 || (x >> 8) >= W || (y >> 8) >= H) && !mosaicWait) {               \
        continue;                                                                           \
    }                                                                                       \
    localX = x;                                                                             \
    localY = y;

#define MODE_2_COORD_OVERFLOW          \
    localX = x & (sizeAdjusted - 1);   \
    localY = y & (sizeAdjusted - 1);

#define MODE_2_COORD_NO_OVERFLOW              \
    if ((x | y) & ~(sizeAdjusted - 1)) {      \
        continue;                             \
    }                                         \
    localX = x;                               \
    localY = y;

#define MODE_2_NO_MOSAIC(COORD)                                                                   \
    COORD                                                                                         \
    mapData = screenBase[(localX >> 11) + (((localY >> 7) & 0x7F0) << background->size)];        \
    pixelData = charBase[(mapData << 6) + ((localY & 0x700) >> 5) + ((localX & 0x700) >> 8)];

#define MODE_2_MOSAIC(COORD)            \
    if (!mosaicWait) {                  \
        MODE_2_NO_MOSAIC(COORD)         \
        mosaicWait = mosaicH;           \
    } else {                            \
        --mosaicWait;                   \
    }

#define MODE_2_LOOP(MOSAIC, COORD, BLEND, OBJWIN)                                                  \
    for (pixel = &renderer->row[outX]; outX < renderer->end; ++outX, ++pixel) {                    \
        x += background->dx;                                                                       \
        y += background->dy;                                                                       \
                                                                                                   \
        uint32_t current = *pixel;                                                                 \
        MOSAIC(COORD)                                                                              \
        if (pixelData && IS_WRITABLE(current)) {                                                   \
            COMPOSITE_256_##OBJWIN(BLEND, 0);                                                      \
        }                                                                                          \
    }

#ifdef PN_HAVE_AVX2
/* The leading whole groups of 8 pixels of MODE_2_LOOP(MODE_2_NO_MOSAIC, ...) (sets outX, x, y). */
#define MODE_2_FAST(OVERFLOW, BLEND, OBJWIN)                                                             \
    if (renderer->simd == PN_PPU_SIMD_AVX2) {                                                            \
        outX = pn_bg_affine_avx2_##OVERFLOW##_##BLEND##_##OBJWIN(renderer, background, &x, &y, screenBase,   \
                                                                charBase, palette, objwinPalette, flags,  \
                                                                objwinFlags, objwinForceEnable, objwinOnly); \
    }
#else
#define MODE_2_FAST(OVERFLOW, BLEND, OBJWIN)
#endif

#define DRAW_BACKGROUND_MODE_2(BLEND, OBJWIN)                                                      \
    outX = renderer->start;                                                                        \
    if (background->overflow) {                                                                    \
        if (mosaicH > 1) {                                                                         \
            localX &= sizeAdjusted - 1;                                                            \
            localY &= sizeAdjusted - 1;                                                            \
            MODE_2_NO_MOSAIC();                                                                    \
            MODE_2_LOOP(MODE_2_MOSAIC, MODE_2_COORD_OVERFLOW, BLEND, OBJWIN);                      \
        } else {                                                                                   \
            MODE_2_FAST(1, BLEND, OBJWIN)                                                          \
            MODE_2_LOOP(MODE_2_NO_MOSAIC, MODE_2_COORD_OVERFLOW, BLEND, OBJWIN);                   \
        }                                                                                          \
    } else {                                                                                       \
        if (mosaicH > 1) {                                                                         \
            if (!((x | y) & ~(sizeAdjusted - 1))) {                                                \
                localX &= sizeAdjusted - 1;                                                        \
                localY &= sizeAdjusted - 1;                                                        \
                MODE_2_NO_MOSAIC();                                                                \
            }                                                                                      \
            MODE_2_LOOP(MODE_2_MOSAIC, MODE_2_COORD_NO_OVERFLOW, BLEND, OBJWIN);                   \
        } else {                                                                                   \
            MODE_2_FAST(0, BLEND, OBJWIN)                                                          \
            MODE_2_LOOP(MODE_2_NO_MOSAIC, MODE_2_COORD_NO_OVERFLOW, BLEND, OBJWIN);                \
        }                                                                                          \
    }

#ifdef PN_HAVE_AVX2
/* MODE_2_LOOP with MODE_2_NO_MOSAIC for whole groups of 8 pixels from renderer->start: lane j uses
 * x + (j + 1) * dx, y + (j + 1) * dy (32-bit wrap-around like the scalar -fwrapv sums). Out-of-range
 * lanes (no overflow/wrap bit) are skipped like the scalar `continue`; their coordinates are zeroed
 * so that the gathers stay inside VRAM. Map and character bytes are read as 32-bit words at the
 * byte offset (at most 0x13804 + 4 and 0x10003, inside the 96 KiB). Returns the next outX. */
PN_AVX2_INLINE int pn_bg_affine_avx2(PnPpu *renderer, const PnBackground *background, int32_t *xp, int32_t *yp,
                                     const uint8_t *screenBase, const uint8_t *charBase, const uint32_t *palette,
                                     const uint32_t *objwinPalette, uint32_t flags, uint32_t objwinFlags,
                                     int objwinForceEnable, int objwinOnly, const int overflow, const int blend,
                                     const int objwin) {
    const int32_t sizeMask = (0x8000 << background->size) - 1;
    const __m256i vsizeMask = _mm256_set1_epi32(sizeMask);
    const __m256i lanes = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i stepX = _mm256_mullo_epi32(lanes, _mm256_set1_epi32(background->dx));
    const __m256i stepY = _mm256_mullo_epi32(lanes, _mm256_set1_epi32(background->dy));
    const __m128i sizeShift = _mm_cvtsi32_si128(background->size);
    const __m256i byteMask = _mm256_set1_epi32(0xFF);
    const __m256i vflags = _mm256_set1_epi32((int) flags);
    const __m256i voflags = _mm256_set1_epi32((int) objwinFlags);
    const __m256i owbit = _mm256_set1_epi32(FLAG_OBJWIN);
    const __m256i zero = _mm256_setzero_si256();
    int32_t x = *xp, y = *yp;
    int outX = renderer->start;
    uint32_t *pixel = &renderer->row[outX];
    for (; outX + 8 <= renderer->end; outX += 8, pixel += 8) {
        __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), stepX);
        __m256i yv = _mm256_add_epi32(_mm256_set1_epi32(y), stepY);
        x += 8 * background->dx;
        y += 8 * background->dy;
        __m256i valid, lx, ly;
        if (overflow) {
            valid = _mm256_cmpeq_epi32(zero, zero);
            lx = _mm256_and_si256(xv, vsizeMask);
            ly = _mm256_and_si256(yv, vsizeMask);
        } else {
            valid = _mm256_cmpeq_epi32(_mm256_andnot_si256(vsizeMask, _mm256_or_si256(xv, yv)), zero);
            if (_mm256_testz_si256(valid, valid)) {
                continue;
            }
            lx = _mm256_and_si256(xv, valid);
            ly = _mm256_and_si256(yv, valid);
        }
        __m256i mapOffset = _mm256_add_epi32(
            _mm256_srli_epi32(lx, 11),
            _mm256_sll_epi32(_mm256_and_si256(_mm256_srli_epi32(ly, 7), _mm256_set1_epi32(0x7F0)), sizeShift));
        __m256i mapData = _mm256_and_si256(_mm256_i32gather_epi32((const int *) screenBase, mapOffset, 1), byteMask);
        __m256i charOffset = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_slli_epi32(mapData, 6),
                             _mm256_srli_epi32(_mm256_and_si256(ly, _mm256_set1_epi32(0x700)), 5)),
            _mm256_srli_epi32(_mm256_and_si256(lx, _mm256_set1_epi32(0x700)), 8));
        __m256i pixelData = _mm256_and_si256(_mm256_i32gather_epi32((const int *) charBase, charOffset, 1), byteMask);
        __m256i current = _mm256_loadu_si256((const __m256i *) pixel);
        /* IS_WRITABLE omitted: see pn_bg_text_pixels_avx2 */
        __m256i m = _mm256_andnot_si256(_mm256_cmpeq_epi32(pixelData, zero), valid);
        if (_mm256_testz_si256(m, m)) {
            continue;
        }
        __m256i color = _mm256_or_si256(_mm256_i32gather_epi32((const int *) palette, pixelData, 4), vflags);
        if (objwin) {
            __m256i sel = _mm256_cmpeq_epi32(_mm256_and_si256(current, owbit), owbit);
            if (objwinPalette != palette || objwinFlags != flags) {
                __m256i ocolor = _mm256_or_si256(_mm256_i32gather_epi32((const int *) objwinPalette, pixelData, 4), voflags);
                color = _mm256_blendv_epi8(color, ocolor, sel);
            }
            if (!objwinForceEnable) {
                m = objwinOnly ? _mm256_andnot_si256(sel, m) : _mm256_and_si256(m, sel);
            }
        }
        _mm256_storeu_si256((__m256i *) pixel, pn_composite_avx2(renderer, current, color, m, blend, objwin));
    }
    *xp = x;
    *yp = y;
    return outX;
}

#define PN_BG_AFFINE_AVX2_FN(OVERFLOW, BLEND, OBJWIN, B, O)                                                        \
    PN_AVX2_FN int pn_bg_affine_avx2_##OVERFLOW##_##BLEND##_##OBJWIN(                                             \
        PnPpu *renderer, const PnBackground *background, int32_t *xp, int32_t *yp, const uint8_t *screenBase,     \
        const uint8_t *charBase, const uint32_t *palette, const uint32_t *objwinPalette, uint32_t flags,         \
        uint32_t objwinFlags, int objwinForceEnable, int objwinOnly) {                                          \
        return pn_bg_affine_avx2(renderer, background, xp, yp, screenBase, charBase, palette, objwinPalette,    \
                                 flags, objwinFlags, objwinForceEnable, objwinOnly, OVERFLOW, B, O);            \
    }
PN_BG_AFFINE_AVX2_FN(0, NoBlend, NO_OBJWIN, 0, 0)
PN_BG_AFFINE_AVX2_FN(0, Blend, NO_OBJWIN, 1, 0)
PN_BG_AFFINE_AVX2_FN(0, NoBlend, OBJWIN, 0, 1)
PN_BG_AFFINE_AVX2_FN(0, Blend, OBJWIN, 1, 1)
PN_BG_AFFINE_AVX2_FN(1, NoBlend, NO_OBJWIN, 0, 0)
PN_BG_AFFINE_AVX2_FN(1, Blend, NO_OBJWIN, 1, 0)
PN_BG_AFFINE_AVX2_FN(1, NoBlend, OBJWIN, 0, 1)
PN_BG_AFFINE_AVX2_FN(1, Blend, OBJWIN, 1, 1)
#endif

static void pn_draw_background_mode2(PnPpu *renderer, PnBackground *background, int inY) {
    int sizeAdjusted = 0x8000 << background->size;

    BACKGROUND_BITMAP_INIT;

    const uint8_t *screenBase = &renderer->vram[background->screenBase];
    const uint8_t *charBase = &renderer->vram[background->charBase];
    uint8_t mapData;
    uint8_t pixelData = 0;

    int outX;
    uint32_t *pixel;

    if (!objwinSlowPath) {
        if (!(flags & FLAG_TARGET_2)) {
            DRAW_BACKGROUND_MODE_2(NoBlend, NO_OBJWIN);
        } else {
            DRAW_BACKGROUND_MODE_2(Blend, NO_OBJWIN);
        }
    } else {
        if (!(flags & FLAG_TARGET_2)) {
            DRAW_BACKGROUND_MODE_2(NoBlend, OBJWIN);
        } else {
            DRAW_BACKGROUND_MODE_2(Blend, OBJWIN);
        }
    }
}

static void pn_draw_background_mode3(PnPpu *renderer, PnBackground *background, int inY) {
    BACKGROUND_BITMAP_INIT;

    uint32_t color = renderer->normalPalette[0];
    if (mosaicWait && localX >= 0 && localY >= 0 && (localX >> 8) < GBA_VIDEO_HORIZONTAL_PIXELS &&
        (localY >> 8) < GBA_VIDEO_VERTICAL_PIXELS) {
        LOAD_16(color, ((localX >> 8) + (localY >> 8) * GBA_VIDEO_HORIZONTAL_PIXELS) << 1, renderer->vram);
        color = mColorFrom555(color);
    }

    int outX;
    uint32_t *pixel;
    for (outX = renderer->start, pixel = &renderer->row[outX]; outX < renderer->end; ++outX, ++pixel) {
        BACKGROUND_BITMAP_ITERATE(GBA_VIDEO_HORIZONTAL_PIXELS, GBA_VIDEO_VERTICAL_PIXELS);

        if (!mosaicWait) {
            LOAD_16(color, ((localX >> 8) + (localY >> 8) * GBA_VIDEO_HORIZONTAL_PIXELS) << 1, renderer->vram);
            color = mColorFrom555(color);
            mosaicWait = mosaicH;
        } else {
            --mosaicWait;
        }

        uint32_t current = *pixel;
        if (!objwinSlowPath || (!(current & FLAG_OBJWIN)) != objwinOnly) {
            unsigned mergedFlags = flags;
            if (current & FLAG_OBJWIN) {
                mergedFlags = objwinFlags;
            }
            if (!variant) {
                _compositeBlendObjwin(renderer, pixel, color | mergedFlags, current);
            } else if (renderer->blendEffect == BLEND_BRIGHTEN) {
                _compositeBlendObjwin(renderer, pixel, _brighten(color, renderer->bldy) | mergedFlags, current);
            } else if (renderer->blendEffect == BLEND_DARKEN) {
                _compositeBlendObjwin(renderer, pixel, _darken(color, renderer->bldy) | mergedFlags, current);
            }
        }
    }
}

static void pn_draw_background_mode4(PnPpu *renderer, PnBackground *background, int inY) {
    BACKGROUND_BITMAP_INIT;

    uint16_t color = 0;
    uint32_t offset = 0;
    if (GBARegisterDISPCNTIsFrameSelect(renderer->dispcnt)) {
        offset = 0xA000;
    }
    if (mosaicWait && localX >= 0 && localY >= 0 && (localX >> 8) < GBA_VIDEO_HORIZONTAL_PIXELS &&
        (localY >> 8) < GBA_VIDEO_VERTICAL_PIXELS) {
        color = renderer->vram[offset + (localX >> 8) + (localY >> 8) * GBA_VIDEO_HORIZONTAL_PIXELS];
    }

    int outX;
    uint32_t *pixel;
    for (outX = renderer->start, pixel = &renderer->row[outX]; outX < renderer->end; ++outX, ++pixel) {
        BACKGROUND_BITMAP_ITERATE(GBA_VIDEO_HORIZONTAL_PIXELS, GBA_VIDEO_VERTICAL_PIXELS);

        if (!mosaicWait) {
            color = renderer->vram[offset + (localX >> 8) + (localY >> 8) * GBA_VIDEO_HORIZONTAL_PIXELS];

            mosaicWait = mosaicH;
        } else {
            --mosaicWait;
        }

        uint32_t current = *pixel;
        if (color && IS_WRITABLE(current)) {
            if (!objwinSlowPath) {
                _compositeBlendNoObjwin(renderer, pixel, palette[color] | flags, current);
            } else if (objwinForceEnable || (!(current & FLAG_OBJWIN)) == objwinOnly) {
                const uint32_t *currentPalette = (current & FLAG_OBJWIN) ? objwinPalette : palette;
                unsigned mergedFlags = flags;
                if (current & FLAG_OBJWIN) {
                    mergedFlags = objwinFlags;
                }
                _compositeBlendObjwin(renderer, pixel, currentPalette[color] | mergedFlags, current);
            }
        }
    }
}

static void pn_draw_background_mode5(PnPpu *renderer, PnBackground *background, int inY) {
    BACKGROUND_BITMAP_INIT;

    uint32_t color = renderer->normalPalette[0];
    uint32_t offset = 0;
    if (GBARegisterDISPCNTIsFrameSelect(renderer->dispcnt)) {
        offset = 0xA000;
    }
    if (mosaicWait && localX >= 0 && localY >= 0 && (localX >> 8) < 160 && (localY >> 8) < 128) {
        LOAD_16(color, offset + (localX >> 8) * 2 + (localY >> 8) * 320, renderer->vram);
        color = mColorFrom555(color);
    }

    int outX;
    uint32_t *pixel;
    for (outX = renderer->start, pixel = &renderer->row[outX]; outX < renderer->end; ++outX, ++pixel) {
        BACKGROUND_BITMAP_ITERATE(160, 128);

        if (!mosaicWait) {
            LOAD_16(color, offset + (localX >> 8) * 2 + (localY >> 8) * 320, renderer->vram);
            color = mColorFrom555(color);
            mosaicWait = mosaicH;
        } else {
            --mosaicWait;
        }

        uint32_t current = *pixel;
        if (!objwinSlowPath || (!(current & FLAG_OBJWIN)) != objwinOnly) {
            unsigned mergedFlags = flags;
            if (current & FLAG_OBJWIN) {
                mergedFlags = objwinFlags;
            }
            if (!variant) {
                _compositeBlendObjwin(renderer, pixel, color | mergedFlags, current);
            } else if (renderer->blendEffect == BLEND_BRIGHTEN) {
                _compositeBlendObjwin(renderer, pixel, _brighten(color, renderer->bldy) | mergedFlags, current);
            } else if (renderer->blendEffect == BLEND_DARKEN) {
                _compositeBlendObjwin(renderer, pixel, _darken(color, renderer->bldy) | mergedFlags, current);
            }
        }
    }
}

/* ================================================================== OBJ (software-obj.c, common.c) */

#define SPRITE_NORMAL_LOOP(DEPTH, TYPE)                         \
    SPRITE_YBASE_##DEPTH(inY);                                  \
    unsigned tileData;                                          \
    for (; outX < condition; ++outX, inX += xOffset) {          \
        SPRITE_XBASE_##DEPTH(inX);                              \
        SPRITE_DRAW_PIXEL_##DEPTH##_##TYPE(inX);                \
    }

#define SPRITE_MOSAIC_LOOP(DEPTH, TYPE)                         \
    SPRITE_YBASE_##DEPTH(inY);                                  \
    unsigned tileData;                                          \
    for (; outX < condition; ++outX, inX += xOffset) {          \
        int localX = inX - xOffset * (outX % mosaicH);          \
        if (localX < 0) {                                       \
            localX = 0;                                         \
        } else if (localX > width - 1) {                        \
            localX = width - 1;                                 \
        }                                                       \
        SPRITE_XBASE_##DEPTH(localX);                           \
        SPRITE_DRAW_PIXEL_##DEPTH##_##TYPE(localX);             \
    }

#define SPRITE_TRANSFORMED_LOOP(DEPTH, TYPE)                    \
    unsigned tileData;                                          \
    unsigned widthMask = ~(width - 1);                          \
    unsigned heightMask = ~(height - 1);                        \
    for (; outX < condition; ++outX, ++inX) {                   \
        xAccum += mat.a;                                        \
        yAccum += mat.c;                                        \
        int localX = xAccum >> 8;                               \
        int localY = yAccum >> 8;                               \
                                                                \
        if (localX & widthMask || localY & heightMask) {        \
            break;                                              \
        }                                                       \
                                                                \
        SPRITE_YBASE_##DEPTH(localY);                           \
        SPRITE_XBASE_##DEPTH(localX);                           \
        SPRITE_DRAW_PIXEL_##DEPTH##_##TYPE(localX);             \
    }

#define SPRITE_TRANSFORMED_MOSAIC_LOOP(DEPTH, TYPE)             \
    unsigned tileData;                                          \
    unsigned widthMask = ~(width - 1);                          \
    unsigned heightMask = ~(height - 1);                        \
    int localX = xAccum >> 8;                                   \
    int localY = yAccum >> 8;                                   \
    for (; outX < condition; ++outX, ++inX) {                   \
        xAccum += mat.a;                                        \
        yAccum += mat.c;                                        \
                                                                \
        if (outX % mosaicH == 0) {                              \
            localX = xAccum >> 8;                               \
            localY = yAccum >> 8;                               \
        }                                                       \
                                                                \
        if (localX & widthMask || localY & heightMask) {        \
            continue;                                           \
        }                                                       \
                                                                \
        SPRITE_YBASE_##DEPTH(localY);                           \
        SPRITE_XBASE_##DEPTH(localX);                           \
        SPRITE_DRAW_PIXEL_##DEPTH##_##TYPE(localX);             \
    }

#define SPRITE_XBASE_16(localX) unsigned xBase = (localX & ~0x7) * 4 + ((localX >> 1) & 2);
#define SPRITE_YBASE_16(localY) unsigned yBase = (localY & ~0x7) * stride + (localY & 0x7) * 4 + maskHi;

#define SPRITE_DRAW_PIXEL_16_NORMAL(localX)                                                        \
    LOAD_16(tileData, (yBase + ((xBase + charBase) & maskLo)) & 0x7FFE, vramBase);                \
    tileData = (tileData >> ((localX & 3) << 2)) & 0xF;                                            \
    current = renderer->spriteLayer[outX];                                                         \
    if ((current & FLAG_ORDER_MASK) > flags) {                                                     \
        if (tileData) {                                                                            \
            renderer->spriteLayer[outX] = palette[tileData] | flags;                               \
        } else if (current != FLAG_UNWRITTEN) {                                                    \
            renderer->spriteLayer[outX] = (current & ~(FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)) | \
                                          (flags & (FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)); \
        }                                                                                          \
    }

#define SPRITE_DRAW_PIXEL_16_NORMAL_OBJWIN(localX)                                                 \
    LOAD_16(tileData, (yBase + ((xBase + charBase) & maskLo)) & 0x7FFE, vramBase);                \
    tileData = (tileData >> ((localX & 3) << 2)) & 0xF;                                            \
    current = renderer->spriteLayer[outX];                                                         \
    if ((current & FLAG_ORDER_MASK) > flags) {                                                     \
        if (tileData) {                                                                            \
            unsigned color = (renderer->row[outX] & FLAG_OBJWIN) ? objwinPalette[tileData] : palette[tileData]; \
            renderer->spriteLayer[outX] = color | flags;                                           \
        } else if (current != FLAG_UNWRITTEN) {                                                    \
            renderer->spriteLayer[outX] = (current & ~(FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)) | \
                                          (flags & (FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)); \
        }                                                                                          \
    }

#define SPRITE_DRAW_PIXEL_16_OBJWIN(localX)                                                        \
    LOAD_16(tileData, (yBase + ((xBase + charBase) & maskLo)) & 0x7FFE, vramBase);                \
    tileData = (tileData >> ((localX & 3) << 2)) & 0xF;                                            \
    if (tileData) {                                                                                \
        renderer->row[outX] |= FLAG_OBJWIN;                                                        \
    }

#define SPRITE_XBASE_256(localX) unsigned xBase = (localX & ~0x7) * 8 + (localX & 6);
#define SPRITE_YBASE_256(localY) unsigned yBase = (localY & ~0x7) * stride + (localY & 0x7) * 8 + maskHi;

#define SPRITE_DRAW_PIXEL_256_NORMAL(localX)                                                       \
    LOAD_16(tileData, (yBase + ((xBase + charBase) & maskLo)) & 0x7FFE, vramBase);                \
    tileData = (tileData >> ((localX & 1) << 3)) & 0xFF;                                           \
    current = renderer->spriteLayer[outX];                                                         \
    if ((current & FLAG_ORDER_MASK) > flags) {                                                     \
        if (tileData) {                                                                            \
            renderer->spriteLayer[outX] = palette[tileData] | flags;                               \
        } else if (current != FLAG_UNWRITTEN) {                                                    \
            renderer->spriteLayer[outX] = (current & ~(FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)) | \
                                          (flags & (FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)); \
        }                                                                                          \
    }

#define SPRITE_DRAW_PIXEL_256_NORMAL_OBJWIN(localX)                                                \
    LOAD_16(tileData, (yBase + ((xBase + charBase) & maskLo)) & 0x7FFE, vramBase);                \
    tileData = (tileData >> ((localX & 1) << 3)) & 0xFF;                                           \
    current = renderer->spriteLayer[outX];                                                         \
    if ((current & FLAG_ORDER_MASK) > flags) {                                                     \
        if (tileData) {                                                                            \
            unsigned color = (renderer->row[outX] & FLAG_OBJWIN) ? objwinPalette[tileData] : palette[tileData]; \
            renderer->spriteLayer[outX] = color | flags;                                           \
        } else if (current != FLAG_UNWRITTEN) {                                                    \
            renderer->spriteLayer[outX] = (current & ~(FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)) | \
                                          (flags & (FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1)); \
        }                                                                                          \
    }

#define SPRITE_DRAW_PIXEL_256_OBJWIN(localX)                                                       \
    LOAD_16(tileData, (yBase + ((xBase + charBase) & maskLo)) & 0x7FFE, vramBase);                \
    tileData = (tileData >> ((localX & 1) << 3)) & 0xFF;                                           \
    if (tileData) {                                                                                \
        renderer->row[outX] |= FLAG_OBJWIN;                                                        \
    }

#ifdef PN_HAVE_AVX2
/* SPRITE_XBASE + SPRITE_DRAW_PIXEL_{16,256}_{NORMAL,NORMAL_OBJWIN,OBJWIN} (type 0, 1, 2) for the 8
 * output pixels at outX, lanes in `lanesOn` only, sprite coordinates localX (lx) and the per-lane
 * SPRITE_YBASE value. The 16-bit VRAM load is taken from the aligned 32-bit word that contains it
 * (addresses are masked to 0x7FFE, so the word stays inside OBJ VRAM whatever the lane). Lanes
 * outside `lanesOn` are stored back unchanged. */
PN_AVX2_INLINE void pn_sprite_pixels_avx2(PnPpu *renderer, int outX, __m256i lx, __m256i yBase, __m256i lanesOn,
                                          unsigned charBase, unsigned maskLo, const uint8_t *vramBase,
                                          const uint32_t *palette, const uint32_t *objwinPalette, uint32_t flags,
                                          const int bpp256, const int type, const int tail) {
    /* tail: the group may extend past the end of the run. Inside the 240-entry arrays the lanes
     * outside lanesOn are stored back unchanged; a group reaching past x = 240 uses masked moves
     * (which touch nothing else; they are slow on some CPUs, hence not used otherwise). */
    const int masked = tail && outX + 8 > GBA_VIDEO_HORIZONTAL_PIXELS;
    const __m256i vflags = _mm256_set1_epi32((int) flags);
    const __m256i owbit = _mm256_set1_epi32(FLAG_OBJWIN);
    const __m256i keepMask = _mm256_set1_epi32((int) (FLAG_ORDER_MASK | FLAG_REBLEND | FLAG_TARGET_1));
    const __m256i zero = _mm256_setzero_si256();
    __m256i xBase;
    if (bpp256) { /* (localX & ~0x7) * 8 + (localX & 6) */
        xBase = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(lx, _mm256_set1_epi32(~0x7)), 3),
                                 _mm256_and_si256(lx, _mm256_set1_epi32(6)));
    } else { /* (localX & ~0x7) * 4 + ((localX >> 1) & 2) */
        xBase = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(lx, _mm256_set1_epi32(~0x7)), 2),
                                 _mm256_and_si256(_mm256_srai_epi32(lx, 1), _mm256_set1_epi32(2)));
    }
    __m256i addr = _mm256_and_si256(
        _mm256_add_epi32(yBase, _mm256_and_si256(_mm256_add_epi32(xBase, _mm256_set1_epi32((int) charBase)),
                                                 _mm256_set1_epi32((int) maskLo))),
        _mm256_set1_epi32(0x7FFE));
    __m256i word = _mm256_i32gather_epi32((const int *) vramBase, _mm256_and_si256(addr, _mm256_set1_epi32(~3)), 1);
    __m256i half = _mm256_srlv_epi32(word, _mm256_slli_epi32(_mm256_and_si256(addr, _mm256_set1_epi32(2)), 3));
    __m256i tileData;
    if (bpp256) {
        tileData = _mm256_and_si256(_mm256_srlv_epi32(half, _mm256_slli_epi32(_mm256_and_si256(lx, _mm256_set1_epi32(1)), 3)),
                                    _mm256_set1_epi32(0xFF));
    } else {
        tileData = _mm256_and_si256(_mm256_srlv_epi32(half, _mm256_slli_epi32(_mm256_and_si256(lx, _mm256_set1_epi32(3)), 2)),
                                    _mm256_set1_epi32(0xF));
    }
    __m256i isZero = _mm256_cmpeq_epi32(tileData, zero);
    if (type == 2) { /* OBJ-window sprite: row[outX] |= FLAG_OBJWIN where the pixel is opaque */
        __m256i set = _mm256_and_si256(_mm256_andnot_si256(isZero, lanesOn), owbit);
        if (_mm256_testz_si256(set, set)) {
            return;
        }
        uint32_t *rowp = &renderer->row[outX];
        if (masked) {
            _mm256_maskstore_epi32((int *) rowp, lanesOn,
                                   _mm256_or_si256(_mm256_maskload_epi32((const int *) rowp, lanesOn), set));
        } else {
            _mm256_storeu_si256((__m256i *) rowp, _mm256_or_si256(_mm256_loadu_si256((const __m256i *) rowp), set));
        }
        return;
    }
    uint32_t *layer = &renderer->spriteLayer[outX];
    __m256i current = masked ? _mm256_maskload_epi32((const int *) layer, lanesOn) : _mm256_loadu_si256((const __m256i *) layer);
    /* (current & FLAG_ORDER_MASK) > flags */
    __m256i inFront = _mm256_and_si256(
        pn_ltu_avx2(vflags, _mm256_and_si256(current, _mm256_set1_epi32((int) FLAG_ORDER_MASK))), lanesOn);
    if (_mm256_testz_si256(inFront, inFront)) {
        return;
    }
    __m256i color = bpp256 ? _mm256_i32gather_epi32((const int *) palette, tileData, 4) : pn_lut16_avx2(palette, tileData);
    if (type == 1) {
        const uint32_t *rowp = &renderer->row[outX];
        __m256i rowv = masked ? _mm256_maskload_epi32((const int *) rowp, lanesOn) : _mm256_loadu_si256((const __m256i *) rowp);
        __m256i sel = _mm256_cmpeq_epi32(_mm256_and_si256(rowv, owbit), owbit);
        __m256i ocolor = bpp256 ? _mm256_i32gather_epi32((const int *) objwinPalette, tileData, 4)
                                : pn_lut16_avx2(objwinPalette, tileData);
        color = _mm256_blendv_epi8(color, ocolor, sel);
    }
    color = _mm256_or_si256(color, vflags);
    /* transparent pixel over a written one: take the sprite's order/reblend/target-1 flags */
    __m256i kept = _mm256_or_si256(_mm256_andnot_si256(keepMask, current), _mm256_and_si256(vflags, keepMask));
    __m256i unwritten = _mm256_cmpeq_epi32(current, _mm256_set1_epi32((int) FLAG_UNWRITTEN));
    __m256i value = _mm256_blendv_epi8(kept, current, unwritten);
    value = _mm256_blendv_epi8(color, value, isZero);
    if (masked) {
        _mm256_maskstore_epi32((int *) layer, inFront, value);
    } else {
        _mm256_storeu_si256((__m256i *) layer, _mm256_blendv_epi8(current, value, inFront));
    }
}

/* SPRITE_NORMAL_LOOP (non-affine, non-mosaic OBJ), 8 output pixels per step, the last step masked
 * to the run; returns condition (the scalar loop then has nothing left). */
PN_AVX2_INLINE int pn_sprite_normal_avx2(PnPpu *renderer, int outX, int condition, int inX, int xOffset,
                                         unsigned yBase, unsigned charBase, unsigned maskLo, const uint8_t *vramBase,
                                         const uint32_t *palette, const uint32_t *objwinPalette, uint32_t flags,
                                         const int bpp256, const int type) {
    const __m256i step = xOffset > 0 ? _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
                                     : _mm256_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7);
    const __m256i vyBase = _mm256_set1_epi32((int) yBase);
    const __m256i all = _mm256_cmpeq_epi32(vyBase, vyBase);
    for (; outX + 8 <= condition; outX += 8, inX += 8 * xOffset) {
        pn_sprite_pixels_avx2(renderer, outX, _mm256_add_epi32(_mm256_set1_epi32(inX), step), vyBase, all, charBase,
                              maskLo, vramBase, palette, objwinPalette, flags, bpp256, type, 0);
    }
    if (outX < condition) {
        __m256i lanesOn = _mm256_cmpgt_epi32(_mm256_set1_epi32(condition - outX), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        pn_sprite_pixels_avx2(renderer, outX, _mm256_add_epi32(_mm256_set1_epi32(inX), step), vyBase, lanesOn,
                              charBase, maskLo, vramBase, palette, objwinPalette, flags, bpp256, type, 1);
        outX = condition;
    }
    return outX;
}

/* SPRITE_TRANSFORMED_LOOP (affine, non-mosaic OBJ), 8 output pixels per step (the last step
 * masked to the run): lane j advances the accumulators by (j + 1) * mat.a / mat.c (32-bit
 * wrap-around like the scalar sums); the scalar loop breaks at the first pixel outside the
 * sprite, so only the lanes before the first such lane draw and the loop then stops. Returns -1
 * when the whole loop is done (always, unless the run is empty). */
PN_AVX2_INLINE int pn_sprite_affine_avx2(PnPpu *renderer, int outX, int condition, int *xAccum, int *yAccum,
                                         int matA, int matC, int width, int height, int stride, unsigned maskHi,
                                         unsigned charBase, unsigned maskLo, const uint8_t *vramBase,
                                         const uint32_t *palette, const uint32_t *objwinPalette, uint32_t flags,
                                         const int bpp256, const int type) {
    const __m256i lanes = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i stepX = _mm256_mullo_epi32(lanes, _mm256_set1_epi32(matA));
    const __m256i stepY = _mm256_mullo_epi32(lanes, _mm256_set1_epi32(matC));
    const __m256i widthMask = _mm256_set1_epi32(~(width - 1));
    const __m256i heightMask = _mm256_set1_epi32(~(height - 1));
    const __m256i vstride = _mm256_set1_epi32(stride);
    const __m256i vmaskHi = _mm256_set1_epi32((int) maskHi);
    const __m256i zero = _mm256_setzero_si256();
    int xa = *xAccum, ya = *yAccum;
    for (; outX < condition; outX += 8) {
        int remaining = condition - outX;
        __m256i lx = _mm256_srai_epi32(_mm256_add_epi32(_mm256_set1_epi32(xa), stepX), 8);
        __m256i ly = _mm256_srai_epi32(_mm256_add_epi32(_mm256_set1_epi32(ya), stepY), 8);
        __m256i outside = _mm256_or_si256(_mm256_and_si256(lx, widthMask), _mm256_and_si256(ly, heightMask));
        unsigned out = (unsigned) _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(outside, zero))) ^ 0xFFu;
        if (remaining < 8) {
            out &= (1u << remaining) - 1u; /* pixels past the run do not exist */
        }
        /* lanes before the first outside one (and inside the run) */
        unsigned on = out ? (1u << __builtin_ctz(out)) - 1u : remaining < 8 ? (1u << remaining) - 1u : 0xFFu;
        /* SPRITE_YBASE: (localY & ~0x7) * stride + (localY & 0x7) * (4 or 8) + maskHi */
        __m256i yBase = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(_mm256_and_si256(ly, _mm256_set1_epi32(~0x7)), vstride),
                             _mm256_slli_epi32(_mm256_and_si256(ly, _mm256_set1_epi32(0x7)), bpp256 ? 3 : 2)),
            vmaskHi);
        if (on) {
            __m256i lanesOn = _mm256_cmpgt_epi32(
                _mm256_and_si256(_mm256_set1_epi32((int) on), _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128)), zero);
            if (on == 0xFFu) {
                pn_sprite_pixels_avx2(renderer, outX, lx, yBase, lanesOn, charBase, maskLo, vramBase, palette,
                                      objwinPalette, flags, bpp256, type, 0);
            } else {
                pn_sprite_pixels_avx2(renderer, outX, lx, yBase, lanesOn, charBase, maskLo, vramBase, palette,
                                      objwinPalette, flags, bpp256, type, 1);
            }
        }
        if (out || remaining <= 8) {
            return -1; /* the scalar loop's break, or its end */
        }
        xa += 8 * matA;
        ya += 8 * matC;
    }
    *xAccum = xa;
    *yAccum = ya;
    return outX;
}

#define PN_SPRITE_AVX2_FNS(DEPTH, TYPE, BPP256, T)                                                                  \
    PN_AVX2_FN int pn_sprite_avx2_##DEPTH##_##TYPE(PnPpu *renderer, int outX, int condition, int inX, int xOffset, \
                                                   unsigned yBase, unsigned charBase, unsigned maskLo,             \
                                                   const uint8_t *vramBase, const uint32_t *palette,               \
                                                   const uint32_t *objwinPalette, uint32_t flags) {                \
        return pn_sprite_normal_avx2(renderer, outX, condition, inX, xOffset, yBase, charBase, maskLo, vramBase,  \
                                     palette, objwinPalette, flags, BPP256, T);                                   \
    }                                                                                                             \
    PN_AVX2_FN int pn_sprite_affine_avx2_##DEPTH##_##TYPE(                                                        \
        PnPpu *renderer, int outX, int condition, int *xAccum, int *yAccum, int matA, int matC, int width,        \
        int height, int stride, unsigned maskHi, unsigned charBase, unsigned maskLo, const uint8_t *vramBase,      \
        const uint32_t *palette, const uint32_t *objwinPalette, uint32_t flags) {                                 \
        return pn_sprite_affine_avx2(renderer, outX, condition, xAccum, yAccum, matA, matC, width, height, stride, \
                                     maskHi, charBase, maskLo, vramBase, palette, objwinPalette, flags, BPP256, T); \
    }
PN_SPRITE_AVX2_FNS(16, NORMAL, 0, 0)
PN_SPRITE_AVX2_FNS(16, NORMAL_OBJWIN, 0, 1)
PN_SPRITE_AVX2_FNS(16, OBJWIN, 0, 2)
PN_SPRITE_AVX2_FNS(256, NORMAL, 1, 0)
PN_SPRITE_AVX2_FNS(256, NORMAL_OBJWIN, 1, 1)
PN_SPRITE_AVX2_FNS(256, OBJWIN, 1, 2)

/* Runs the AVX2 kernel on the leading whole groups of 8 pixels, then SPRITE_NORMAL_LOOP finishes. */
#define SPRITE_NORMAL_LOOP_FAST(DEPTH, TYPE)                                                                    \
    if (renderer->simd == PN_PPU_SIMD_AVX2) {                                                                   \
        unsigned yBaseFast = (inY & ~0x7) * stride + (inY & 0x7) * (DEPTH == 16 ? 4 : 8) + maskHi;              \
        int newOutX = pn_sprite_avx2_##DEPTH##_##TYPE(renderer, outX, condition, inX, xOffset, yBaseFast,       \
                                                     charBase, maskLo, vramBase, palette, objwinPalette, flags); \
        inX += (newOutX - outX) * xOffset;                                                                      \
        outX = newOutX;                                                                                         \
    }                                                                                                           \
    SPRITE_NORMAL_LOOP(DEPTH, TYPE)

/* Same for SPRITE_TRANSFORMED_LOOP; the kernel returns -1 when it reached the loop's `break`. */
#define SPRITE_TRANSFORMED_LOOP_FAST(DEPTH, TYPE)                                                               \
    if (renderer->simd == PN_PPU_SIMD_AVX2) {                                                                   \
        int newOutX = pn_sprite_affine_avx2_##DEPTH##_##TYPE(renderer, outX, condition, &xAccum, &yAccum, mat.a, \
                                                            mat.c, width, height, stride, maskHi, charBase,     \
                                                            maskLo, vramBase, palette, objwinPalette, flags);  \
        if (newOutX < 0) {                                                                                      \
            return 1;                                                                                           \
        }                                                                                                       \
        inX += newOutX - outX;                                                                                  \
        outX = newOutX;                                                                                         \
    }                                                                                                           \
    SPRITE_TRANSFORMED_LOOP(DEPTH, TYPE)
#else
#define SPRITE_NORMAL_LOOP_FAST(DEPTH, TYPE) SPRITE_NORMAL_LOOP(DEPTH, TYPE)
#define SPRITE_TRANSFORMED_LOOP_FAST(DEPTH, TYPE) SPRITE_TRANSFORMED_LOOP(DEPTH, TYPE)
#endif

typedef struct PnOAMMatrix {
    int16_t a, b, c, d;
} PnOAMMatrix;

static int pn_clean_oam(const uint8_t *oam, PnSprite *sprites, int offsetY) {
    int i;
    int oamMax = 0;
    for (i = 0; i < 128; ++i) {
        PnObj obj;
        obj.a = (uint16_t) ld16(oam, i * 8 + 0);
        obj.b = (uint16_t) ld16(oam, i * 8 + 2);
        obj.c = (uint16_t) ld16(oam, i * 8 + 4);
        obj.d = (uint16_t) ld16(oam, i * 8 + 6);
        if (GBAObjAttributesAIsTransformed(obj.a) || !GBAObjAttributesAIsDisable(obj.a)) {
            int width = GBAVideoObjSizes[GBAObjAttributesAGetShape(obj.a) * 4 + GBAObjAttributesBGetSize(obj.b)][0];
            int height = GBAVideoObjSizes[GBAObjAttributesAGetShape(obj.a) * 4 + GBAObjAttributesBGetSize(obj.b)][1];
            int cycles = width;
            if (GBAObjAttributesAIsTransformed(obj.a)) {
                height <<= GBAObjAttributesAGetDoubleSize(obj.a);
                width <<= GBAObjAttributesAGetDoubleSize(obj.a);
                cycles = 10 + width * 2;
            }
            if (GBAObjAttributesAGetY(obj.a) >= GBA_VIDEO_VERTICAL_PIXELS &&
                GBAObjAttributesAGetY(obj.a) + height < VIDEO_VERTICAL_TOTAL_PIXELS) {
                continue;
            }
            if (GBAObjAttributesBGetX(obj.b) >= GBA_VIDEO_HORIZONTAL_PIXELS && GBAObjAttributesBGetX(obj.b) + width < 512) {
                continue;
            }
            int y = GBAObjAttributesAGetY(obj.a) + offsetY;
            sprites[oamMax].y = (int16_t) y;
            sprites[oamMax].endY = (int16_t) (y + height);
            sprites[oamMax].cycles = (int16_t) cycles;
            sprites[oamMax].obj = obj;
            sprites[oamMax].index = (int8_t) i;
            ++oamMax;
        }
    }
    return oamMax;
}

/* The sprite loops below write spriteLayer[x] (OBJ-window sprites: row[x]) for x in [from, to).
 * Records that range for the next line's reset of spriteLayer. Only an affine OBJ-mosaic sprite
 * can reach past x = 240 (mGBA's condition rounding, fuzz-only): spriteLayer[240..] are the fields
 * after it (not reset), row[240 + k] is spriteLayer[k] (same layout as mGBA). */
static inline void pn_mark_sprite_layer(PnPpu *renderer, int from, int to, int objwinSprite) {
    if (from >= to) {
        return;
    }
    if (objwinSprite) {
        if (to <= GBA_VIDEO_HORIZONTAL_PIXELS) {
            return;
        }
        from = 0;
        to -= GBA_VIDEO_HORIZONTAL_PIXELS;
    }
    if (from < renderer->spriteDirtyStart) {
        renderer->spriteDirtyStart = from;
    }
    if (to > renderer->spriteDirtyEnd) {
        renderer->spriteDirtyEnd = to;
    }
}

static int pn_preprocess_sprite(PnPpu *renderer, const PnObj *sprite, int index, int y) {
    (void) index;
    int width = GBAVideoObjSizes[GBAObjAttributesAGetShape(sprite->a) * 4 + GBAObjAttributesBGetSize(sprite->b)][0];
    int height = GBAVideoObjSizes[GBAObjAttributesAGetShape(sprite->a) * 4 + GBAObjAttributesBGetSize(sprite->b)][1];
    int start = renderer->start;
    int end = renderer->end;
    uint32_t flags = (uint32_t) GBAObjAttributesCGetPriority(sprite->c) << OFFSET_PRIORITY;
    flags |= FLAG_TARGET_1 * ((GBAWindowControlIsBlendEnable(renderer->currentWindow.packed) && renderer->target1Obj &&
                               renderer->blendEffect == BLEND_ALPHA) ||
                              GBAObjAttributesAGetMode(sprite->a) == OBJ_MODE_SEMITRANSPARENT);
    flags |= FLAG_OBJWIN * (GBAObjAttributesAGetMode(sprite->a) == OBJ_MODE_OBJWIN);
    if ((flags & FLAG_OBJWIN) && (renderer->currentWindow.priority < renderer->objwin.priority)) {
        return 0;
    }
    int32_t x = (int32_t) ((uint32_t) GBAObjAttributesBGetX(sprite->b) << 23);
    x >>= 23;
    const uint8_t *vramBase = &renderer->vram[BASE_TILE];
    unsigned align = GBAObjAttributesAIs256Color(sprite->a) && !GBARegisterDISPCNTIsObjCharacterMapping(renderer->dispcnt);
    unsigned charBase = (GBAObjAttributesCGetTile(sprite->c) & ~align) * 0x20;
    unsigned maskLo = GBARegisterDISPCNTIsObjCharacterMapping(renderer->dispcnt) ? 0x7FFE : 0x3FE;
    unsigned maskHi = GBARegisterDISPCNTIsObjCharacterMapping(renderer->dispcnt) ? 0 : charBase & 0x7C00;
    if (GBARegisterDISPCNTGetMode(renderer->dispcnt) >= 3 && GBAObjAttributesCGetTile(sprite->c) < 512) {
        return 0;
    }

    /* Kept exactly as mGBA: GetBlendEnable (0/1) is compared with IsBlendEnable (0/0x20). */
    int objwinSlowPath = GBARegisterDISPCNTIsObjwinEnable(renderer->dispcnt) &&
                         GBAWindowControlGetBlendEnable(renderer->objwin.packed) !=
                             GBAWindowControlIsBlendEnable(renderer->currentWindow.packed);
    int variant = renderer->target1Obj && GBAWindowControlIsBlendEnable(renderer->currentWindow.packed) &&
                  (renderer->blendEffect == BLEND_BRIGHTEN || renderer->blendEffect == BLEND_DARKEN);
    if (GBAObjAttributesAGetMode(sprite->a) == OBJ_MODE_SEMITRANSPARENT ||
        (renderer->target1Obj && renderer->blendEffect == BLEND_ALPHA) || objwinSlowPath) {
        int target2 = renderer->target2Bd;
        target2 |= renderer->bg[0].target2 && renderer->bg[0].enabled;
        target2 |= renderer->bg[1].target2 && renderer->bg[1].enabled;
        target2 |= renderer->bg[2].target2 && renderer->bg[2].enabled;
        target2 |= renderer->bg[3].target2 && renderer->bg[3].enabled;
        if (target2) {
            renderer->forceTarget1 = true;
            flags |= FLAG_REBLEND;
            variant = 0;
        } else {
            flags &= ~FLAG_TARGET_1;
        }
    }

    const uint32_t *palette = &renderer->normalPalette[0x100];
    const uint32_t *objwinPalette = palette;

    if (variant) {
        palette = &renderer->variantPalette[0x100];
        if (GBAWindowControlIsBlendEnable(renderer->objwin.packed)) {
            objwinPalette = palette;
        }
    }

    int inY = y - (int) GBAObjAttributesAGetY(sprite->a);
    int stride = GBARegisterDISPCNTIsObjCharacterMapping(renderer->dispcnt)
                     ? (width >> !GBAObjAttributesAIs256Color(sprite->a))
                     : 0x80;

    uint32_t current;
    if (GBAObjAttributesAIsTransformed(sprite->a)) {
        int totalWidth = width << GBAObjAttributesAGetDoubleSize(sprite->a);
        int totalHeight = height << GBAObjAttributesAGetDoubleSize(sprite->a);
        PnOAMMatrix mat;
        unsigned matIndex = GBAObjAttributesBGetMatIndex(sprite->b);
        mat.a = (int16_t) ld16(renderer->oam, matIndex * 32 + 6);
        mat.b = (int16_t) ld16(renderer->oam, matIndex * 32 + 14);
        mat.c = (int16_t) ld16(renderer->oam, matIndex * 32 + 22);
        mat.d = (int16_t) ld16(renderer->oam, matIndex * 32 + 30);

        if (inY < 0) {
            inY += 256;
        }
        int outX = x >= start ? x : start;
        int condition = x + totalWidth;
        int inX = outX - x;
        if (end < condition) {
            condition = end;
        }
        int mosaicH = 1;
        if (GBAObjAttributesAIsMosaic(sprite->a)) {
            mosaicH = GBAMosaicControlGetObjH(renderer->mosaic) + 1;
            if (condition != end && condition % mosaicH) {
                condition += mosaicH - (condition % mosaicH);
            }
        }

        int xAccum = mat.a * (inX - 1 - (totalWidth >> 1)) + mat.b * (inY - (totalHeight >> 1)) + (width << 7);
        int yAccum = mat.c * (inX - 1 - (totalWidth >> 1)) + mat.d * (inY - (totalHeight >> 1)) + (height << 7);

        /* Clip off early pixels */
        if (mat.a) {
            if ((xAccum >> 8) < 0) {
                int32_t diffX = -xAccum - 1;
                int32_t x = mat.a ? diffX / mat.a : 0;
                xAccum += mat.a * x;
                yAccum += mat.c * x;
                outX += x;
                inX += x;
            } else if ((xAccum >> 8) >= width) {
                int32_t diffX = (width << 8) - xAccum;
                int32_t x = mat.a ? diffX / mat.a : 0;
                xAccum += mat.a * x;
                yAccum += mat.c * x;
                outX += x;
                inX += x;
            }
        }
        if (mat.c) {
            if ((yAccum >> 8) < 0) {
                int32_t diffY = -yAccum - 1;
                int32_t y = mat.c ? diffY / mat.c : 0;
                xAccum += mat.a * y;
                yAccum += mat.c * y;
                outX += y;
                inX += y;
            } else if ((yAccum >> 8) >= height) {
                int32_t diffY = (height << 8) - yAccum;
                int32_t y = mat.c ? diffY / mat.c : 0;
                xAccum += mat.a * y;
                yAccum += mat.c * y;
                outX += y;
                inX += y;
            }
        }

        if (outX < start || outX >= condition) {
            return 0;
        }
        pn_mark_sprite_layer(renderer, outX, condition, (flags & FLAG_OBJWIN) != 0);

        if (!GBAObjAttributesAIs256Color(sprite->a)) {
            palette = &palette[GBAObjAttributesCGetPalette(sprite->c) << 4];
            if (flags & FLAG_OBJWIN) {
                SPRITE_TRANSFORMED_LOOP_FAST(16, OBJWIN);
            } else if (mosaicH > 1) {
                if (objwinSlowPath) {
                    objwinPalette = &objwinPalette[GBAObjAttributesCGetPalette(sprite->c) << 4];
                    SPRITE_TRANSFORMED_MOSAIC_LOOP(16, NORMAL_OBJWIN);
                } else {
                    SPRITE_TRANSFORMED_MOSAIC_LOOP(16, NORMAL);
                }
            } else if (objwinSlowPath) {
                objwinPalette = &objwinPalette[GBAObjAttributesCGetPalette(sprite->c) << 4];
                SPRITE_TRANSFORMED_LOOP_FAST(16, NORMAL_OBJWIN);
            } else {
                SPRITE_TRANSFORMED_LOOP_FAST(16, NORMAL);
            }
        } else {
            if (flags & FLAG_OBJWIN) {
                SPRITE_TRANSFORMED_LOOP_FAST(256, OBJWIN);
            } else if (mosaicH > 1) {
                if (objwinSlowPath) {
                    SPRITE_TRANSFORMED_MOSAIC_LOOP(256, NORMAL_OBJWIN);
                } else {
                    SPRITE_TRANSFORMED_MOSAIC_LOOP(256, NORMAL);
                }
            } else if (objwinSlowPath) {
                SPRITE_TRANSFORMED_LOOP_FAST(256, NORMAL_OBJWIN);
            } else {
                SPRITE_TRANSFORMED_LOOP_FAST(256, NORMAL);
            }
        }
    } else {
        int outX = x >= start ? x : start;
        int condition = x + width;
        int mosaicH = 1;
        if (GBAObjAttributesAIsMosaic(sprite->a)) {
            mosaicH = GBAMosaicControlGetObjH(renderer->mosaic) + 1;
            if (condition % mosaicH) {
                condition += mosaicH - (condition % mosaicH);
            }
        }
        if ((int) GBAObjAttributesAGetY(sprite->a) + height - 256 >= 0) {
            inY += 256;
        }
        if (GBAObjAttributesBIsVFlip(sprite->b)) {
            inY = height - inY - 1;
        }
        if (end < condition) {
            condition = end;
        }
        int inX = outX - x;
        int xOffset = 1;
        if (GBAObjAttributesBIsHFlip(sprite->b)) {
            inX = width - inX - 1;
            xOffset = -1;
        }
        pn_mark_sprite_layer(renderer, outX, condition, (flags & FLAG_OBJWIN) != 0);
        if (!GBAObjAttributesAIs256Color(sprite->a)) {
            palette = &palette[GBAObjAttributesCGetPalette(sprite->c) << 4];
            if (flags & FLAG_OBJWIN) {
                SPRITE_NORMAL_LOOP_FAST(16, OBJWIN);
            } else if (mosaicH > 1) {
                if (objwinSlowPath) {
                    objwinPalette = &objwinPalette[GBAObjAttributesCGetPalette(sprite->c) << 4];
                    SPRITE_MOSAIC_LOOP(16, NORMAL_OBJWIN);
                } else {
                    SPRITE_MOSAIC_LOOP(16, NORMAL);
                }
            } else if (objwinSlowPath) {
                objwinPalette = &objwinPalette[GBAObjAttributesCGetPalette(sprite->c) << 4];
                SPRITE_NORMAL_LOOP_FAST(16, NORMAL_OBJWIN);
            } else {
                SPRITE_NORMAL_LOOP_FAST(16, NORMAL);
            }
        } else {
            if (flags & FLAG_OBJWIN) {
                SPRITE_NORMAL_LOOP_FAST(256, OBJWIN);
            } else if (mosaicH > 1) {
                if (objwinSlowPath) {
                    SPRITE_MOSAIC_LOOP(256, NORMAL_OBJWIN);
                } else {
                    SPRITE_MOSAIC_LOOP(256, NORMAL);
                }
            } else if (objwinSlowPath) {
                SPRITE_NORMAL_LOOP_FAST(256, NORMAL_OBJWIN);
            } else {
                SPRITE_NORMAL_LOOP_FAST(256, NORMAL);
            }
        }
    }
    return 1;
}

#ifdef PN_HAVE_AVX2
/* One of the four loops of pn_postprocess_sprite below (mode 0: objwinDisable, 1: objwinOnly, 2: OBJ
 * window on, 3: no OBJ window), 8 pixels at a time from renderer->start while whole groups fit
 * before renderer->end; returns the x where the scalar loop continues. */
PN_AVX2_INLINE int pn_postprocess_sprite_avx2(PnPpu *renderer, unsigned priority, uint32_t flags, const int mode) {
    const __m256i unwritten = _mm256_set1_epi32((int) FLAG_UNWRITTEN);
    const __m256i owbit = _mm256_set1_epi32(FLAG_OBJWIN);
    const __m256i prio = _mm256_set1_epi32((int) priority);
    const __m256i vflags = _mm256_set1_epi32((int) flags);
    int x = renderer->start;
    for (; x + 8 <= renderer->end; x += 8) {
        __m256i color = _mm256_andnot_si256(owbit, _mm256_loadu_si256((const __m256i *) &renderer->spriteLayer[x]));
        __m256i m = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(color, unwritten), unwritten),
                                        _mm256_cmpeq_epi32(_mm256_srli_epi32(color, OFFSET_PRIORITY), prio));
        if (_mm256_testz_si256(m, m)) {
            continue;
        }
        __m256i current = _mm256_loadu_si256((const __m256i *) &renderer->row[x]);
        if (mode == 0) {
            m = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(current, owbit), owbit), m);
        } else if (mode == 1) {
            m = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(current, owbit), owbit), m);
        }
        __m256i out = pn_composite_avx2(renderer, current, _mm256_or_si256(color, vflags), m, 1, mode != 3);
        _mm256_storeu_si256((__m256i *) &renderer->row[x], out);
    }
    return x;
}
PN_AVX2_FN int pn_postprocess_sprite_avx2_0(PnPpu *r, unsigned p, uint32_t f) { return pn_postprocess_sprite_avx2(r, p, f, 0); }
PN_AVX2_FN int pn_postprocess_sprite_avx2_1(PnPpu *r, unsigned p, uint32_t f) { return pn_postprocess_sprite_avx2(r, p, f, 1); }
PN_AVX2_FN int pn_postprocess_sprite_avx2_2(PnPpu *r, unsigned p, uint32_t f) { return pn_postprocess_sprite_avx2(r, p, f, 2); }
PN_AVX2_FN int pn_postprocess_sprite_avx2_3(PnPpu *r, unsigned p, uint32_t f) { return pn_postprocess_sprite_avx2(r, p, f, 3); }
#define PN_POSTPROCESS_SPRITE_AVX2(MODE)                                         \
    if (renderer->simd == PN_PPU_SIMD_AVX2) {                                    \
        x = pn_postprocess_sprite_avx2_##MODE(renderer, priority, flags);        \
        pixel = &renderer->row[x];                                               \
    }
#else
#define PN_POSTPROCESS_SPRITE_AVX2(MODE)
#endif

static void pn_postprocess_sprite_segment(PnPpu *renderer, unsigned priority);

static void pn_postprocess_sprite(PnPpu *renderer, unsigned priority) {
    /* Outside [spriteDirtyStart, spriteDirtyEnd) every spriteLayer entry is FLAG_UNWRITTEN, which
     * the loops below skip: only that part of the segment is visited. */
    int segmentStart = renderer->start, segmentEnd = renderer->end;
    if (renderer->start < renderer->spriteDirtyStart) {
        renderer->start = renderer->spriteDirtyStart;
    }
    if (renderer->end > renderer->spriteDirtyEnd) {
        renderer->end = renderer->spriteDirtyEnd;
    }
    pn_postprocess_sprite_segment(renderer, priority);
    renderer->start = segmentStart;
    renderer->end = segmentEnd;
}

static void pn_postprocess_sprite_segment(PnPpu *renderer, unsigned priority) {
    int x = renderer->start;
    uint32_t *pixel = &renderer->row[renderer->start];
    uint32_t flags = FLAG_TARGET_2 * renderer->target2Obj;

    int objwinSlowPath = GBARegisterDISPCNTIsObjwinEnable(renderer->dispcnt);
    bool objwinDisable = false;
    bool objwinOnly = false;
    if (objwinSlowPath) {
        objwinDisable = !GBAWindowControlIsObjEnable(renderer->objwin.packed);
        objwinOnly = !objwinDisable && !GBAWindowControlIsObjEnable(renderer->currentWindow.packed);
        if (objwinDisable && !GBAWindowControlIsObjEnable(renderer->currentWindow.packed)) {
            return;
        }

        if (objwinDisable) {
            PN_POSTPROCESS_SPRITE_AVX2(0)
            for (; x < renderer->end; ++x, ++pixel) {
                uint32_t color = renderer->spriteLayer[x] & ~FLAG_OBJWIN;
                uint32_t current = *pixel;
                if ((color & FLAG_UNWRITTEN) != FLAG_UNWRITTEN && !(current & FLAG_OBJWIN) &&
                    (color & FLAG_PRIORITY) >> OFFSET_PRIORITY == priority) {
                    _compositeBlendObjwin(renderer, pixel, color | flags, current);
                }
            }
            return;
        } else if (objwinOnly) {
            PN_POSTPROCESS_SPRITE_AVX2(1)
            for (; x < renderer->end; ++x, ++pixel) {
                uint32_t color = renderer->spriteLayer[x] & ~FLAG_OBJWIN;
                uint32_t current = *pixel;
                if ((color & FLAG_UNWRITTEN) != FLAG_UNWRITTEN && (current & FLAG_OBJWIN) &&
                    (color & FLAG_PRIORITY) >> OFFSET_PRIORITY == priority) {
                    _compositeBlendObjwin(renderer, pixel, color | flags, current);
                }
            }
            return;
        } else {
            PN_POSTPROCESS_SPRITE_AVX2(2)
            for (; x < renderer->end; ++x, ++pixel) {
                uint32_t color = renderer->spriteLayer[x] & ~FLAG_OBJWIN;
                uint32_t current = *pixel;
                if ((color & FLAG_UNWRITTEN) != FLAG_UNWRITTEN && (color & FLAG_PRIORITY) >> OFFSET_PRIORITY == priority) {
                    _compositeBlendObjwin(renderer, pixel, color | flags, current);
                }
            }
            return;
        }
    } else if (!GBAWindowControlIsObjEnable(renderer->currentWindow.packed)) {
        return;
    }
    PN_POSTPROCESS_SPRITE_AVX2(3)
    for (; x < renderer->end; ++x, ++pixel) {
        uint32_t color = renderer->spriteLayer[x] & ~FLAG_OBJWIN;
        uint32_t current = *pixel;
        if ((color & FLAG_UNWRITTEN) != FLAG_UNWRITTEN && (color & FLAG_PRIORITY) >> OFFSET_PRIORITY == priority) {
            _compositeBlendNoObjwin(renderer, pixel, color | flags, current);
        }
    }
}

/* renderer->spriteLines[y]: bit i set iff sprites[i] passes the line test of
 * pn_preprocess_sprite_layer at line y, i.e. y < endY && (y >= y0 || (endY >= 256 && y < endY - 256)):
 * [y0, endY) and [0, endY - 256) clipped to the screen. Rebuilt with the sprite list. */
static void pn_bucket_sprites(PnPpu *renderer) {
    int i, line;
    memset(renderer->spriteLines, 0, sizeof(renderer->spriteLines));
    for (i = 0; i < renderer->oamMax; ++i) {
        const PnSprite *sprite = &renderer->sprites[i];
        uint32_t bit = 1u << (i & 31);
        int word = i >> 5;
        int a = sprite->y, b = sprite->endY;
        if (a < 0) {
            a = 0;
        }
        for (line = a; line < b && line < GBA_VIDEO_VERTICAL_PIXELS; ++line) {
            renderer->spriteLines[line][word] |= bit;
        }
        for (line = 0; line < sprite->endY - 256 && line < GBA_VIDEO_VERTICAL_PIXELS; ++line) {
            renderer->spriteLines[line][word] |= bit;
        }
    }
}

static int pn_preprocess_sprite_layer(PnPpu *renderer, int y) {
    int w;
    int spriteLayers = 0;
    if (GBARegisterDISPCNTIsObjEnable(renderer->dispcnt)) {
        if (renderer->oamDirty) {
            renderer->oamMax = pn_clean_oam(renderer->oam, renderer->sprites, 0);
            renderer->oamDirty = false;
            pn_bucket_sprites(renderer);
        }
        int mosaicV = GBAMosaicControlGetObjV(renderer->mosaic) + 1;
        int mosaicY = mosaicV > 1 ? y - (y % mosaicV) : y;
        int i;
        /* mGBA tests every sprite of the list in index order; the line's bucket holds exactly the
         * ones that pass, and is walked in the same order. (mGBA also sets renderer->end = 0 for
         * the skipped ones; it is reassigned before any use.) */
        const uint32_t *lineBits = renderer->spriteLines[y < 0 ? 0 : y >= GBA_VIDEO_VERTICAL_PIXELS ? GBA_VIDEO_VERTICAL_PIXELS - 1 : y];
        int word;
        for (word = 0, i = 0; word < 4; ++word) {
            uint32_t bits = lineBits[word];
            for (; bits; bits &= bits - 1) {
                i = word * 32 + __builtin_ctz(bits);
                PnSprite *sprite = &renderer->sprites[i];
                int localY = y;
                renderer->end = 0;
                if ((y < sprite->y && (sprite->endY - 256 < 0 || y >= sprite->endY - 256)) || y >= sprite->endY) {
                    continue;
                }
                if (GBAObjAttributesAIsMosaic(sprite->obj.a) && mosaicV > 1) {
                    localY = mosaicY;
                    if (localY < sprite->y && sprite->y < GBA_VIDEO_VERTICAL_PIXELS) {
                        localY = sprite->y;
                    }
                    if (localY >= (sprite->endY & 0xFF)) {
                        localY = sprite->endY - 1;
                    }
                }
                for (w = 0; w < renderer->nWindows; ++w) {
                    renderer->currentWindow = renderer->windows[w].control;
                    renderer->start = renderer->end;
                    renderer->end = renderer->windows[w].endX;
                    if (!GBAWindowControlIsObjEnable(renderer->currentWindow.packed) &&
                        !GBARegisterDISPCNTIsObjwinEnable(renderer->dispcnt)) {
                        continue;
                    }

                    int drawn = pn_preprocess_sprite(renderer, &sprite->obj, sprite->index, localY);
                    spriteLayers |= drawn << GBAObjAttributesCGetPriority(sprite->obj.c);
                }
                renderer->spriteCyclesRemaining -= sprite->cycles;
                if (renderer->spriteCyclesRemaining <= 0) {
                    return spriteLayers;
                }
            }
        }
    }
    return spriteLayers;
}

/* ================================================================== registers, palettes (video-software.c) */

static void _updatePalettes(PnPpu *renderer) {
    int i;
    if (renderer->blendEffect == BLEND_BRIGHTEN) {
        for (i = 0; i < 512; ++i) {
            renderer->variantPalette[i] = _brighten(renderer->normalPalette[i], renderer->bldy);
        }
    } else if (renderer->blendEffect == BLEND_DARKEN) {
        for (i = 0; i < 512; ++i) {
            renderer->variantPalette[i] = _darken(renderer->normalPalette[i], renderer->bldy);
        }
    } else {
        for (i = 0; i < 512; ++i) {
            renderer->variantPalette[i] = renderer->normalPalette[i];
        }
    }
}

static void _updateFlags(PnPpu *renderer, PnBackground *background) {
    uint32_t flags = (background->priority << OFFSET_PRIORITY) | (background->index << OFFSET_INDEX) | FLAG_IS_BACKGROUND;
    if (background->target2) {
        flags |= FLAG_TARGET_2;
    }
    uint32_t objwinFlags = flags;
    if (renderer->blendEffect == BLEND_ALPHA) {
        if (renderer->blda == 0x10 && renderer->bldb == 0) {
            flags &= ~FLAG_TARGET_2;
            objwinFlags &= ~FLAG_TARGET_2;
        } else if (background->target1) {
            if (GBAWindowControlIsBlendEnable(renderer->currentWindow.packed)) {
                flags |= FLAG_TARGET_1;
            }
            if (GBAWindowControlIsBlendEnable(renderer->objwin.packed)) {
                objwinFlags |= FLAG_TARGET_1;
            }
        }
    }
    background->flags = flags;
    background->objwinFlags = objwinFlags;
    background->variant = background->target1 && GBAWindowControlIsBlendEnable(renderer->currentWindow.packed) &&
                          (renderer->blendEffect == BLEND_BRIGHTEN || renderer->blendEffect == BLEND_DARKEN);
}

static void _enableBg(PnPpu *renderer, int bg, bool active) {
    int wasActive = renderer->bg[bg].enabled;
    if (!active) {
        if (renderer->nextY == 0 || (wasActive > 0 && wasActive < ENABLED_MAX)) {
            renderer->bg[bg].enabled = 0;
        } else if (wasActive == ENABLED_MAX) {
            renderer->bg[bg].enabled = -2;
        }
    } else if (!wasActive && active) {
        if (renderer->nextY == 0) {
            renderer->bg[bg].enabled = ENABLED_MAX;
        } else if (GBARegisterDISPCNTGetMode(renderer->dispcnt) > 2) {
            renderer->bg[bg].enabled = 2;
        } else {
            renderer->bg[bg].enabled = 1;
        }
    } else if (wasActive < 0 && active) {
        renderer->bg[bg].enabled = ENABLED_MAX;
    }
}

static void _updateDISPCNT(PnPpu *renderer) {
    _enableBg(renderer, 0, GBARegisterDISPCNTGetBg0Enable(renderer->dispcnt));
    _enableBg(renderer, 1, GBARegisterDISPCNTGetBg1Enable(renderer->dispcnt));
    _enableBg(renderer, 2, GBARegisterDISPCNTGetBg2Enable(renderer->dispcnt));
    _enableBg(renderer, 3, GBARegisterDISPCNTGetBg3Enable(renderer->dispcnt));
}

static void _writeBGCNT(PnPpu *renderer, PnBackground *bg, uint16_t value) {
    bg->priority = GBARegisterBGCNTGetPriority(value);
    bg->charBase = GBARegisterBGCNTGetCharBase(value) << 14;
    bg->mosaic = GBARegisterBGCNTGetMosaic(value);
    bg->multipalette = GBARegisterBGCNTGet256Color(value);
    bg->screenBase = GBARegisterBGCNTGetScreenBase(value) << 11;
    bg->overflow = GBARegisterBGCNTGetOverflow(value);
    bg->size = GBARegisterBGCNTGetSize(value);
    bg->yCache = -1;

    _updateFlags(renderer, bg);
}

static void _writeBGX_LO(PnBackground *bg, uint16_t value) {
    bg->refx = (int32_t) (((uint32_t) bg->refx & 0xFFFF0000u) | value);
    bg->sx = bg->refx;
}

static void _writeBGX_HI(PnBackground *bg, uint16_t value) {
    uint32_t r = ((uint32_t) bg->refx & 0x0000FFFFu) | ((uint32_t) value << 16);
    bg->refx = (int32_t) (r << 4) >> 4;
    bg->sx = bg->refx;
}

static void _writeBGY_LO(PnBackground *bg, uint16_t value) {
    bg->refy = (int32_t) (((uint32_t) bg->refy & 0xFFFF0000u) | value);
    bg->sy = bg->refy;
}

static void _writeBGY_HI(PnBackground *bg, uint16_t value) {
    uint32_t r = ((uint32_t) bg->refy & 0x0000FFFFu) | ((uint32_t) value << 16);
    bg->refy = (int32_t) (r << 4) >> 4;
    bg->sy = bg->refy;
}

static void _writeBLDCNT(PnPpu *renderer, uint16_t value) {
    int oldEffect = renderer->blendEffect;

    renderer->bg[0].target1 = GBARegisterBLDCNTGetTarget1Bg0(value);
    renderer->bg[1].target1 = GBARegisterBLDCNTGetTarget1Bg1(value);
    renderer->bg[2].target1 = GBARegisterBLDCNTGetTarget1Bg2(value);
    renderer->bg[3].target1 = GBARegisterBLDCNTGetTarget1Bg3(value);
    renderer->bg[0].target2 = GBARegisterBLDCNTGetTarget2Bg0(value);
    renderer->bg[1].target2 = GBARegisterBLDCNTGetTarget2Bg1(value);
    renderer->bg[2].target2 = GBARegisterBLDCNTGetTarget2Bg2(value);
    renderer->bg[3].target2 = GBARegisterBLDCNTGetTarget2Bg3(value);

    renderer->blendEffect = GBARegisterBLDCNTGetEffect(value);
    renderer->target1Obj = GBARegisterBLDCNTGetTarget1Obj(value);
    renderer->target1Bd = GBARegisterBLDCNTGetTarget1Bd(value);
    renderer->target2Obj = GBARegisterBLDCNTGetTarget2Obj(value);
    renderer->target2Bd = GBARegisterBLDCNTGetTarget2Bd(value);

    if (oldEffect != renderer->blendEffect) {
        renderer->blendDirty = true;
    }
}

static void _writeWindowH(PnWindowN *win, uint16_t value) {
    win->h.end = (uint8_t) value;
    win->h.start = (uint8_t) (value >> 8);
    if (win->h.start > GBA_VIDEO_HORIZONTAL_PIXELS && win->h.start > win->h.end) {
        win->h.start = 0;
    }
    if (win->h.end > GBA_VIDEO_HORIZONTAL_PIXELS) {
        win->h.end = GBA_VIDEO_HORIZONTAL_PIXELS;
        if (win->h.start > GBA_VIDEO_HORIZONTAL_PIXELS) {
            win->h.start = GBA_VIDEO_HORIZONTAL_PIXELS;
        }
    }
}

static void _writeWindowV(PnWindowN *win, uint16_t value) {
    win->v.end = (uint8_t) value;
    win->v.start = (uint8_t) (value >> 8);
    if (win->v.start > GBA_VIDEO_VERTICAL_PIXELS && win->v.start > win->v.end) {
        win->v.start = 0;
    }
    if (win->v.end > GBA_VIDEO_VERTICAL_PIXELS) {
        win->v.end = GBA_VIDEO_VERTICAL_PIXELS;
        if (win->v.start > GBA_VIDEO_VERTICAL_PIXELS) {
            win->v.start = GBA_VIDEO_VERTICAL_PIXELS;
        }
    }
}

uint16_t pn_ppu_write_video_register(PnPpu *renderer, uint32_t address, uint16_t value) {
    switch (address) {
    case REG_DISPCNT:
        value &= 0xFFF7;
        renderer->dispcnt = value;
        _updateDISPCNT(renderer);
        break;
    case REG_GREENSWP:
        renderer->greenswap = value & 1;
        break;
    case REG_BG0CNT:
        value &= 0xDFFF;
        _writeBGCNT(renderer, &renderer->bg[0], value);
        break;
    case REG_BG1CNT:
        value &= 0xDFFF;
        _writeBGCNT(renderer, &renderer->bg[1], value);
        break;
    case REG_BG2CNT:
        _writeBGCNT(renderer, &renderer->bg[2], value);
        break;
    case REG_BG3CNT:
        _writeBGCNT(renderer, &renderer->bg[3], value);
        break;
    case REG_BG0HOFS:
        value &= 0x01FF;
        renderer->bg[0].x = value;
        break;
    case REG_BG0VOFS:
        value &= 0x01FF;
        renderer->bg[0].y = value;
        break;
    case REG_BG1HOFS:
        value &= 0x01FF;
        renderer->bg[1].x = value;
        break;
    case REG_BG1VOFS:
        value &= 0x01FF;
        renderer->bg[1].y = value;
        break;
    case REG_BG2HOFS:
        value &= 0x01FF;
        renderer->bg[2].x = value;
        break;
    case REG_BG2VOFS:
        value &= 0x01FF;
        renderer->bg[2].y = value;
        break;
    case REG_BG3HOFS:
        value &= 0x01FF;
        renderer->bg[3].x = value;
        break;
    case REG_BG3VOFS:
        value &= 0x01FF;
        renderer->bg[3].y = value;
        break;
    case REG_BG2PA:
        renderer->bg[2].dx = (int16_t) value;
        break;
    case REG_BG2PB:
        renderer->bg[2].dmx = (int16_t) value;
        break;
    case REG_BG2PC:
        renderer->bg[2].dy = (int16_t) value;
        break;
    case REG_BG2PD:
        renderer->bg[2].dmy = (int16_t) value;
        break;
    case REG_BG2X_LO:
        _writeBGX_LO(&renderer->bg[2], value);
        break;
    case REG_BG2X_HI:
        _writeBGX_HI(&renderer->bg[2], value);
        break;
    case REG_BG2Y_LO:
        _writeBGY_LO(&renderer->bg[2], value);
        break;
    case REG_BG2Y_HI:
        _writeBGY_HI(&renderer->bg[2], value);
        break;
    case REG_BG3PA:
        renderer->bg[3].dx = (int16_t) value;
        break;
    case REG_BG3PB:
        renderer->bg[3].dmx = (int16_t) value;
        break;
    case REG_BG3PC:
        renderer->bg[3].dy = (int16_t) value;
        break;
    case REG_BG3PD:
        renderer->bg[3].dmy = (int16_t) value;
        break;
    case REG_BG3X_LO:
        _writeBGX_LO(&renderer->bg[3], value);
        break;
    case REG_BG3X_HI:
        _writeBGX_HI(&renderer->bg[3], value);
        break;
    case REG_BG3Y_LO:
        _writeBGY_LO(&renderer->bg[3], value);
        break;
    case REG_BG3Y_HI:
        _writeBGY_HI(&renderer->bg[3], value);
        break;
    case REG_BLDCNT:
        _writeBLDCNT(renderer, value);
        value &= 0x3FFF;
        break;
    case REG_BLDALPHA:
        renderer->blda = value & 0x1F;
        if (renderer->blda > 0x10) {
            renderer->blda = 0x10;
        }
        renderer->bldb = (value >> 8) & 0x1F;
        if (renderer->bldb > 0x10) {
            renderer->bldb = 0x10;
        }
        value &= 0x1F1F;
        break;
    case REG_BLDY:
        value &= 0x1F;
        if (value > 0x10) {
            value = 0x10;
        }
        if (renderer->bldy != value) {
            renderer->bldy = value;
            renderer->blendDirty = true;
        }
        break;
    case REG_WIN0H:
        _writeWindowH(&renderer->winN[0], value);
        break;
    case REG_WIN1H:
        _writeWindowH(&renderer->winN[1], value);
        break;
    case REG_WIN0V:
        _writeWindowV(&renderer->winN[0], value);
        break;
    case REG_WIN1V:
        _writeWindowV(&renderer->winN[1], value);
        break;
    case REG_WININ:
        value &= 0x3F3F;
        renderer->winN[0].control.packed = (uint8_t) value;
        renderer->winN[1].control.packed = (uint8_t) (value >> 8);
        break;
    case REG_WINOUT:
        value &= 0x3F3F;
        renderer->winout.packed = (uint8_t) value;
        renderer->objwin.packed = (uint8_t) (value >> 8);
        break;
    case REG_MOSAIC:
        renderer->mosaic = value;
        break;
    default:
        /* mGBA logs "Invalid video register" and keeps the value. */
        break;
    }
    return value;
}

void pn_ppu_write_palette(PnPpu *renderer, uint32_t address, uint16_t value) {
    uint32_t color = mColorFrom555(value);
    renderer->normalPalette[address >> 1] = color;
    if (renderer->blendEffect == BLEND_BRIGHTEN) {
        renderer->variantPalette[address >> 1] = _brighten(color, renderer->bldy);
    } else if (renderer->blendEffect == BLEND_DARKEN) {
        renderer->variantPalette[address >> 1] = _darken(color, renderer->bldy);
    }
}

void pn_ppu_vram_changed(PnPpu *renderer, const uint8_t *vram) {
    renderer->vram = vram;
    renderer->bg[0].yCache = -1;
    renderer->bg[1].yCache = -1;
    renderer->bg[2].yCache = -1;
    renderer->bg[3].yCache = -1;
}

void pn_ppu_oam_changed(PnPpu *renderer, const uint8_t *oam) {
    renderer->oam = oam;
    renderer->oamDirty = true;
}

void pn_ppu_reset(PnPpu *renderer, const uint8_t *vram, const uint8_t *oam) {
    int i;
    memset(renderer, 0, sizeof(*renderer));
    renderer->vram = vram;
    renderer->oam = oam;
    renderer->dispcnt = 0x0080;
    renderer->blendEffect = BLEND_NONE;
    renderer->blendDirty = true;
    renderer->winN[0].control.priority = 0;
    renderer->winN[1].control.priority = 1;
    renderer->objwin.priority = 2;
    renderer->winout.priority = 3;
    renderer->oamDirty = true;
    renderer->oamMax = 0;
    renderer->nextY = 0;
    renderer->spriteDirtyStart = 0;
    renderer->spriteDirtyEnd = GBA_VIDEO_HORIZONTAL_PIXELS;
    for (i = 0; i < 4; ++i) {
        PnBackground *bg = &renderer->bg[i];
        bg->index = (unsigned) i;
        bg->dx = 256;
        bg->dmy = 256;
        bg->yCache = -1;
    }
}

void pn_ppu_latch_frame_start(PnPpu *renderer) {
    /* At the end of a frame mGBA promotes every enabling BG to ENABLED_MAX, and DISPCNT writes
     * during VBlank (nextY == 0) take effect immediately: at line 0 a BG is fully on or off.
     * The affine reference points are latched from BGxX/BGxY (sx = refx). */
    int i;
    for (i = 0; i < 4; ++i) {
        renderer->bg[i].enabled = ((renderer->dispcnt >> (8 + i)) & 1) ? ENABLED_MAX : 0;
        renderer->bg[i].sx = renderer->bg[i].refx;
        renderer->bg[i].sy = renderer->bg[i].refy;
    }
}

/* ================================================================== windows, buffers (video-software.c) */

static void _breakWindowInner(PnPpu *renderer, PnWindowN *win) {
    int activeWindow;
    int startX = 0;
    if (win->h.end > 0) {
        for (activeWindow = 0; activeWindow < renderer->nWindows; ++activeWindow) {
            if (win->h.start < renderer->windows[activeWindow].endX) {
                /* Insert a window before the end of the active window */
                PnWindow oldWindow = renderer->windows[activeWindow];
                if (win->h.start > startX) {
                    /* And after the start of the active window */
                    int nextWindow = renderer->nWindows;
                    ++renderer->nWindows;
                    for (; nextWindow > activeWindow; --nextWindow) {
                        renderer->windows[nextWindow] = renderer->windows[nextWindow - 1];
                    }
                    renderer->windows[activeWindow].endX = win->h.start;
                    ++activeWindow;
                }
                renderer->windows[activeWindow].control = win->control;
                renderer->windows[activeWindow].endX = win->h.end;
                if (win->h.end >= oldWindow.endX) {
                    /* Trim off extra windows we've overwritten */
                    for (++activeWindow; renderer->nWindows > activeWindow + 1 &&
                                         win->h.end >= renderer->windows[activeWindow].endX;
                         ++activeWindow) {
                        renderer->windows[activeWindow] = renderer->windows[activeWindow + 1];
                        --renderer->nWindows;
                    }
                } else {
                    ++activeWindow;
                    int nextWindow = renderer->nWindows;
                    ++renderer->nWindows;
                    for (; nextWindow > activeWindow; --nextWindow) {
                        renderer->windows[nextWindow] = renderer->windows[nextWindow - 1];
                    }
                    renderer->windows[activeWindow] = oldWindow;
                }
                break;
            }
            startX = renderer->windows[activeWindow].endX;
        }
    }
}

static void _breakWindow(PnPpu *renderer, PnWindowN *win, int y) {
    if (win->v.end >= win->v.start) {
        if (y >= win->v.end) {
            return;
        }
        if (y < win->v.start) {
            return;
        }
    } else if (y >= win->v.end && y < win->v.start) {
        return;
    }
    if (win->h.end > GBA_VIDEO_HORIZONTAL_PIXELS || win->h.end < win->h.start) {
        PnWindowN splits[2] = {*win, *win};
        splits[0].h.start = 0;
        splits[1].h.end = GBA_VIDEO_HORIZONTAL_PIXELS;
        _breakWindowInner(renderer, &splits[0]);
        _breakWindowInner(renderer, &splits[1]);
    } else {
        _breakWindowInner(renderer, win);
    }
}

/* The per-line state pn_preprocess_buffer sets up besides the sprite-layer reset and the backdrop
 * fill; pn_ppu_skip_scanline runs it too for the lines it does not compose: the window segments of
 * line y (windows[0].control.priority carries over between lines), the BG enable states
 * (_updateDISPCNT with the line's nextY), the variant palette, forceTarget1. */
static void pn_prepare_line(PnPpu *renderer, int y) {
    renderer->windows[0].endX = GBA_VIDEO_HORIZONTAL_PIXELS;
    renderer->nWindows = 1;
    if (GBARegisterDISPCNTIsWin0Enable(renderer->dispcnt) || GBARegisterDISPCNTIsWin1Enable(renderer->dispcnt) ||
        GBARegisterDISPCNTIsObjwinEnable(renderer->dispcnt)) {
        renderer->windows[0].control = renderer->winout;
        if (GBARegisterDISPCNTIsWin1Enable(renderer->dispcnt)) {
            _breakWindow(renderer, &renderer->winN[1], y);
        }
        if (GBARegisterDISPCNTIsWin0Enable(renderer->dispcnt)) {
            _breakWindow(renderer, &renderer->winN[0], y);
        }
    } else {
        /* Only the packed byte changes; the priority keeps its previous value, as in mGBA. */
        renderer->windows[0].control.packed = 0xFF;
    }

    _updateDISPCNT(renderer);

    if (renderer->blendDirty) {
        _updatePalettes(renderer);
        renderer->blendDirty = false;
    }
    renderer->forceTarget1 = false;
}

static void pn_preprocess_buffer(PnPpu *renderer, int y) {
    int x;
    /* spriteLayer[0..239] = FLAG_UNWRITTEN, only where the previous line's sprites wrote. */
    {
        int fillEnd = renderer->spriteDirtyEnd < GBA_VIDEO_HORIZONTAL_PIXELS ? renderer->spriteDirtyEnd
                                                                             : GBA_VIDEO_HORIZONTAL_PIXELS;
        x = renderer->spriteDirtyStart;
#ifdef PN_HAVE_SSE2
        {
            const __m128i unwritten = _mm_set1_epi32((int) FLAG_UNWRITTEN);
            for (; x + 4 <= fillEnd; x += 4) {
                _mm_storeu_si128((__m128i *) &renderer->spriteLayer[x], unwritten);
            }
        }
#endif
        for (; x < fillEnd; ++x) {
            renderer->spriteLayer[x] = FLAG_UNWRITTEN;
        }
        renderer->spriteDirtyStart = GBA_VIDEO_HORIZONTAL_PIXELS;
        renderer->spriteDirtyEnd = 0;
    }

    pn_prepare_line(renderer, y);

    int w;
    x = 0;
    for (w = 0; w < renderer->nWindows; ++w) {
        uint32_t backdrop = FLAG_UNWRITTEN | FLAG_PRIORITY | FLAG_IS_BACKGROUND;
        if (!renderer->target1Bd || renderer->blendEffect == BLEND_NONE || renderer->blendEffect == BLEND_ALPHA ||
            !GBAWindowControlIsBlendEnable(renderer->windows[w].control.packed)) {
            backdrop |= renderer->normalPalette[0];
        } else {
            backdrop |= renderer->variantPalette[0];
        }
        int end = renderer->windows[w].endX;
        /* Kept exactly as mGBA: the alignment loop can run past `end` (into the next window
         * segment) when a segment boundary is not a multiple of 4; the next segment then starts
         * at the advanced x and does not overwrite those pixels. */
        for (; x & 3; ++x) {
            renderer->row[x] = backdrop;
        }
#ifdef PN_HAVE_SSE2
        {
            const __m128i b4 = _mm_set1_epi32((int) backdrop);
            for (; x < end - 7; x += 8) {
                _mm_storeu_si128((__m128i *) &renderer->row[x], b4);
                _mm_storeu_si128((__m128i *) &renderer->row[x + 4], b4);
            }
            for (; x < end - 3; x += 4) {
                _mm_storeu_si128((__m128i *) &renderer->row[x], b4);
            }
        }
#else
        for (; x < end - 3; x += 4) {
            renderer->row[x] = backdrop;
            renderer->row[x + 1] = backdrop;
            renderer->row[x + 2] = backdrop;
            renderer->row[x + 3] = backdrop;
        }
#endif
        for (; x < end; ++x) {
            renderer->row[x] = backdrop;
        }
    }
}

#ifdef PN_HAVE_AVX2
/* The alpha pass of pn_postprocess_buffer below on [x, end), 8 pixels at a time; returns where the
 * scalar loop continues. */
PN_AVX2_FN int pn_blend_backdrop_avx2(PnPpu *renderer, int x, int end, uint32_t backdrop) {
    const __m256i t1 = _mm256_set1_epi32(FLAG_TARGET_1);
    const __m256i vb = _mm256_set1_epi32((int) backdrop);
    const __m256i wb = _mm256_set1_epi16((short) renderer->bldb);
    const __m256i wa = _mm256_set1_epi16((short) renderer->blda);
    for (; x + 8 <= end; x += 8) {
        __m256i color = _mm256_loadu_si256((const __m256i *) &renderer->row[x]);
        __m256i m = _mm256_cmpeq_epi32(_mm256_and_si256(color, t1), t1);
        if (_mm256_testz_si256(m, m)) {
            continue;
        }
        __m256i mixed = pn_mix_avx2(wb, vb, wa, color);
        _mm256_storeu_si256((__m256i *) &renderer->row[x], _mm256_blendv_epi8(color, mixed, m));
    }
    return x;
}
#endif

static void pn_postprocess_buffer(PnPpu *renderer) {
    int x, w;
    if ((renderer->forceTarget1 || renderer->bg[0].target1 || renderer->bg[1].target1 || renderer->bg[2].target1 ||
         renderer->bg[3].target1) &&
        renderer->target2Bd) {
        x = 0;
        for (w = 0; w < renderer->nWindows; ++w) {
            uint32_t backdrop = 0;
            if (!renderer->target1Bd || renderer->blendEffect == BLEND_NONE || renderer->blendEffect == BLEND_ALPHA ||
                !GBAWindowControlIsBlendEnable(renderer->windows[w].control.packed)) {
                backdrop |= renderer->normalPalette[0];
            } else {
                backdrop |= renderer->variantPalette[0];
            }
            int end = renderer->windows[w].endX;
#ifdef PN_HAVE_AVX2
            if (renderer->simd == PN_PPU_SIMD_AVX2) {
                x = pn_blend_backdrop_avx2(renderer, x, end, backdrop);
            }
#endif
            for (; x < end; ++x) {
                uint32_t color = renderer->row[x];
                if (color & FLAG_TARGET_1) {
                    renderer->row[x] = mColorMix5Bit(renderer->bldb, backdrop, renderer->blda, color);
                }
            }
        }
    }
    if (renderer->forceTarget1 && (renderer->blendEffect == BLEND_DARKEN || renderer->blendEffect == BLEND_BRIGHTEN)) {
        x = 0;
        for (w = 0; w < renderer->nWindows; ++w) {
            int end = renderer->windows[w].endX;
            uint32_t mask = FLAG_REBLEND | FLAG_IS_BACKGROUND;
            uint32_t match = FLAG_REBLEND;
            bool objBlend = GBAWindowControlIsBlendEnable(renderer->objwin.packed);
            bool winBlend = GBAWindowControlIsBlendEnable(renderer->windows[w].control.packed);
            if (GBARegisterDISPCNTIsObjwinEnable(renderer->dispcnt) && objBlend != winBlend) {
                mask |= FLAG_OBJWIN;
                if (objBlend) {
                    match |= FLAG_OBJWIN;
                }
            } else if (!winBlend) {
                x = end;
                continue;
            }
            if (renderer->blendEffect == BLEND_DARKEN) {
                for (; x < end; ++x) {
                    uint32_t color = renderer->row[x];
                    if ((color & mask) == match) {
                        renderer->row[x] = _darken(color, renderer->bldy);
                    }
                }
            } else if (renderer->blendEffect == BLEND_BRIGHTEN) {
                for (; x < end; ++x) {
                    uint32_t color = renderer->row[x];
                    if ((color & mask) == match) {
                        renderer->row[x] = _brighten(color, renderer->bldy);
                    }
                }
            }
        }
    }
}

/* ================================================================== scanline (video-software.c DrawScanline) */

/* The BG flags of the current window segment (the segment loop of pn_ppu_draw_scanline). */
static void pn_update_segment_flags(PnPpu *renderer) {
    switch (GBARegisterDISPCNTGetMode(renderer->dispcnt)) {
    case 0:
        if (renderer->bg[0].enabled == ENABLED_MAX) {
            _updateFlags(renderer, &renderer->bg[0]);
        }
        if (renderer->bg[1].enabled == ENABLED_MAX) {
            _updateFlags(renderer, &renderer->bg[1]);
        }
        /* Fall through */
    case 2:
        if (renderer->bg[3].enabled == ENABLED_MAX) {
            _updateFlags(renderer, &renderer->bg[3]);
        }
        /* Fall through */
    case 3:
    case 4:
    case 5:
        if (renderer->bg[2].enabled == ENABLED_MAX) {
            _updateFlags(renderer, &renderer->bg[2]);
        }
        break;
    case 1:
        if (renderer->bg[0].enabled == ENABLED_MAX) {
            _updateFlags(renderer, &renderer->bg[0]);
        }
        if (renderer->bg[1].enabled == ENABLED_MAX) {
            _updateFlags(renderer, &renderer->bg[1]);
        }
        if (renderer->bg[2].enabled == ENABLED_MAX) {
            _updateFlags(renderer, &renderer->bg[2]);
        }
        break;
    }
}

/* The side effects at the end of a composed (not forced-blank) line: the affine reference points
 * advance (modes != 0, fully enabled BGs) and the BG enable latency counts down. */
static void pn_finish_line(PnPpu *renderer) {
    if (GBARegisterDISPCNTGetMode(renderer->dispcnt) != 0) {
        if (renderer->bg[2].enabled == ENABLED_MAX) {
            renderer->bg[2].sx += renderer->bg[2].dmx;
            renderer->bg[2].sy += renderer->bg[2].dmy;
        }
        if (renderer->bg[3].enabled == ENABLED_MAX) {
            renderer->bg[3].sx += renderer->bg[3].dmx;
            renderer->bg[3].sy += renderer->bg[3].dmy;
        }
    }

    int i;
    for (i = 0; i < 4; ++i) {
        if (renderer->bg[i].enabled != 0 && renderer->bg[i].enabled < ENABLED_MAX) {
            ++renderer->bg[i].enabled;
        }
    }
}

int pn_ppu_draw_scanline(PnPpu *renderer, int y) {
    if (y == GBA_VIDEO_VERTICAL_PIXELS - 1) {
        renderer->nextY = 0;
    } else {
        renderer->nextY = y + 1;
    }

    if (GBARegisterDISPCNTIsForcedBlank(renderer->dispcnt)) {
        int x;
        for (x = 0; x < GBA_VIDEO_HORIZONTAL_PIXELS; ++x) {
            renderer->row[x] = M_COLOR_WHITE;
        }
        return 0;
    }

    pn_preprocess_buffer(renderer, y);
    renderer->spriteCyclesRemaining =
        GBARegisterDISPCNTIsHblankIntervalFree(renderer->dispcnt) ? OBJ_HBLANK_FREE_LENGTH : OBJ_LENGTH;
    int spriteLayers = pn_preprocess_sprite_layer(renderer, y);

    int w;
    unsigned priority;
    renderer->end = 0;
    for (w = 0; w < renderer->nWindows; ++w) {
        renderer->start = renderer->end;
        renderer->end = renderer->windows[w].endX;
        renderer->currentWindow = renderer->windows[w].control;
        pn_update_segment_flags(renderer);

        /* TEST_LAYER_ENABLED(X) for all four priorities at once: nothing it reads (enable state,
         * priority, window controls, DISPCNT) changes while the segment is drawn. */
        unsigned layersAt[4] = {0, 0, 0, 0};
        {
            int bgIndex;
            for (bgIndex = 0; bgIndex < 4; ++bgIndex) {
                const PnBackground *bg = &renderer->bg[bgIndex];
                if (bg->enabled == ENABLED_MAX && bg->priority < 4 &&
                    (((renderer->currentWindow.packed >> bgIndex) & 1) ||
                     (GBARegisterDISPCNTIsObjwinEnable(renderer->dispcnt) && ((renderer->objwin.packed >> bgIndex) & 1)))) {
                    layersAt[bg->priority] |= 1u << bgIndex;
                }
            }
        }
        for (priority = 0; priority < 4; ++priority) {
            if (spriteLayers & (1 << priority)) {
                pn_postprocess_sprite(renderer, priority);
            }
            if (!layersAt[priority]) {
                continue;
            }
            if ((layersAt[priority] & 1) && GBARegisterDISPCNTGetMode(renderer->dispcnt) < 2) {
                pn_draw_background_mode0(renderer, &renderer->bg[0], y);
            }
            if ((layersAt[priority] & 2) && GBARegisterDISPCNTGetMode(renderer->dispcnt) < 2) {
                pn_draw_background_mode0(renderer, &renderer->bg[1], y);
            }
            if (layersAt[priority] & 4) {
                switch (GBARegisterDISPCNTGetMode(renderer->dispcnt)) {
                case 0:
                    pn_draw_background_mode0(renderer, &renderer->bg[2], y);
                    break;
                case 1:
                case 2:
                    pn_draw_background_mode2(renderer, &renderer->bg[2], y);
                    break;
                case 3:
                    pn_draw_background_mode3(renderer, &renderer->bg[2], y);
                    break;
                case 4:
                    pn_draw_background_mode4(renderer, &renderer->bg[2], y);
                    break;
                case 5:
                    pn_draw_background_mode5(renderer, &renderer->bg[2], y);
                    break;
                }
            }
            if (layersAt[priority] & 8) {
                switch (GBARegisterDISPCNTGetMode(renderer->dispcnt)) {
                case 0:
                    pn_draw_background_mode0(renderer, &renderer->bg[3], y);
                    break;
                case 2:
                    pn_draw_background_mode2(renderer, &renderer->bg[3], y);
                    break;
                }
            }
        }
    }

    pn_postprocess_buffer(renderer);
    pn_finish_line(renderer);
    return 1;
}

/* Line y without composing it (pn_render_ex's sub-sampled gray formats): every effect
 * pn_ppu_draw_scanline(renderer, y) has on the state that later lines read, nothing else.
 *   - nextY (read by DISPCNT writes: BG enable latency);
 *   - forced blank: nothing more (the composed path returns early too);
 *   - pn_prepare_line: window segments (windows[0].control.priority carries over), _updateDISPCNT,
 *     the variant palette, forceTarget1;
 *   - per window segment: currentWindow and the BG flags (_updateFlags also runs on BGxCNT writes
 *     with the current window, so it is left as the composed line leaves it);
 *   - pn_finish_line: affine reference points, BG enable latency.
 * Not repeated, because the next composed line recomputes them from scratch before reading them:
 * row, the sprite layer and its dirty range (only sprites write it; the reset before the next
 * composed line covers what the last composed line wrote), the sprite list (rebuilt from OAM on
 * demand when oamDirty), spriteCyclesRemaining, start/end, the per-BG map-row caches (keyed on the
 * map row, invalidated by VRAM / BGCNT changes). Returns what pn_ppu_draw_scanline would. */
int pn_ppu_skip_scanline(PnPpu *renderer, int y) {
    if (y == GBA_VIDEO_VERTICAL_PIXELS - 1) {
        renderer->nextY = 0;
    } else {
        renderer->nextY = y + 1;
    }
    if (GBARegisterDISPCNTIsForcedBlank(renderer->dispcnt)) {
        return 0;
    }
    pn_prepare_line(renderer, y);
    int w;
    for (w = 0; w < renderer->nWindows; ++w) {
        renderer->currentWindow = renderer->windows[w].control;
        pn_update_segment_flags(renderer);
    }
    pn_finish_line(renderer);
    return 1;
}
