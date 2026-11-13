/* pn_ppu.h -- scanline compositor state for fireredsim's renderer (internal header).
 *
 * Derived from mGBA 0.10.5 (commit 26b7884bc25a5933960f3cdcd98bac1ae14d42e2):
 * include/mgba/internal/gba/renderers/video-software.h and src/gba/renderers/software-private.h,
 * Copyright (c) 2013-2015 Jeffrey Pfau. Modified for fireredsim (render/THIRD_PARTY.md):
 * state is rebuilt from a register/memory snapshot per frame instead of being fed by bus writes;
 * scanline caching, highlighting, debug layer toggles and cache-set hooks removed; 32-bit color_t
 * (0x00BBGGRR) only.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#ifndef PN_PPU_INTERNAL_H
#define PN_PPU_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>

#define PN_PPU_W 240
#define PN_PPU_H 160

/* Window list capacity. mGBA uses 5 (MAX_WINDOW), which its _breakWindowInner never exceeds
 * (WINOUT + at most two segments per window); a little headroom costs nothing. */
#define PN_MAX_WINDOW 8

enum PnBlendEffect { BLEND_NONE = 0, BLEND_ALPHA = 1, BLEND_BRIGHTEN = 2, BLEND_DARKEN = 3 };

typedef struct PnWindowControl {
    uint8_t packed; /* WININ/WINOUT byte: bit0-3 BG0-3, bit4 OBJ, bit5 colour effect */
    int8_t priority; /* 0 WIN0, 1 WIN1, 2 OBJWIN, 3 WINOUT */
} PnWindowControl;

typedef struct PnWindowRegion {
    uint8_t end;
    uint8_t start;
} PnWindowRegion;

typedef struct PnWindowN {
    PnWindowRegion h;
    PnWindowRegion v;
    PnWindowControl control;
} PnWindowN;

typedef struct PnWindow {
    uint8_t endX;
    PnWindowControl control;
} PnWindow;

typedef struct PnBackground {
    unsigned index;
    int enabled; /* 0 off, 1..3 enabling (mGBA's 3-line latency), 4 = ENABLED_MAX, -2 off (fast re-enable) */
    unsigned priority;
    uint32_t charBase;
    int mosaic;
    int multipalette;
    uint32_t screenBase;
    int overflow;
    int size;
    int target1;
    int target2;
    uint16_t x;
    uint16_t y;
    int32_t refx;
    int32_t refy;
    int16_t dx;
    int16_t dmx;
    int16_t dy;
    int16_t dmy;
    int32_t sx;
    int32_t sy;
    int yCache;
    uint16_t mapCache[64];
    uint32_t flags;
    uint32_t objwinFlags;
    bool variant;
} PnBackground;

typedef struct PnObj {
    uint16_t a, b, c, d;
} PnObj;

typedef struct PnSprite {
    PnObj obj;
    int16_t y;
    int16_t endY;
    int16_t cycles;
    int8_t index;
} PnSprite;

typedef struct PnPpu {
    const uint8_t *vram; /* 96 KiB */
    const uint8_t *oam;  /* 1 KiB */

    uint16_t dispcnt;

    uint32_t row[PN_PPU_W];
    uint32_t spriteLayer[PN_PPU_W];
    int32_t spriteCyclesRemaining;

    /* BLDCNT */
    unsigned target1Obj;
    unsigned target1Bd;
    unsigned target2Obj;
    unsigned target2Bd;
    bool blendDirty;
    int blendEffect;
    uint32_t normalPalette[512];
    uint32_t variantPalette[512];

    uint16_t blda;
    uint16_t bldb;
    uint16_t bldy;

    uint16_t mosaic;
    bool greenswap;

    PnWindowN winN[2];
    PnWindowControl winout;
    PnWindowControl objwin;
    PnWindowControl currentWindow;

    int nWindows;
    PnWindow windows[PN_MAX_WINDOW];

    PnBackground bg[4];

    bool forceTarget1;
    bool oamDirty;
    int oamMax;
    PnSprite sprites[128];

    int nextY;
    int start;
    int end;

    /* fireredsim additions (after every mGBA field: the fuzz-only OBJ-mosaic overrun of
     * spriteLayer must hit the same fields as in mGBA). */
    int simd; /* PN_PPU_SIMD_*: which kernels draw (all give identical pixels) */
    /* spriteLayer[x] == FLAG_UNWRITTEN for every x in [0, 240) outside [spriteDirtyStart,
     * spriteDirtyEnd): the per-line reset only refills what the sprite pass wrote. */
    int spriteDirtyStart;
    int spriteDirtyEnd;
    /* bg[i].mapCache twice in a row (rebuilt together with it): any run of up to 64 consecutive
     * map entries, wrapped at 64 like mapCache[(localX >> 3) & 0x3F], is contiguous here. */
    uint16_t mapRow[4][128];
    /* Per visible line, bit i: sprites[i] covers the line (rebuilt with the sprite list). */
    uint32_t spriteLines[PN_PPU_H][4];
    /* Scratch of the text-BG kernel (one run of at most 32 tiles; see pn_bg_text_tiles_avx2). */
    uint32_t tileInfo[3 * 32];
} PnPpu;

enum { PN_PPU_SIMD_NONE = 0, PN_PPU_SIMD_AVX2 = 1 };

/* 1 if the AVX2 kernels are compiled in and this CPU (and OS) can run them. */
int pn_ppu_avx2_usable(void);

/* Power-on state of mGBA's renderer (GBAVideoSoftwareRendererReset). */
void pn_ppu_reset(PnPpu *renderer, const uint8_t *vram, const uint8_t *oam);
/* A write to a video register (0x00-0x56 except DISPSTAT/VCOUNT); returns the value the I/O
 * backing store keeps (mGBA's writeVideoRegister). Uses renderer->nextY like mGBA. */
uint16_t pn_ppu_write_video_register(PnPpu *renderer, uint32_t address, uint16_t value);
/* A 16-bit write to palette RAM at byte offset `address` (0..0x3FE). */
void pn_ppu_write_palette(PnPpu *renderer, uint32_t address, uint16_t value);
/* VRAM / OAM contents changed (or their backing pointer moved). */
void pn_ppu_vram_changed(PnPpu *renderer, const uint8_t *vram);
void pn_ppu_oam_changed(PnPpu *renderer, const uint8_t *oam);
/* Sets every BG's enable state as if DISPCNT had been written during VBlank. */
void pn_ppu_latch_frame_start(PnPpu *renderer);
/* Composes scanline y into renderer->row (0x00BBGGRR | flags). Returns 1 for a normal line,
 * 0 for a forced-blank line (row = white, green swap must not be applied). */
int pn_ppu_draw_scanline(PnPpu *renderer, int y);
/* Line y without composing it: every per-line side effect of pn_ppu_draw_scanline that later lines
 * depend on (nextY, windows, BG enable latency, affine reference points, ...), renderer->row
 * untouched. Same return value as pn_ppu_draw_scanline. */
int pn_ppu_skip_scanline(PnPpu *renderer, int y);

#endif /* PN_PPU_INTERNAL_H */
