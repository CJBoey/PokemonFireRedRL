// fireredsim platform layer: the BIOS calls used by FireRed (libagbsyscall.s).
//
// Derived from mGBA 0.10.5's HLE BIOS, src/gba/bios.c (C routines) and
// src/gba/hle-bios.s (CpuSet/CpuFastSet), Copyright (c) 2013-2015 Jeffrey Pfau:
// what the mGBA oracle runs without an official BIOS image. Each routine below
// mirrors that code, including its corner cases (alignment, 20-bit counts,
// division by zero, LZ77 overrun, float affine math), so that RAM after a BIOS
// call is bit-identical to the oracle's. Cycle counts (biosStall) are not
// modelled. Modifications for fireredsim: the routines read and write game
// memory directly instead of through the emulated CPU, and the BIOS open bus
// is tracked for the fault emulation (memfault.c).
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#include "port_internal.h"
#include "port_fault.h"
#include "bios_affine_trig.h"

// The BIOS open bus (mGBA: memory.biosPrefetch): the opcode the CPU had
// prefetched when it last left the BIOS. 0 at power-on (mGBA skips the BIOS
// boot), 0xE3A02004 after any BIOS call (HLE SWIs end there), 0xE25EF004 while
// an interrupt handler runs and 0xE55EC002 after it returned (platform/frame.c).
u32 gPortBiosPrefetch;
#define BIOS_CALL_DONE() (gPortBiosPrefetch = 0xE3A02004)

// Reads requested explicitly by PORTABLE patches (game code that reads through
// NULL and is patched instead of relying on the generic fault emulation in
// memfault.c). Addresses >= 0x4000 are unmapped: the CPU open bus there is the
// prefetched game opcode, which native code cannot reproduce.
static u32 BiosOpenBus(unsigned int addr, void *caller)
{
    if (addr >= 0x4000)
        Port_Inexact("read from unmapped GBA address 0x%08X: CPU open bus (the prefetched game opcode), "
                     "not reproducible natively", addr);
    Port_FaultNoteExplicit(PN_SITE_EXPLICIT_BIOS, addr, gPortBiosPrefetch, caller);
    return gPortBiosPrefetch;
}

unsigned int Port_BiosOpenBus32(unsigned int addr)
{
    return BiosOpenBus(addr, __builtin_return_address(0));
}

unsigned short Port_BiosOpenBus16(unsigned int addr)
{
    return (BiosOpenBus(addr, __builtin_return_address(0)) >> ((addr & 2) * 8)) & 0xFFFF;
}

unsigned char Port_BiosOpenBus8(unsigned int addr)
{
    return (BiosOpenBus(addr, __builtin_return_address(0)) >> ((addr & 3) * 8)) & 0xFF;
}

// ---- memory access helpers --------------------------------------------------
// Native addresses are plain host memory. The BIOS routines access memory
// through these helpers so that the GBA bus rules that differ from plain memory
// (8-bit stores to video memory, rotated unaligned loads) are applied.

static inline bool32 InRange(u32 addr, u32 base, u32 size)
{
    return addr - base < size;
}

static inline u32 Load32(u32 addr)
{
    // ARM7 LDR from an unaligned address returns the aligned word rotated.
    u32 v = *(const u32 *)(addr & ~3u);
    u32 rot = (addr & 3) * 8;
    return rot ? (v >> rot) | (v << (32 - rot)) : v;
}

static inline u32 Load16(u32 addr)
{
    // LDRH from an odd address: aligned halfword rotated right by 8 (32-bit).
    u32 v = *(const u16 *)(addr & ~1u);
    return (addr & 1) ? (v >> 8) | (v << 24) : v;
}

static inline u8 Load8(u32 addr)
{
    return *(const u8 *)addr;
}

static inline void Store32(u32 addr, u32 v)
{
    *(u32 *)(addr & ~3u) = v;
}

static inline void Store16(u32 addr, u32 v)
{
    *(u16 *)(addr & ~1u) = (u16)v;
}

static inline void Store8(u32 addr, u32 v)
{
    v &= 0xFF;
    if (InRange(addr, VRAM, VRAM_SIZE))
    {
        // GBAStore8: byte writes to BG VRAM write the byte to both halves of
        // the halfword; writes to OBJ VRAM are ignored.
        u32 off = addr - VRAM;
        u32 objStart = ((PORT_IO16(REG_OFFSET_DISPCNT) & 7) >= 3) ? 0x14000 : 0x10000;
        if (off < objStart)
            *(u16 *)(addr & ~1u) = (u16)(v | (v << 8));
        return;
    }
    if (InRange(addr, OAM, OAM_SIZE))
        return; // byte writes to OAM are ignored
    if (InRange(addr, PLTT, PLTT_SIZE))
    {
        *(u16 *)(addr & ~1u) = (u16)(v | (v << 8));
        return;
    }
    *(u8 *)addr = (u8)v;
}

// ---- CpuSet / CpuFastSet (hle-bios.s) ---------------------------------------

static void CpuSet_Impl(const void *src, void *dest, u32 control)
{
    u32 s = (u32)src;
    u32 d = (u32)dest;
    u32 count = control & 0xFFFFF; // mov r4, r2, lsl #12: 20-bit unit count
    u32 i;

    if (s < 0x02000000)
        return; // "Cannot CpuSet from BIOS": mGBA skips the call

    if (control & CPU_SET_SRC_FIXED)
    {
        if (control & CPU_SET_32BIT)
        {
            u32 v = Load32(s & ~3u); // ldmia r0!, {r3}
            u32 *p = (u32 *)(d & ~3u);
            for (i = 0; i < count; i++)
                p[i] = v;
        }
        else
        {
            u16 v = (u16)Load16(s & ~1u);
            u16 *p = (u16 *)(d & ~1u);
            for (i = 0; i < count; i++)
                p[i] = v;
        }
    }
    else if (control & CPU_SET_32BIT)
    {
        // ldmia/stmia: word-aligned accesses, forward order.
        const u32 *sp = (const u32 *)(s & ~3u);
        u32 *dp = (u32 *)(d & ~3u);
        if ((u32)dp > (u32)sp && (u32)dp < (u32)sp + count * 4)
        {
            for (i = 0; i < count; i++)
                dp[i] = sp[i];
        }
        else
        {
            memmove(dp, sp, count * 4);
        }
    }
    else
    {
        // ldrh/strh without the alignment the fill path applies.
        if (!(s & 1) && !(d & 1) && !(d > s && d < s + count * 2))
        {
            memmove((void *)d, (const void *)s, count * 2);
        }
        else
        {
            for (i = 0; i < count; i++)
                Store16(d + 2 * i, Load16(s + 2 * i));
        }
    }
}

void CpuSet(const void *src, void *dest, u32 control)
{
    CpuSet_Impl(src, dest, control);
    BIOS_CALL_DONE();
}

static void CpuFastSet_Impl(const void *src, void *dest, u32 control)
{
    u32 s = (u32)src;
    u32 d = (u32)dest;
    u32 count = control & 0xFFFFF;
    u32 blocks = (count + 7) / 8; // stmia of 8 words while r1 < dest + count*4
    u32 i;

    if (s < 0x02000000)
        return;

    if (control & CPU_FAST_SET_SRC_FIXED)
    {
        u32 v = Load32(s); // ldr r3, [r0]
        u32 *p = (u32 *)(d & ~3u);
        for (i = 0; i < blocks * 8; i++)
            p[i] = v;
    }
    else
    {
        const u32 *sp = (const u32 *)(s & ~3u);
        u32 *dp = (u32 *)(d & ~3u);
        u32 n = blocks * 8;
        if ((u32)dp > (u32)sp && (u32)dp < (u32)sp + n * 4)
        {
            // Forward copy in blocks of 8 words (ldmia then stmia).
            for (i = 0; i < n; i += 8)
            {
                u32 tmp[8];
                memcpy(tmp, sp + i, sizeof(tmp));
                memcpy(dp + i, tmp, sizeof(tmp));
            }
        }
        else
        {
            memmove(dp, sp, n * 4);
        }
    }
}

void CpuFastSet(const void *src, void *dest, u32 control)
{
    CpuFastSet_Impl(src, dest, control);
    BIOS_CALL_DONE();
}

// ---- decompression (_unLz77, _unRl) ------------------------------------------

static void UnLz77(u32 source, u32 dest, int width)
{
    int remaining = (Load32(source) & 0xFFFFFF00) >> 8;
    int blockheader = 0;
    int blocksRemaining = 0;
    u32 disp;
    int bytes;
    int byte;
    int halfword = 0;

    if (!(source & 0x0E000000))
        return; // "Bad LZ77 source"
    source += 4;
    while (remaining > 0)
    {
        if (blocksRemaining)
        {
            if (blockheader & 0x80)
            {
                // Compressed
                int block = Load8(source + 1) | (Load8(source) << 8);
                source += 2;
                disp = dest - (block & 0x0FFF) - 1;
                bytes = (block >> 12) + 3;
                while (bytes--)
                {
                    if (remaining)
                        --remaining;
                    // else: improperly compressed data; mGBA (without
                    // vbaBugCompat) keeps copying past the end like hardware.
                    if (width == 2)
                    {
                        byte = (s16)Load16(disp & ~1u);
                        if (dest & 1)
                        {
                            byte >>= (disp & 1) * 8;
                            halfword |= byte << 8;
                            Store16(dest ^ 1, halfword);
                        }
                        else
                        {
                            byte >>= (disp & 1) * 8;
                            halfword = byte & 0xFF;
                        }
                    }
                    else
                    {
                        byte = Load8(disp);
                        Store8(dest, byte);
                    }
                    ++disp;
                    ++dest;
                }
            }
            else
            {
                // Uncompressed
                byte = Load8(source);
                ++source;
                if (width == 2)
                {
                    if (dest & 1)
                    {
                        halfword |= byte << 8;
                        Store16(dest ^ 1, halfword);
                    }
                    else
                    {
                        halfword = byte;
                    }
                }
                else
                {
                    Store8(dest, byte);
                }
                ++dest;
                --remaining;
            }
            blockheader <<= 1;
            --blocksRemaining;
        }
        else
        {
            blockheader = Load8(source);
            ++source;
            blocksRemaining = 8;
        }
    }
}

static void LZ77UnCompWram_Impl(const void *src, void *dest)
{
    UnLz77((u32)src, (u32)dest, 1);
}

void LZ77UnCompWram(const void *src, void *dest)
{
    LZ77UnCompWram_Impl(src, dest);
    BIOS_CALL_DONE();
}

static void LZ77UnCompVram_Impl(const void *src, void *dest)
{
    UnLz77((u32)src, (u32)dest, 2);
}

void LZ77UnCompVram(const void *src, void *dest)
{
    LZ77UnCompVram_Impl(src, dest);
    BIOS_CALL_DONE();
}

static void UnRl(u32 source, u32 dest, int width)
{
    int remaining;
    int padding;
    int blockheader;
    int block;
    int halfword = 0;

    if (!(source & 0x0E000000))
        return; // "Bad RL source"
    remaining = (Load32(source & ~3u) & 0xFFFFFF00) >> 8;
    padding = (4 - remaining) & 0x3;
    source += 4;
    while (remaining > 0)
    {
        blockheader = Load8(source);
        ++source;
        if (blockheader & 0x80)
        {
            // Compressed
            blockheader &= 0x7F;
            blockheader += 3;
            block = Load8(source);
            ++source;
            while (blockheader-- && remaining)
            {
                --remaining;
                if (width == 2)
                {
                    if (dest & 1)
                    {
                        halfword |= block << 8;
                        Store16(dest ^ 1, halfword);
                    }
                    else
                    {
                        halfword = block;
                    }
                }
                else
                {
                    Store8(dest, block);
                }
                ++dest;
            }
        }
        else
        {
            // Uncompressed
            blockheader++;
            while (blockheader-- && remaining)
            {
                int byte;
                --remaining;
                byte = Load8(source);
                ++source;
                if (width == 2)
                {
                    if (dest & 1)
                    {
                        halfword |= byte << 8;
                        Store16(dest ^ 1, halfword);
                    }
                    else
                    {
                        halfword = byte;
                    }
                }
                else
                {
                    Store8(dest, byte);
                }
                ++dest;
            }
        }
    }
    if (width == 2)
    {
        if (dest & 1)
        {
            --padding;
            ++dest;
        }
        for (; padding > 0; padding -= 2, dest += 2)
            Store16(dest, 0);
    }
    else
    {
        while (padding--)
        {
            Store8(dest, 0);
            ++dest;
        }
    }
}

static void RLUnCompWram_Impl(const void *src, void *dest)
{
    UnRl((u32)src, (u32)dest, 1);
}

void RLUnCompWram(const void *src, void *dest)
{
    RLUnCompWram_Impl(src, dest);
    BIOS_CALL_DONE();
}

static void RLUnCompVram_Impl(const void *src, void *dest)
{
    UnRl((u32)src, (u32)dest, 2);
}

void RLUnCompVram(const void *src, void *dest)
{
    RLUnCompVram_Impl(src, dest);
    BIOS_CALL_DONE();
}

// ---- arithmetic (_Div, _Sqrt, _ArcTan2) ----------------------------------------

s32 Div(s32 num, s32 denom)
{
    BIOS_CALL_DONE();
    if (denom == 0)
    {
        // The real BIOS hangs for |num| > 1; mGBA returns +-1. Either way the
        // retail game must never get here: report it.
        if (num != 0 && num != 1 && num != -1)
            Port_Hang("BIOS Div(%d, 0): the real BIOS loops forever (mGBA's HLE returns +-1)", (int)num);
        return (num < 0) ? -1 : 1;
    }
    if (denom == -1 && num == INT32_MIN)
        return INT32_MIN;
    return num / denom; // C division truncates toward zero, as div() does
}

u16 Sqrt(u32 x)
{
    BIOS_CALL_DONE();
    u32 lower;
    u32 upper = x;
    u32 bound = 1;

    if (!x)
        return 0;
    while (bound < upper)
    {
        upper >>= 1;
        bound <<= 1;
    }
    while (TRUE)
    {
        u32 accum = 0;
        u32 oldBound;

        upper = x;
        lower = bound;
        while (TRUE)
        {
            u32 oldLower = lower;
            if (lower <= upper >> 1)
                lower <<= 1;
            if (oldLower >= upper >> 1)
                break;
        }
        while (TRUE)
        {
            accum <<= 1;
            if (upper >= lower)
            {
                ++accum;
                upper -= lower;
            }
            if (lower == bound)
                break;
            lower >>= 1;
        }
        oldBound = bound;
        bound += accum;
        bound >>= 1;
        if (bound >= oldBound)
        {
            bound = oldBound;
            break;
        }
    }
    return (u16)bound;
}

static s16 BiosArcTan(s32 i)
{
    s32 a = -((i * i) >> 14);
    s32 b = ((0xA9 * a) >> 14) + 0x390;
    b = ((b * a) >> 14) + 0x91C;
    b = ((b * a) >> 14) + 0xFB6;
    b = ((b * a) >> 14) + 0x16AA;
    b = ((b * a) >> 14) + 0x2081;
    b = ((b * a) >> 14) + 0x3651;
    b = ((b * a) >> 14) + 0xA2F9;
    return (s16)((i * b) >> 16);
}

static s16 BiosArcTan2(s32 x, s32 y)
{
    if (!y)
        return (x >= 0) ? 0 : (s16)0x8000;
    if (!x)
        return (y >= 0) ? 0x4000 : (s16)0xC000;
    if (y >= 0)
    {
        if (x >= 0)
        {
            if (x >= y)
                return BiosArcTan((y << 14) / x);
        }
        else if (-x >= y)
        {
            return (s16)(BiosArcTan((y << 14) / x) + 0x8000);
        }
        return (s16)(0x4000 - BiosArcTan((x << 14) / y));
    }
    else
    {
        if (x <= 0)
        {
            if (-x > -y)
                return (s16)(BiosArcTan((y << 14) / x) + 0x8000);
        }
        else if (x >= -y)
        {
            return (s16)(BiosArcTan((y << 14) / x) + 0x10000);
        }
        return (s16)(0xC000 - BiosArcTan((x << 14) / y));
    }
}

u16 ArcTan2(s16 x, s16 y)
{
    BIOS_CALL_DONE();
    return (u16)BiosArcTan2(x, y);
}

// ---- affine matrices (_BgAffineSet, _ObjAffineSet: single-precision floats) ---

static inline float FloatFromBits(u32 b)
{
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static void BgAffineSet_Impl(struct BgAffineSrcData *src, struct BgAffineDstData *dest, s32 count)
{
    u32 offset = (u32)src;
    u32 destination = (u32)dest;
    s32 i = count;

    while (i--)
    {
        float ox = (s32)Load32(offset) / 256.f;
        float oy = (s32)Load32(offset + 4) / 256.f;
        float cx = (s16)Load16(offset + 8);
        float cy = (s16)Load16(offset + 10);
        float sx = (s16)Load16(offset + 12) / 256.f;
        float sy = (s16)Load16(offset + 14) / 256.f;
        u32 k = (Load16(offset + 16) >> 8) & 0xFF;
        float a, b, c, d, rx, ry;

        offset += 20;
        a = d = FloatFromBits(sAffineCos[k]);
        b = c = FloatFromBits(sAffineSin[k]);
        a *= sx;
        b *= -sx;
        c *= sy;
        d *= sy;
        rx = ox - (a * cx + b * cy);
        ry = oy - (c * cx + d * cy);
        Store16(destination, (s16)(a * 256));
        Store16(destination + 2, (s16)(b * 256));
        Store16(destination + 4, (s16)(c * 256));
        Store16(destination + 6, (s16)(d * 256));
        Store32(destination + 8, (s32)(rx * 256));
        Store32(destination + 12, (s32)(ry * 256));
        destination += 16;
    }
}

void BgAffineSet(struct BgAffineSrcData *src, struct BgAffineDstData *dest, s32 count)
{
    BgAffineSet_Impl(src, dest, count);
    BIOS_CALL_DONE();
}

static void ObjAffineSet_Impl(struct ObjAffineSrcData *src, void *dest, s32 count, s32 diff)
{
    u32 offset = (u32)src;
    u32 destination = (u32)dest;
    s32 i = count;

    while (i--)
    {
        float sx = (s16)Load16(offset) / 256.f;
        float sy = (s16)Load16(offset + 2) / 256.f;
        u32 k = (Load16(offset + 4) >> 8) & 0xFF;
        float a, b, c, d;

        offset += 8;
        a = d = FloatFromBits(sAffineCos[k]);
        b = c = FloatFromBits(sAffineSin[k]);
        a *= sx;
        b *= -sx;
        c *= sy;
        d *= sy;
        Store16(destination, (s16)(a * 256));
        Store16(destination + diff, (s16)(b * 256));
        Store16(destination + diff * 2, (s16)(c * 256));
        Store16(destination + diff * 3, (s16)(d * 256));
        destination += diff * 4;
    }
}

void ObjAffineSet(struct ObjAffineSrcData *src, void *dest, s32 count, s32 diff)
{
    ObjAffineSet_Impl(src, dest, count, diff);
    BIOS_CALL_DONE();
}

// ---- reset / waits / multiboot ------------------------------------------------

static void RegisterRamReset_Impl(u32 flags)
{
    // _RegisterRamReset (mGBA HLE): EWRAM, IWRAM except its top 0x200 bytes
    // (stacks and the SOUND_INFO_PTR/INTR_CHECK/INTR_VECTOR words), palette,
    // VRAM and OAM are plain memory at their GBA addresses (mirror layout).
    PORT_IO16(REG_OFFSET_DISPCNT) = DISPCNT_FORCED_BLANK;
    if (flags & RESET_EWRAM)
        memset((void *)EWRAM_START, 0, EWRAM_END - EWRAM_START);
    if (flags & RESET_IWRAM)
        memset((void *)IWRAM_START, 0, PORT_IWRAM_RESET_SIZE);
    if (flags & RESET_PALETTE)
        memset((void *)PLTT, 0, PLTT_SIZE);
    if (flags & RESET_VRAM)
        memset((void *)VRAM, 0, VRAM_SIZE);
    if (flags & RESET_OAM)
        memset((void *)OAM, 0, OAM_SIZE);
    if (flags & RESET_SIO_REGS)
    {
        PORT_IO16(REG_OFFSET_SIOCNT) = 0;
        PORT_IO16(REG_OFFSET_RCNT) = 0x8000;
        PORT_IO16(REG_OFFSET_SIOMLT_SEND) = 0;
        PORT_IO16(REG_OFFSET_JOYCNT) = 0;
        PORT_IO32(REG_OFFSET_JOY_RECV) = 0;
        PORT_IO32(REG_OFFSET_JOY_TRANS) = 0;
    }
    if (flags & RESET_SOUND_REGS)
    {
        static const u16 sSoundRegs[] = {
            REG_OFFSET_SOUND1CNT_L, REG_OFFSET_SOUND1CNT_H, REG_OFFSET_SOUND1CNT_X,
            REG_OFFSET_SOUND2CNT_L, REG_OFFSET_SOUND2CNT_H,
            REG_OFFSET_SOUND3CNT_L, REG_OFFSET_SOUND3CNT_H, REG_OFFSET_SOUND3CNT_X,
            REG_OFFSET_SOUND4CNT_L, REG_OFFSET_SOUND4CNT_H,
            REG_OFFSET_SOUNDCNT_L, REG_OFFSET_SOUNDCNT_H, REG_OFFSET_SOUNDCNT_X,
        };
        u32 i;
        for (i = 0; i < ARRAY_COUNT(sSoundRegs); i++)
            PORT_IO16(sSoundRegs[i]) = 0;
        PORT_IO16(REG_OFFSET_SOUNDBIAS) = 0x200;
        memset((void *)(REG_BASE + REG_OFFSET_WAVE_RAM0), 0, 16);
    }
    if (flags & RESET_REGS)
    {
        // DISPSTAT..BLDY, DMA, timers, IE, IF, WAITCNT, IME. VCOUNT is read
        // only (mGBA ignores the store). IF=0xFFFF acknowledges everything.
        u32 off;
        PORT_IO16(REG_OFFSET_DISPSTAT) &= DISPSTAT_VBLANK | DISPSTAT_HBLANK | DISPSTAT_VCOUNT;
        for (off = REG_OFFSET_BG0CNT; off <= REG_OFFSET_BLDY; off += 2)
            PORT_IO16(off) = 0;
        PORT_IO16(REG_OFFSET_BG2PA) = 0x100;
        PORT_IO16(REG_OFFSET_BG2PD) = 0x100;
        PORT_IO16(REG_OFFSET_BG3PA) = 0x100;
        PORT_IO16(REG_OFFSET_BG3PD) = 0x100;
        for (off = REG_OFFSET_DMA0SAD; off < REG_OFFSET_DMA3CNT + 4; off += 2)
            PORT_IO16(off) = 0;
        for (off = REG_OFFSET_TM0CNT; off < REG_OFFSET_TM3CNT + 4; off += 2)
            PORT_IO16(off) = 0;
        PORT_IO16(REG_OFFSET_IE) = 0;
        PORT_IO16(REG_OFFSET_IF) = 0;
        PORT_IO16(REG_OFFSET_WAITCNT) = 0;
        PORT_IO16(REG_OFFSET_IME) = 0;
        Port_DmaPowerOn();
    }
}

void RegisterRamReset(u32 flags)
{
    RegisterRamReset_Impl(flags);
    BIOS_CALL_DONE();
}

void SoftReset(u32 resetFlags)
{
    // libagbsyscall: IME = 0; RegisterRamReset(resetFlags); SWI 0 (SoftReset),
    // which clears the top 0x200 bytes of IWRAM and restarts the ROM. The
    // simulator hands control back to the host, which powers the environment
    // on again keeping the flash contents (see Port_RunFrame).
    PORT_IO16(REG_OFFSET_IME) = 0;
    RegisterRamReset(resetFlags);
    memset((void *)(IWRAM_START + PORT_IWRAM_RESET_SIZE), 0, IWRAM_END - IWRAM_START - PORT_IWRAM_RESET_SIZE);
    Port_RequestSoftReset();
}

void VBlankIntrWait(void)
{
    // Only the e-Reader code calls it; a nested wait inside a frame is not
    // supported by the frame model (ARCHITECTURE.md section 1.3).
    Port_Unsupported("VBlankIntrWait (e-Reader)");
}

int MultiBoot(struct MultiBootParam *mp)
{
    // Sending a multiboot image to another GBA requires a link partner.
    Port_Unsupported("MultiBoot (link cable)");
}
