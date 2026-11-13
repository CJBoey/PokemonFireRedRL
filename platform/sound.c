// fireredsim platform layer: C port of game/src/m4a_1.s (the m4a sound engine
// routines that are hand-written ARM/Thumb assembly in the ROM).
//
// This is a line-by-line transliteration of the assembly, not a
// re-implementation: register-level details that change game-visible state
// are kept (32-bit copies of adjacent struct fields, unsigned/signed compares,
// the full-width LFO sum, the MPlayMain chain order, the shared BDPCM decoding
// buffer, the integer PCM mixer with its rotate-accumulate trick and its
// end-of-sample handling). Labels in comments refer to m4a_1.s.
//
// Like game/native/pokefirered.patch, this file is derived from the game's own
// code (through pret's disassembly) and is not covered by this repository's
// MIT licence (THIRD_PARTY.md).
//
// The mixer writes gSoundInfo.pcmBuffer exactly like SoundMainRAM; nothing
// plays it (the sound FIFO DMA is not executed, see platform/dma.c). The mixer
// must still run because channel state (count, currentPointer, statusFlags,
// envelope) decides when sounds end, which scripts wait for (waitse,
// waitmoncry, fanfares). By default it runs in logic-only mode: that channel
// state exactly, no sample synthesis (see "Logic-only mixing" below,
// Port_SetSoundMix); the full mixer stays available at run time.
// m4a_internal.h declares SoundMainBTM(void), but the routine receives the
// address to clear in r0 (it is called as Clear64byte through the jump table).
#define SoundMainBTM SoundMainBTM_HeaderDeclaration
#include "port_internal.h"
#include "m4a.h"
#undef SoundMainBTM

// Constants of m4a_1.s (constants/m4a_constants.inc) without a C definition.
#define TONEDATA_TYPE_REV          0x10
#define TONEDATA_TYPE_CMP          0x20
#define SOUND_CHANNEL_SF_SPECIAL   0x20
#define WAVE_DATA_STATUS_LOOP      0xC000 // WAVE_DATA_FLAG_LOOP (0xC0) of the byte at offset 3

u32 MidiKeyToFreq(struct WaveData *wav, u8 key, u8 fineAdjust);

extern const u8 gClockTable[];
extern const s8 gDeltaEncodingTable[];
extern void *const gMPlayJumpTableTemplate[];

// .bss in m4a_1.s: 64 decoded samples of one BDPCM block, shared by all channels.
// IWRAM_DATA: placed at its ROM address (0x03002088) by the mirror layout.
IWRAM_DATA static s8 sDecodingBuffer[0x40] ALIGNED(8);

typedef void (*MPlayMainFunc)(struct MusicPlayerInfo *);
typedef void (*TrackFunc)(struct MusicPlayerInfo *, struct MusicPlayerTrack *);
typedef void (*PlyNoteFunc)(u32, struct MusicPlayerInfo *, struct MusicPlayerTrack *);
typedef u32 (*MidiKeyToCgbFreqF)(u8, u8, u8);
typedef void (*CgbOscOffF)(u8);

#define ROR32(v, n) (((v) >> (n)) | ((v) << ((32 - (n)) & 31)))

// ---------------------------------------------------------------------------
// umul3232H32

u32 umul3232H32(u32 multiplier, u32 multiplicand)
{
    return (u32)(((u64)multiplier * multiplicand) >> 32);
}

// ---------------------------------------------------------------------------
// SoundMainRAM: the PCM mixer (and SoundMainRAM_Unk1/Unk2 for compressed or
// reversed samples).

// Per-call state of the mixer (registers/stack slots of SoundMainRAM).
struct MixState
{
    u32 samplesPerVBlank;   // r8 / [sp]
    u8 *mixPos;             // [sp+0x8]: start of this frame's segment in pcmBuffer
    s8 *loopStart;          // [sp+0xC]
    s32 loopLen;            // [sp+0x10]
    u32 divFreq;            // r12
};

// One mixing step for a sample: the top byte of the rotating accumulator words
// gets the scaled sample added (r6 = (env*s & 0xFF00..) + ror(r6, 8)).
static inline void MixSample(u32 *r6, u32 *r7, u32 envR16, u32 envL16, s32 s)
{
    u32 r1 = (envR16 * (u32)s) & ~0xFF0000u;
    *r6 = r1 + ROR32(*r6, 8);
    r1 = (envL16 * (u32)s) & ~0xFF0000u;
    *r7 = r1 + ROR32(*r7, 8);
}

// _081DD174 / _081DD4F4: channel ended after `done` samples of the current
// 4-sample word; rotate the partial words back into place and store them.
static inline void FlushPartialWord(u32 r5, u32 r6, u32 r7)
{
    u32 done = r5 >> 30;
    u32 addr = r5 & ~0xC0000000u;
    u32 rot = ((3 - done) << 3) & 31;
    r6 = ROR32(r6, rot);
    r7 = ROR32(r7, rot);
    *(u32 *)(addr + PCM_DMA_BUF_SIZE) = r7;
    *(u32 *)addr = r6;
}

// The mix pointer r5 is a 30-bit address plus a 2-bit sample counter in its
// top bits. Native addresses fit in 30 bits (checked in Port_SoundPowerOn).
#define R5_ADDR(r5) ((r5) & ~0xC0000000u)

// A byte of a BDPCM block (SoundMainRAM_Unk2's LDRB). A channel that plays a
// compressed sample in reverse steps its sample index below 0 (SoundMainRAM_Unk1,
// "sub r3, r3, 1"), so the block address wav + 0x21 * (index >> 6) + 0x10 wraps
// into unmapped GBA memory (>= 0x10000000). The ARM CPU then reads its open bus:
// the opcode it prefetched, i.e. the instruction two after the load in the IWRAM
// copy of SoundMainRAM_Unk2 (byte addr & 3; mGBA GBALoad8 -> GBALoadBad in ARM
// state, ROM 0x081DC754..0x081DC76C). Natively such a read ended the environment
// as INEXACT (found by the coverage runs, docs/COVERAGE.md). Other addresses are
// read normally (memfault.c routes the BIOS region and mirrors). Caveat: mGBA
// returns the DMA bus value instead when a DMA ran right before the load (a sound
// FIFO DMA landing exactly there); not modelled.
static u8 LoadBlockByte(const u8 *p, u32 prefetchedOpcode)
{
    u32 a = (u32)p;
    if (a >= 0x10000000u || (a >= 0x00004000u && a < 0x02000000u))
        return (u8)(prefetchedOpcode >> ((a & 3) * 8));
    return *p;
}

#define OPC_AFTER_FIRST_LDRB 0xE4D21001u  // ldrb r1, [r2], #1 at SoundMainRAM_Unk2+0x40
#define OPC_AFTER_SECOND_LDRB 0xE4D21001u // ldrb r1, [r2], #1 at +0x48 (_081DD568)
#define OPC_AFTER_LOOP_LDRB 0xE19600D0u   // ldrsb r0, [r6, r0] at +0x50

// SoundMainRAM_Unk2: decoded BDPCM sample at index r3 of chan's wave.
static s32 DecodeSample(struct SoundChannel *chan, u32 r3)
{
    u32 r0 = r3 >> 6;
    u32 xp = chan->xpi | ((u32)chan->xpc << 16);
    if (r0 != xp)
    {
        const u8 *r2;
        s8 *r5 = sDecodingBuffer;
        s32 lr;
        u32 r1;
        s32 r7 = 0x40;

        chan->xpi = (u16)r0;
        chan->xpc = (u16)(r0 >> 16);
        r2 = (const u8 *)((u32)chan->wav + r0 * 0x21 + 0x10);
        lr = LoadBlockByte(r2++, OPC_AFTER_FIRST_LDRB);
        *r5++ = (s8)lr;
        r1 = LoadBlockByte(r2++, OPC_AFTER_SECOND_LDRB);
        goto low;
        do
        {
            r1 = LoadBlockByte(r2++, OPC_AFTER_LOOP_LDRB);
            lr += gDeltaEncodingTable[r1 >> 4];
            *r5++ = (s8)lr;
        low:
            lr += gDeltaEncodingTable[r1 & 0xF];
            *r5++ = (s8)lr;
            r7 -= 2;
        } while (r7 > 0);
    }
    return sDecodingBuffer[r3 & 0x3F];
}

// SoundMainRAM_Unk1. r2 (count) and r3 (currentPointer / sample index) are
// updated in place; returns with the same register meaning as the caller
// expects at _081DD228 (fw, count and pointer are stored afterwards).
static void MixSpecial(struct SoundChannel *chan, struct MixState *ms, u32 *pr2, u32 *pr3, u32 *pr9,
                       u32 envR, u32 envL)
{
    struct WaveData *wav = chan->wav;
    u32 r2 = *pr2;
    u32 r3 = *pr3;
    u32 r9 = *pr9;
    u32 r5 = (u32)ms->mixPos;
    u32 r6, r7, r8, lr;
    s32 r0, r1;
    s32 samplesLeft = ms->samplesPerVBlank; // [sp] after push {r8, r12, lr}
    u32 envR16 = envR << 16;
    u32 envL16 = envL << 16;

    if (!(chan->statusFlags & SOUND_CHANNEL_SF_SPECIAL))
    {
        chan->statusFlags |= SOUND_CHANNEL_SF_SPECIAL;
        if (chan->type & TONEDATA_TYPE_REV)
        {
            r3 = wav->size + ((u32)wav << 1) + 0x20 - r3;
            chan->currentPointer = (s8 *)r3;
        }
        if (wav->type != 0)
        {
            r3 = r3 - (u32)wav - 0x10;
            chan->currentPointer = (s8 *)r3;
        }
    }
    if (chan->type & TONEDATA_TYPE_FIX)
        r8 = 0x800000;
    else
        r8 = ms->divFreq * chan->frequency;

    if (wav->type != 0)
    {
        chan->xpi = 0;
        chan->xpc = 0xFF00;
        if (!(chan->type & TONEDATA_TYPE_REV))
        {
            // compressed, forward
            r1 = DecodeSample(chan, r3);
            r0 = r1;
            r3 += 1;
            r1 = DecodeSample(chan, r3);
            r1 = r1 - r0;
            for (;;) // _081DD308
            {
                r6 = *(u32 *)R5_ADDR(r5);
                r7 = *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE);
                do // _081DD310
                {
                    s32 s = r0 + ((s32)(r9 * (u32)r1) >> 23);
                    MixSample(&r6, &r7, envR16, envL16, s);
                    r9 += r8;
                    lr = r9 >> 23;
                    if (lr != 0)
                    {
                        r9 &= ~0x3F800000u;
                        r2 -= lr;
                        if ((s32)r2 <= 0)
                        {
                            // _081DD398
                            s32 loopLen = ms->loopLen;
                            if (loopLen == 0)
                                goto stop;
                            r3 = chan->wav->loopStart;
                            lr = -(s32)r2;
                            for (;;) // _081DD3B0
                            {
                                r2 += loopLen;
                                if ((s32)r2 > 0)
                                    break;
                                lr -= loopLen;
                            }
                            goto reload; // _081DD358
                        }
                        lr -= 1;
                        if (lr != 0)
                        {
                        reload: // _081DD358
                            r3 += lr;
                            r1 = DecodeSample(chan, r3);
                            r0 = r1;
                        }
                        else
                        {
                            r0 = r0 + r1;
                        }
                        // _081DD364
                        r3 += 1;
                        r1 = DecodeSample(chan, r3);
                        r1 = r1 - r0;
                    }
                    r5 += 0x40000000; // _081DD370
                } while (r5 >> 30);
                *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE) = r7;
                *(u32 *)R5_ADDR(r5) = r6;
                r5 += 4;
                samplesLeft -= 4;
                if (samplesLeft <= 0)
                    break;
            }
            r3 -= 1;
            goto done;
        }
        else
        {
            // compressed, reverse (_081DD3C0)
            r3 -= 1;
            r1 = DecodeSample(chan, r3);
            r0 = r1;
            r3 -= 1;
            r1 = DecodeSample(chan, r3);
            r1 = r1 - r0;
            for (;;) // _081DD3D8
            {
                r6 = *(u32 *)R5_ADDR(r5);
                r7 = *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE);
                do // _081DD3E0
                {
                    s32 s = r0 + ((s32)(r9 * (u32)r1) >> 23);
                    MixSample(&r6, &r7, envR16, envL16, s);
                    r9 += r8;
                    lr = r9 >> 23;
                    if (lr != 0)
                    {
                        r9 &= ~0x3F800000u;
                        r2 -= lr;
                        if ((s32)r2 <= 0)
                            goto stop;
                        lr -= 1;
                        if (lr != 0)
                        {
                            r3 -= lr; // _081DD428
                            r1 = DecodeSample(chan, r3);
                            r0 = r1;
                        }
                        else
                        {
                            r0 = r0 + r1;
                        }
                        r3 -= 1; // _081DD434
                        r1 = DecodeSample(chan, r3);
                        r1 = r1 - r0;
                    }
                    r5 += 0x40000000; // _081DD440
                } while (r5 >> 30);
                *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE) = r7;
                *(u32 *)R5_ADDR(r5) = r6;
                r5 += 4;
                samplesLeft -= 4;
                if (samplesLeft <= 0)
                    break;
            }
            r3 += 2;
            goto done;
        }
    }
    else
    {
        // _081DD468: uncompressed
        if (!(chan->type & TONEDATA_TYPE_REV))
            goto done; // forward uncompressed with CMP set: nothing is mixed
        r3 -= 1;
        r0 = *(s8 *)r3;
        r1 = *(s8 *)(r3 - 1);
        r1 = r1 - r0;
        for (;;) // _081DD480
        {
            r6 = *(u32 *)R5_ADDR(r5);
            r7 = *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE);
            do // _081DD488
            {
                s32 s = r0 + ((s32)(r9 * (u32)r1) >> 23);
                MixSample(&r6, &r7, envR16, envL16, s);
                r9 += r8;
                lr = r9 >> 23;
                if (lr != 0)
                {
                    r9 &= ~0x3F800000u;
                    r2 -= lr;
                    if ((s32)r2 <= 0)
                        goto stop;
                    r3 -= lr;
                    r0 = *(s8 *)r3;
                    r1 = *(s8 *)(r3 - 1);
                    r1 = r1 - r0;
                }
                r5 += 0x40000000; // _081DD4CC
            } while (r5 >> 30);
            *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE) = r7;
            *(u32 *)R5_ADDR(r5) = r6;
            r5 += 4;
            samplesLeft -= 4;
            if (samplesLeft <= 0)
                break;
        }
        r3 += 1;
        goto done;
    }

stop: // _081DD4F4
    r2 = 0;
    chan->statusFlags = 0;
    FlushPartialWord(r5, r6, r7);
done: // _081DD4F0
    *pr2 = r2;
    *pr3 = r3;
    *pr9 = r9;
}

// Mixes one channel that is on (from _081DD006 to _081DD234).
static void MixChannel(struct SoundInfo *si, struct SoundChannel *chan, struct MixState *ms, u32 r6flags, u32 r5env)
{
    struct WaveData *wav = chan->wav;
    u32 vol, envR, envL;
    u32 r2, r3, r9, r5, r6, r7, r8;
    s32 r0, r1, lr;

    // _081DD006
    chan->envelopeVolume = (u8)r5env;
    vol = ((si->masterVolume + 1) * r5env) >> 4; // ldrb [soundInfo, o_SoundChannel_release]: masterVolume
    chan->envelopeVolumeRight = (u8)((chan->rightVolume * vol) >> 8);
    chan->envelopeVolumeLeft = (u8)((chan->leftVolume * vol) >> 8);
    ms->loopLen = r6flags & SOUND_CHANNEL_SF_LOOP;
    if (ms->loopLen)
    {
        ms->loopStart = (s8 *)wav->data + wav->loopStart;
        ms->loopLen = wav->size - wav->loopStart;
    }
    // _081DD03A
    r5 = (u32)ms->mixPos;
    r2 = chan->count;
    r3 = (u32)chan->currentPointer;
    r8 = ms->samplesPerVBlank;
    r9 = chan->fw;
    envR = chan->envelopeVolumeRight;
    envL = chan->envelopeVolumeLeft;

    if (chan->type & (TONEDATA_TYPE_CMP | TONEDATA_TYPE_REV))
    {
        MixSpecial(chan, ms, &r2, &r3, &r9, envR, envL);
        goto store_fw; // _081DD228
    }

    {
        u32 envR16 = envR << 16;
        u32 envL16 = envL << 16;

        if (chan->type & TONEDATA_TYPE_FIX)
        {
            // _081DD07C: no resampling, one source sample per output sample
            s32 r8s = (s32)r8;
            s32 r9s;
            for (;;)
            {
                if ((s32)r2 > 4)
                {
                    r2 -= r8s;
                    if ((s32)r2 > 0)
                    {
                        r9s = 0;
                    }
                    else
                    {
                        r9s = r8s;
                        r2 += r8s;
                        r8s = (s32)r2 - 4;
                        r9s -= r8s;
                        r2 &= 3;
                        if (r2 == 0)
                            r2 = 4;
                    }
                    // _081DD0A8: blocks of 4 samples without end checks
                    do
                    {
                        r6 = *(u32 *)R5_ADDR(r5);
                        r7 = *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE);
                        do
                        {
                            r0 = *(s8 *)r3;
                            r3++;
                            MixSample(&r6, &r7, envR16, envL16, r0);
                            r5 += 0x40000000;
                        } while (r5 >> 30);
                        *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE) = r7;
                        *(u32 *)R5_ADDR(r5) = r6;
                        r5 += 4;
                        r8s -= 4;
                    } while (r8s > 0);
                    r8s += r9s;
                    if (r8s == 0)
                        goto store_count; // _081DD22C
                }
                // _081DD0EC: sample by sample, checking the end of the sample
                r6 = *(u32 *)R5_ADDR(r5);
                r7 = *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE);
                for (;;)
                {
                    r0 = *(s8 *)r3;
                    r3++;
                    MixSample(&r6, &r7, envR16, envL16, r0);
                    r2 -= 1;
                    if (r2 == 0)
                    {
                        // _081DD164
                        r2 = ms->loopLen;
                        if (r2 == 0)
                        {
                            // _081DD174: sample ended, no loop
                            chan->statusFlags = 0;
                            FlushPartialWord(r5, r6, r7);
                            return; // _081DD234: count/pointer not stored
                        }
                        r3 = (u32)ms->loopStart;
                    }
                    // _081DD118
                    r5 += 0x40000000;
                    if (!(r5 >> 30))
                        break;
                }
                *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE) = r7;
                *(u32 *)R5_ADDR(r5) = r6;
                r5 += 4;
                r8s -= 4;
                if (r8s <= 0)
                    goto store_count;
                // bgt _081DD07C
            }
        }
        else
        {
            // _081DD19C: resampling with linear interpolation
            u32 step = ms->divFreq * chan->frequency;
            r0 = *(s8 *)r3;
            r3 += 1;
            r1 = *(s8 *)r3;
            r1 = r1 - r0;
            for (;;) // _081DD1B4
            {
                r6 = *(u32 *)R5_ADDR(r5);
                r7 = *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE);
                do // _081DD1BC
                {
                    s32 s = r0 + ((s32)(r9 * (u32)r1) >> 23);
                    MixSample(&r6, &r7, envR16, envL16, s);
                    r9 += step;
                    lr = (s32)(r9 >> 23);
                    if (lr != 0)
                    {
                        r9 &= ~0x3F800000u;
                        r2 -= lr;
                        if ((s32)r2 <= 0)
                        {
                            // _081DD134
                            s32 loopLen = ms->loopLen;
                            if (loopLen == 0)
                            {
                                // _081DD158 -> _081DD174
                                chan->statusFlags = 0;
                                FlushPartialWord(r5, r6, r7);
                                return;
                            }
                            r3 = (u32)ms->loopStart;
                            lr = -(s32)r2;
                            for (;;) // _081DD148
                            {
                                r2 += loopLen;
                                if ((s32)r2 > 0)
                                    break;
                                lr -= loopLen;
                            }
                            // _081DD1FC with flags NE (from adds): reload
                            r3 += lr;
                            r0 = *(s8 *)r3;
                        }
                        else
                        {
                            lr -= 1;
                            if (lr == 0)
                            {
                                r0 = r0 + r1;
                            }
                            else
                            {
                                r3 += lr; // _081DD1FC ldrsbne r0, [r3, lr]!
                                r0 = *(s8 *)r3;
                            }
                        }
                        r3 += 1;
                        r1 = *(s8 *)r3;
                        r1 = r1 - r0;
                    }
                    r5 += 0x40000000; // _081DD208
                } while (r5 >> 30);
                *(u32 *)(R5_ADDR(r5) + PCM_DMA_BUF_SIZE) = r7;
                *(u32 *)R5_ADDR(r5) = r6;
                r5 += 4;
                if ((s32)(r8 -= 4) <= 0)
                    break;
            }
            r3 -= 1;
        }
    }

store_fw: // _081DD228
    chan->fw = r9;
store_count: // _081DD22C
    chan->count = r2;
    chan->currentPointer = (s8 *)r3;
}

// ---------------------------------------------------------------------------
// Logic-only mixing (PORT_SOUND_MIX_LOGIC, the default; docs/DEVLOG.md "sim
// performance agent", docs/BENCHMARKS.md).
//
// Everything the mixer writes besides gSoundInfo.pcmBuffer is game state: the
// channel fields (statusFlags, envelopeVolume/Right/Left, pseudoEchoLength,
// count, currentPointer, fw, xpi/xpc), the shared BDPCM decoding buffer and
// soundInfo->ident. The PCM buffer itself is audio output only: the sound
// FIFO DMA reads it on hardware (not executed natively), no game code reads it,
// and nothing the mixer computes depends on it (the accumulate step reads it
// only to write it back). In logic mode the mixer updates every other field
// exactly as the full mixer does and does not synthesize samples: the per-sample
// position advance of each loop has a closed form with the loop's own integer
// rounding (below), and the PCM buffer is left untouched (it keeps whatever it
// held: zero after power-on and SoundInit's clear).
//
// Equivalence holds whenever the closed forms are used under their
// preconditions; anything else takes the full code path with the PCM output
// redirected to a host scratch buffer, which computes the channel state with the
// very same code:
//
//   * per frame (LogicMixAllowed): every write of the full mixer (buffer clear
//     or reverb, the samplesPerVBlank output samples of each channel) lies inside
//     gSoundInfo.pcmBuffer and the reverb reads stay inside it too; otherwise the
//     frame is mixed in full (a mixer writing outside the PCM buffer would change
//     other game state);
//   * per channel (MixChannelLogic, MixSpecialLogic): fw < 2^23 and the
//     position step <= 0x3F800000 (so r9 + step never reaches bit 30 and the
//     integer part of fw + k*step is exactly the sum of the per-sample advances),
//     count > 0 and loop length >= 0 (as s32), samplesPerVBlank a multiple
//     of 4 (the FIX path's block loop), and every sample byte the full mixer
//     would read lies inside one loaded segment of the executable (the ROM
//     window or a state region), so that the full mixer cannot have faulted
//     there (memfault.c) where the logic mixer does not read at all.
//
// The closed forms, per channel, with K = samplesPerVBlank output samples,
// r2 = count, r3 = currentPointer, r9 = fw, step = divFreq * frequency (FIX:
// one source sample per output sample; special paths: 0x800000),
// A = (r9 + K * step) >> 23 the source samples consumed:
//   resampling (_081DD19C): A < r2: count -= A, pointer += A, fw = low 23 bits;
//     A >= r2 without loop: the channel stops (statusFlags = 0, nothing else
//     stored); with loop length L: count = L - (A - r2) % L, pointer = loop end
//     - count (each wrap re-bases the pointer on the loop end);
//   FIX (_081DD07C): the same with A = K (fw is neither used nor stored);
//   special (SoundMainRAM_Unk1, compressed and/or reversed): the same, except
//     that a stop happens at the first sample k* with (r9 + k* * step) >> 23 >=
//     count and stores count 0, the pointer before that sample's advance and
//     fw at k*; only compressed forward samples loop. Compressed samples end
//     with the block of the last DecodeSample call decoded into
//     sDecodingBuffer and its index in xpi/xpc.
// The mixer's stack frames are kept off the game stack (SoundMain): both modes
// leave the same environment state except the PCM buffer.
// tests/fuzz/test_sound_mix.cpp checks the closed forms against the full mixer
// on random channel states; tools/fuzz (fireredsim-mixer-ab, fireredsim-fuzz
// --ab-mixer) compares whole runs.

HOST_STATE static uint8_t sSoundMix;      // PORT_SOUND_MIX_LOGIC (0, default) or PORT_SOUND_MIX_FULL
HOST_STATE static struct PortSoundMixStats sMixStats;
// Scratch PCM output for channels that take the full code path in logic mode
// (same offsets as in pcmBuffer; never read for anything but PCM output).
HOST_STATE static u32 sScratchPcm[(PCM_DMA_BUF_SIZE * 2 + 16) / 4] ALIGNED(8);

// Sound data inside gSoundInfo.pcmBuffer. Logic mode leaves the PCM buffer
// untouched, so a sound-engine read of it would see other bytes than after a
// full mix (Codex review, docs/coordination/codex.md CX-010: a WaveData header
// placed at pcmBuffer, which the full mixer clears before the channel reads it).
// Retail sound data (songs, voice groups, wave headers and samples) is in the
// ROM and the songs the game builds in RAM (cries) lie elsewhere, so only a
// stray sound pointer reaches the buffer. Every sound-engine read that could
// reach it is checked (command fetches, song headers in MPlayStart, voice
// tables and key split tables, note tones, wave headers and wave data at note
// start, wave headers at every logic mix, the logic mixer's sample spans and its
// full-path fallback, CgbSound's wave copy, ply_memacc's area); in logic mode such a read ends the environment as
// INEXACT, while PORT_SOUND_MIX_FULL computes it exactly. Game code outside the
// sound engine never reads the PCM buffer. The checks outside the mixer run on
// the game stack, so they execute the same instructions in both modes and only
// PcmAliasHit reads the mode: both modes leave the same stack bytes
// (sound_mix_ab_*). tests/fuzz/test_sound_alias.cpp.
#define PCM_LO ((u32)gSoundInfo.pcmBuffer)
#define PCM_HI ((u32)gSoundInfo.pcmBuffer + (u32)sizeof(gSoundInfo.pcmBuffer))

static void __attribute__((noipa)) PcmAliasHit(u32 lo, u32 n, const char *what)
{
    if (sSoundMix == PORT_SOUND_MIX_LOGIC)
        Port_Inexact("sound %s at 0x%08x (%u bytes) overlaps gSoundInfo.pcmBuffer, which logic-only mixing does not "
                     "compute (PORT_SOUND_MIX_FULL is exact here)", what, lo, n);
}

// IWRAM is mirrored every 0x8000 bytes up to 0x03FFFFFF (platform/memfault.c
// maps a mirror access to the IWRAM bytes; Codex review CX-018): the part of
// [lo, lo + n) inside 0x03000000..0x03FFFFFF, as IWRAM offsets, against the
// buffer's offsets.
#define IWRAM_LO 0x03000000u
#define IWRAM_MIRRORS 0x01000000u
static void __attribute__((noipa)) PcmAliasCheckIwram(u32 lo, u64 n, const char *what)
{
    u64 a = lo < IWRAM_LO ? IWRAM_LO : lo, b = (u64)lo + n;
    u32 o, m, plo = PCM_LO - IWRAM_LO, phi = PCM_HI - IWRAM_LO;
    u32 shown = n > 0xFFFFFFFFu ? 0xFFFFFFFFu : (u32)n;

    if (b > (u64)IWRAM_LO + IWRAM_MIRRORS)
        b = (u64)IWRAM_LO + IWRAM_MIRRORS;
    if (a >= b)
        return;
    if (b - a >= 0x8000)
    {
        PcmAliasHit(lo, shown, what);
        return;
    }
    o = (u32)(a - IWRAM_LO) & 0x7FFF;
    m = (u32)(b - a);
    if ((o < phi && o + m > plo) || (o + m > 0x8000 && o + m - 0x8000 > plo))
        PcmAliasHit(lo, shown, what);
}

// [lo, lo + n), 0 < n < 2^24: only addresses in the IWRAM window can reach the
// buffer (two compares otherwise).
static inline void PcmAliasCheck(u32 lo, u32 n, const char *what)
{
    if (lo - IWRAM_LO < IWRAM_MIRRORS || lo + n - 1 - IWRAM_LO < IWRAM_MIRRORS)
        PcmAliasCheckIwram(lo, n, what);
}

// The same for any extent (one wrapping past 2^32 is not considered).
static inline void PcmAliasCheck64(u32 lo, u64 n, const char *what)
{
    if (n != 0 && (lo - IWRAM_LO < IWRAM_MIRRORS || (lo < IWRAM_LO && (u64)lo + n > IWRAM_LO)))
        PcmAliasCheckIwram(lo, n, what);
}

// Game-side sound-engine reads (game/src/m4a.c: CgbSound's wave copy, ply_memacc).
void Port_SoundReadCheck(const void *p, unsigned int size)
{
    PcmAliasCheck64((u32)p, size, "data");
}

// MPlayStart (game/src/m4a.c): the song header and the track pointers it can
// read (at most the player's track count; nothing is read here).
void Port_SoundSongCheck(const void *songHeader, unsigned int playerTracks)
{
    PcmAliasCheck64((u32)songHeader, 8 + 4 * (u64)playerTracks, "song header");
}

// A new DirectSound note's wave (ply_note, before MidiKeyToFreq reads the
// header): the header, then everything the channel can read from it: size
// samples (+1: the resampler reads the next one) or 0x21-byte blocks of 64
// compressed samples. The extent is conservative (a channel reaches its end only
// after enough frames); SpanReadable checks the exact spans of the closed forms.
static void WaveAliasCheck(const struct WaveData *wav)
{
    PcmAliasCheck((u32)wav, 16, "wave header");
    PcmAliasCheck64((u32)wav + 16, wav->type != 0 ? 0x21 * (((u64)wav->size + 63) >> 6) : (u64)wav->size + 1,
                    "wave data");
}

// Readable memory: the PT_LOAD segments of the executable (code and read-only
// data at 0x08000000 = the ROM window, the state regions), from its program
// headers, which are mapped with the ELF header at __executable_start.
#define SOUND_MAX_SEGMENTS 32
extern const u8 __executable_start[];
HOST_STATE static u32 sSegLo[SOUND_MAX_SEGMENTS], sSegHi[SOUND_MAX_SEGMENTS];
HOST_STATE static u32 sSegCount;
HOST_STATE static u8 sSegDone;

static void FindReadableSegments(void)
{
    const u8 *eh = __executable_start;
    u32 phoff, phentsize, phnum, i;

    sSegDone = 1;
    sSegCount = 0;
    // ELFCLASS32 executable: e_phoff at 28, e_phentsize at 42, e_phnum at 44;
    // Elf32_Phdr: p_type at 0 (PT_LOAD = 1), p_vaddr at 8, p_memsz at 20.
    if (eh[0] != 0x7F || eh[1] != 'E' || eh[2] != 'L' || eh[3] != 'F' || eh[4] != 1)
        return;
    phoff = *(const u32 *)(eh + 28);
    phentsize = *(const u16 *)(eh + 42);
    phnum = *(const u16 *)(eh + 44);
    if (phentsize < 32)
        return;
    for (i = 0; i < phnum && sSegCount < SOUND_MAX_SEGMENTS; i++)
    {
        const u8 *ph = eh + phoff + i * phentsize;
        u32 vaddr = *(const u32 *)(ph + 8), memsz = *(const u32 *)(ph + 20);
        if (*(const u32 *)ph != 1 || memsz == 0 || vaddr + memsz < vaddr)
            continue;
        sSegLo[sSegCount] = vaddr;
        sSegHi[sSegCount] = vaddr + memsz; // exclusive
        sSegCount++;
    }
}

HOST_STATE static u32 sSegLast; // segment of the last readable span (sample data: mostly the same)

// TRUE if every byte of [lo, hi] (inclusive) lies inside one readable segment.
// Logic mode only (sample spans of the closed forms): a span inside
// gSoundInfo.pcmBuffer ends the environment (PcmAliasCheck).
static bool32 SpanReadable(s64 lo, s64 hi)
{
    u32 i;
    if (lo < 0 || hi < lo || hi > 0xFFFFFFFFll)
        return FALSE;
    PcmAliasCheck64((u32)lo, (u64)(hi - lo) + 1, "samples");
    i = sSegLast;
    if (i < sSegCount && (u32)lo >= sSegLo[i] && (u32)hi < sSegHi[i])
        return TRUE;
    for (i = 0; i < sSegCount; i++)
        if ((u32)lo >= sSegLo[i] && (u32)hi < sSegHi[i])
        {
            sSegLast = i;
            return TRUE;
        }
    return FALSE;
}

// Bytes the NoReverb loop of SoundMainRAM_Mix clears in each half for n samples.
static u32 ClearBytes(u32 n)
{
    u32 words = ((n >> 2) & 1) + 2 * ((n >> 3) & 1) + 4 * ((n >> 4) != 0 ? (n >> 4) : 1);
    return words * 4;
}

// TRUE if all writes (and reverb reads) of the full mixer in this frame lie
// inside gSoundInfo.pcmBuffer: then skipping them changes nothing else.
static bool32 LogicMixAllowed(struct SoundInfo *si, struct MixState *ms, u32 pcmDmaCounter)
{
    u32 n = ms->samplesPerVBlank;
    u32 base = (u32)gSoundInfo.pcmBuffer;
    u32 pos = (u32)ms->mixPos;

    if (si != &gSoundInfo || (s32)n <= 0 || (n & 3) != 0 || n > PCM_DMA_BUF_SIZE)
        return FALSE;
    if (pos < base || pos - base > PCM_DMA_BUF_SIZE - n)
        return FALSE;
    if (si->reverb != 0)
    {
        u32 r7 = (pcmDmaCounter == 2) ? base : pos + n;
        if (r7 < base || r7 - base > PCM_DMA_BUF_SIZE - n)
            return FALSE;
    }
    else if (ClearBytes(n) > PCM_DMA_BUF_SIZE || pos - base > PCM_DMA_BUF_SIZE - ClearBytes(n))
    {
        return FALSE;
    }
    return TRUE;
}

// The full code path for one channel, writing its PCM output to the scratch buffer.
static void MixChannelToScratch(struct SoundInfo *si, struct SoundChannel *chan, struct MixState *ms, u32 r6flags,
                                u32 r5env)
{
    u8 *mixPos = ms->mixPos;
    ms->mixPos = (u8 *)sScratchPcm + (mixPos - (u8 *)gSoundInfo.pcmBuffer);
    MixChannel(si, chan, ms, r6flags, r5env);
    ms->mixPos = mixPos;
    sMixStats.channelsFull++;
}

// SoundMainRAM_Unk1 in logic mode (chan->type has CMP or REV). r2 > 0 and the
// loop length are checked by the caller. Returns FALSE if the closed form does
// not apply (the caller then runs the full code path, which sees the
// SOUND_CHANNEL_SF_SPECIAL prologue as already done, exactly as in a later frame).
static bool32 MixSpecialLogic(struct SoundChannel *chan, struct MixState *ms, u32 r2, u32 r3, u32 r9)
{
    struct WaveData *wav = chan->wav;
    u32 type = chan->type;
    u32 n = ms->samplesPerVBlank;
    u32 r8, A, Abefore = 0, r9out, count, ptr, last = 0;
    s32 L = ms->loopLen;
    s64 lo, hi;
    u64 total;
    bool32 stop = FALSE, compressed, wrap = FALSE;
    int path = 5;

    // Prologue, as in MixSpecial.
    if (!(chan->statusFlags & SOUND_CHANNEL_SF_SPECIAL))
    {
        chan->statusFlags |= SOUND_CHANNEL_SF_SPECIAL;
        if (type & TONEDATA_TYPE_REV)
        {
            r3 = wav->size + ((u32)wav << 1) + 0x20 - r3;
            chan->currentPointer = (s8 *)r3;
        }
        if (wav->type != 0)
        {
            r3 = r3 - (u32)wav - 0x10;
            chan->currentPointer = (s8 *)r3;
        }
    }
    if (type & TONEDATA_TYPE_FIX)
        r8 = 0x800000;
    else
        r8 = ms->divFreq * chan->frequency;
    compressed = wav->type != 0;
    if (!compressed && !(type & TONEDATA_TYPE_REV))
    {
        // forward uncompressed with CMP set: nothing is mixed, count/pointer/fw
        // are stored back unchanged.
        sMixStats.channelsLogic++;
        return TRUE;
    }
    if ((r9 >> 23) != 0 || r8 > 0x3F800000u)
        return FALSE;
    total = (u64)r9 + (u64)n * r8;
    A = (u32)(total >> 23);
    r9out = (u32)total & 0x7FFFFF;
    if (A >= r2 && !(compressed && !(type & TONEDATA_TYPE_REV) && L != 0))
    {
        // Stops at the first sample k with (r9 + k * r8) >> 23 >= r2 (r8 > 0
        // here: A >= r2 > 0 with r9 < 2^23).
        u64 k = ((((u64)r2) << 23) - r9 + r8 - 1) / r8;
        Abefore = (u32)(((u64)r9 + (k - 1) * r8) >> 23);
        r9out = (u32)(((u64)r9 + k * r8) & 0x7FFFFF);
        stop = TRUE;
    }

    if (!compressed)
    {
        // _081DD468: uncompressed, reverse. Inside the loop r3 is pointer - 1 and
        // moves down by the advance; samples are read at r3 and r3 - 1.
        if (!SpanReadable((s64)r3 - 2 - A, (s64)r3 - 1))
            return FALSE;
        if (stop)
        {
            count = 0;
            ptr = r3 - 1 - Abefore;
        }
        else
        {
            count = r2 - A;
            ptr = r3 - A;
        }
        sMixStats.paths[4]++;
    }
    else
    {
        // Compressed: r3 is a sample index; DecodeSample reads the 33-byte block
        // index >> 6 at wav + 0x10.
        s64 ilo, ihi;
        if (!(type & TONEDATA_TYPE_REV))
        {
            // forward: index r3 + 1 inside the loop, moving up by the advance
            ilo = r3;
            ihi = (s64)r3 + 1 + A;
            path = 5;
            if (stop)
            {
                count = 0;
                ptr = r3 + 1 + Abefore;
                last = ptr;
            }
            else if (A < r2)
            {
                count = r2 - A;
                ptr = r3 + A;
                last = ptr + 1;
            }
            else
            {
                // loop (_081DD398): index = wav->loopStart + L - count + 1 after a wrap
                count = (u32)L - (A - r2) % (u32)L;
                ptr = wav->loopStart + (u32)L - count;
                last = ptr + 1;
                wrap = TRUE;
                if ((s64)wav->loopStart < ilo)
                    ilo = wav->loopStart;
                if ((s64)wav->loopStart + L > ihi)
                    ihi = (s64)wav->loopStart + L;
            }
        }
        else
        {
            // reverse (_081DD3C0): index r3 - 2 inside the loop, moving down
            ilo = (s64)r3 - 2 - A;
            ihi = (s64)r3 - 1;
            path = 7;
            if (stop)
            {
                count = 0;
                ptr = r3 - 2 - Abefore;
                last = ptr;
            }
            else
            {
                count = r2 - A;
                ptr = r3 - A;
                last = ptr - 2;
            }
        }
        if (ilo < 0 || ihi > 0xFFFFFFFFll)
            return FALSE;
        lo = (s64)(u32)wav + 0x10 + ((ilo >> 6) * 0x21);
        hi = (s64)(u32)wav + 0x10 + ((ihi >> 6) * 0x21) + 0x20;
        if (!SpanReadable(lo, hi))
            return FALSE;
        chan->xpi = 0;
        chan->xpc = 0xFF00;
        DecodeSample(chan, last);
        sMixStats.paths[path]++;
        if (wrap)
            sMixStats.paths[6]++;
    }
    if (stop)
    {
        chan->statusFlags = 0;
        sMixStats.channelsStopped++;
    }
    // _081DD228
    chan->fw = r9out;
    chan->count = count;
    chan->currentPointer = (s8 *)ptr;
    sMixStats.channelsLogic++;
    return TRUE;
}

// MixChannel in logic mode.
static void MixChannelLogic(struct SoundInfo *si, struct SoundChannel *chan, struct MixState *ms, u32 r6flags,
                            u32 r5env)
{
    struct WaveData *wav = chan->wav;
    u32 vol, r2, r3, r9, n = ms->samplesPerVBlank;
    s32 L;
    s64 lo, hi;

    PcmAliasCheck((u32)wav, 16, "wave header"); // read in every mix (also a channel set up without ply_note)
    // _081DD006 (as MixChannel)
    chan->envelopeVolume = (u8)r5env;
    vol = ((si->masterVolume + 1) * r5env) >> 4;
    chan->envelopeVolumeRight = (u8)((chan->rightVolume * vol) >> 8);
    chan->envelopeVolumeLeft = (u8)((chan->leftVolume * vol) >> 8);
    ms->loopLen = r6flags & SOUND_CHANNEL_SF_LOOP;
    if (ms->loopLen)
    {
        ms->loopStart = (s8 *)wav->data + wav->loopStart;
        ms->loopLen = wav->size - wav->loopStart;
    }
    L = ms->loopLen;
    r2 = chan->count;
    r3 = (u32)chan->currentPointer;
    r9 = chan->fw;
    if ((s32)r2 <= 0 || L < 0)
        goto full;

    if (chan->type & (TONEDATA_TYPE_CMP | TONEDATA_TYPE_REV))
    {
        if (MixSpecialLogic(chan, ms, r2, r3, r9))
            return;
        goto full;
    }

    if (chan->type & TONEDATA_TYPE_FIX)
    {
        // _081DD07C: one source sample per output sample, read at r3, r3 + 1, ...
        lo = r3;
        hi = (s64)r3 + n - 1;
        if (L != 0)
        {
            if ((s64)(u32)ms->loopStart < lo)
                lo = (u32)ms->loopStart;
            if ((s64)(u32)ms->loopStart + L - 1 > hi)
                hi = (s64)(u32)ms->loopStart + L - 1;
        }
        if (!SpanReadable(lo, hi))
            goto full;
        if (r2 > n)
        {
            chan->count = r2 - n;
            chan->currentPointer = (s8 *)(r3 + n);
        }
        else if (L == 0)
        {
            // _081DD174: count reached 0, no loop: count/pointer are not stored
            chan->statusFlags = 0;
            sMixStats.channelsStopped++;
        }
        else
        {
            u32 m = (n - r2) % (u32)L;
            chan->count = (u32)L - m;
            chan->currentPointer = (s8 *)((u32)ms->loopStart + m);
            sMixStats.paths[3]++;
        }
        sMixStats.paths[2]++;
        sMixStats.channelsLogic++;
        return;
    }

    {
        // _081DD19C: resampling; samples are read at r3 and r3 + 1 of the current
        // position (between the loop start and the loop end after a wrap).
        u32 step = ms->divFreq * chan->frequency;
        u64 total;
        u32 A;
        if ((r9 >> 23) != 0 || step > 0x3F800000u)
            goto full;
        total = (u64)r9 + (u64)n * step;
        A = (u32)(total >> 23);
        lo = r3;
        hi = (s64)r3 + A + 1;
        if (L != 0)
        {
            if ((s64)(u32)ms->loopStart < lo)
                lo = (u32)ms->loopStart;
            if ((s64)(u32)ms->loopStart + L > hi)
                hi = (s64)(u32)ms->loopStart + L;
        }
        if (!SpanReadable(lo, hi))
            goto full;
        if (A < r2)
        {
            chan->fw = (u32)total & 0x7FFFFF;
            chan->count = r2 - A;
            chan->currentPointer = (s8 *)(r3 + A);
        }
        else if (L == 0)
        {
            // _081DD158 -> _081DD174: stopped, count/pointer/fw are not stored
            chan->statusFlags = 0;
            sMixStats.channelsStopped++;
        }
        else
        {
            u32 count = (u32)L - (A - r2) % (u32)L;
            chan->fw = (u32)total & 0x7FFFFF;
            chan->count = count;
            chan->currentPointer = (s8 *)((u32)ms->loopStart + (u32)L - count);
            sMixStats.paths[1]++;
        }
        sMixStats.paths[0]++;
        sMixStats.channelsLogic++;
        return;
    }

full:
    // The full code path's sample reads (no exact span here): the wave's extent and,
    // conservatively, the samples one mix can consume on both sides of the current
    // pointer ((fw + K * step) >> 23, + 2 for the interpolation and the loop re-base).
    WaveAliasCheck(wav);
    {
        u64 step = (chan->type & TONEDATA_TYPE_FIX) ? 0x800000u : (u64)ms->divFreq * chan->frequency;
        u64 reach = (((u64)chan->fw + (u64)n * step) >> 23) + 2;
        PcmAliasCheck64((u32)((u64)(u32)chan->currentPointer - reach), 2 * reach + 1, "samples");
    }
    MixChannelToScratch(si, chan, ms, r6flags, r5env);
}

void Port_SetSoundMix(int mode)
{
    sSoundMix = (mode == PORT_SOUND_MIX_FULL) ? PORT_SOUND_MIX_FULL : PORT_SOUND_MIX_LOGIC;
}

int Port_GetSoundMix(void)
{
    return sSoundMix;
}

void Port_GetSoundMixStats(struct PortSoundMixStats *out)
{
    *out = sMixStats;
}

void Port_SoundPcmBuffer(uint32_t *addr, uint32_t *size)
{
    *addr = (u32)gSoundInfo.pcmBuffer;
    *size = sizeof(gSoundInfo.pcmBuffer);
}

// `logic`: logic-only mixing (see above): no PCM buffer clear/reverb, channels
// through MixChannelLogic.
static void SoundMainRAM_Mix(struct SoundInfo *si, struct MixState *ms, u32 pcmDmaCounter, u32 maxLinesEnd,
                             bool32 logic)
{
    u32 n = ms->samplesPerVBlank;
    u8 *r5 = ms->mixPos;
    struct SoundChannel *chan;
    s32 chansLeft;
    u32 r3 = si->reverb;

    if (logic)
    {
        // The PCM buffer is not touched (LogicMixAllowed).
    }
    else if (r3 != 0)
    {
        // SoundMainRAM_Reverb
        u8 *r7 = (pcmDmaCounter == 2) ? (u8 *)si->pcmBuffer : r5 + n;
        s32 i;
        for (i = (s32)n; i > 0; i--)
        {
            s32 r0 = (s8)r5[PCM_DMA_BUF_SIZE] + (s8)r5[0] + (s8)r7[PCM_DMA_BUF_SIZE] + (s8)r7[0];
            r0 = (r0 * (s32)r3) >> 9;
            if (r0 & 0x80)
                r0 += 1;
            r5[PCM_DMA_BUF_SIZE] = (u8)r0;
            r5[0] = (u8)r0;
            r5++;
            r7++;
        }
    }
    else
    {
        // SoundMainRAM_NoReverb: clears n bytes of both buffers (4-byte
        // stores; the loop body always runs at least once).
        u32 *a = (u32 *)r5;
        u32 *b = (u32 *)(r5 + PCM_DMA_BUF_SIZE);
        u32 r1 = n >> 3;
        if (n & 4)
        {
            *a++ = 0;
            *b++ = 0;
        }
        if (r1 & 1)
        {
            *a++ = 0; *b++ = 0;
            *a++ = 0; *b++ = 0;
        }
        r1 >>= 1;
        do
        {
            *a++ = 0; *b++ = 0;
            *a++ = 0; *b++ = 0;
            *a++ = 0; *b++ = 0;
            *a++ = 0; *b++ = 0;
        } while ((s32)--r1 > 0);
    }

    // _081DCF36
    ms->divFreq = si->divFreq;
    chansLeft = si->maxChans;
    chan = si->chans;
    for (;;)
    {
        u32 r6, r5env;
        struct WaveData *wav = chan->wav;

        // SoundMainRAM_ChanLoop: optional time budget in scanlines
        if (maxLinesEnd != 0)
        {
            u32 line = PORT_IO16(REG_OFFSET_VCOUNT) & 0xFF;
            if (line < 160)
                line += 228;
            if (line >= maxLinesEnd)
                break;
        }
        // _081DCF60
        r6 = chan->statusFlags;
        if (r6 & SOUND_CHANNEL_SF_ON)
        {
            if (r6 & SOUND_CHANNEL_SF_START)
            {
                if (r6 & SOUND_CHANNEL_SF_STOP)
                {
                    chan->statusFlags = 0;
                    goto next;
                }
                r6 = SOUND_CHANNEL_SF_ENV_ATTACK;
                chan->statusFlags = r6;
                chan->currentPointer = (s8 *)wav->data + chan->count;
                chan->count = wav->size - chan->count;
                r5env = 0;
                chan->envelopeVolume = 0;
                chan->fw = 0;
                if (wav->status & WAVE_DATA_STATUS_LOOP)
                {
                    r6 |= SOUND_CHANNEL_SF_LOOP;
                    chan->statusFlags = r6;
                }
                goto attack;
            }
            // _081DCFA0
            r5env = chan->envelopeVolume;
            if (r6 & SOUND_CHANNEL_SF_IEC)
            {
                u8 len = chan->pseudoEchoLength;
                chan->pseudoEchoLength = len - 1;
                if (len > 1)
                    goto mix;
                chan->statusFlags = 0; // _081DCFB0
                goto next;
            }
            if (r6 & SOUND_CHANNEL_SF_STOP)
            {
                r5env = (r5env * chan->release) >> 8;
                if (r5env > chan->pseudoEchoVolume)
                    goto mix;
            echo: // _081DCFC8
                r5env = chan->pseudoEchoVolume;
                if (r5env == 0)
                {
                    chan->statusFlags = 0;
                    goto next;
                }
                r6 |= SOUND_CHANNEL_SF_IEC;
                chan->statusFlags = r6;
                goto mix;
            }
            // _081DCFD6
            if ((r6 & SOUND_CHANNEL_SF_ENV) == SOUND_CHANNEL_SF_ENV_DECAY)
            {
                r5env = (r5env * chan->decay) >> 8;
                if (r5env > chan->sustain)
                    goto mix;
                r5env = chan->sustain;
                if (r5env == 0)
                    goto echo;
                r6 -= 1;
                chan->statusFlags = r6;
                goto mix;
            }
            if ((r6 & SOUND_CHANNEL_SF_ENV) != SOUND_CHANNEL_SF_ENV_ATTACK)
                goto mix;
        attack: // _081DCFF8
            r5env += chan->attack;
            if (r5env >= 0xFF)
            {
                r5env = 0xFF;
                r6 -= 1;
                chan->statusFlags = r6;
            }
        mix:
            if (logic)
                MixChannelLogic(si, chan, ms, r6, r5env);
            else
                MixChannel(si, chan, ms, r6, r5env);
        }
    next: // _081DD240
        if (--chansLeft <= 0)
            break;
        chan++;
    }
    // _081DD24A
    si->ident = ID_NUMBER;
}

// Bytes below SoundMain's stack pointer that the mixer may use: MixIsolated and
// its callees, at most 0x1C0 by gcc -fstack-usage of this file (deepest path
// MixIsolated -> SoundMainRAM_Mix -> MixChannelLogic -> MixChannelToScratch ->
// MixChannel -> MixSpecial -> DecodeSample; libgcc's 64-bit division is shallower),
// 0x190 measured over the scenario chains. The 16 bytes below the saved area are
// a canary: they must be unchanged after the mix.
#define MIX_STACK_BYTES 0x300
#define MIX_STACK_CANARY 16
HOST_STATE static u32 sMixStackSave[(MIX_STACK_BYTES + MIX_STACK_CANARY) / 4] ALIGNED(16);

static inline __attribute__((always_inline)) void CopyWordsInline(void *dst, const void *src, u32 words)
{
    __asm__ volatile("cld\n\trep movsl" : "+D"(dst), "+S"(src), "+c"(words) : : "memory", "cc");
}

// Caller-saved registers after the mixer are mode-dependent; clear them so
// that no stale value can reach game code (only through undefined behaviour,
// e.g. an uninitialized local kept in a register).
static inline __attribute__((always_inline)) void ClearScratchRegisters(void)
{
    __asm__ volatile("xorl %%eax, %%eax\n\txorl %%edx, %%edx\n\txorl %%ecx, %%ecx\n\t"
                     "pxor %%xmm0, %%xmm0\n\tpxor %%xmm1, %%xmm1\n\tpxor %%xmm2, %%xmm2\n\t"
                     "pxor %%xmm3, %%xmm3\n\tpxor %%xmm4, %%xmm4\n\tpxor %%xmm5, %%xmm5\n\t"
                     "pxor %%xmm6, %%xmm6\n\tpxor %%xmm7, %%xmm7"
                     : : : "eax", "edx", "ecx", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
                       "xmm7", "cc");
}

// The mix of one SoundMain call, in the mode of the process.
static __attribute__((noinline)) void MixIsolated(struct SoundInfo *si, struct MixState *ms, u32 pcmDmaCounter,
                                                 u32 maxLines)
{
    if (sSoundMix == PORT_SOUND_MIX_LOGIC && LogicMixAllowed(si, ms, pcmDmaCounter))
    {
        sMixStats.mixesLogic++;
        SoundMainRAM_Mix(si, ms, pcmDmaCounter, maxLines, TRUE);
    }
    else
    {
        sMixStats.mixesFull++;
        SoundMainRAM_Mix(si, ms, pcmDmaCounter, maxLines, FALSE);
    }
}

void SoundMain(void)
{
    struct SoundInfo *si = SOUND_INFO_PTR;
    struct MixState ms;
    u32 maxLines;
    u8 *r5;
    u32 r4;

    if (si->ident != ID_NUMBER)
        return;
    si->ident++;

    maxLines = si->maxLines;
    if (maxLines != 0)
    {
        u32 line = PORT_IO16(REG_OFFSET_VCOUNT) & 0xFF;
        if (line < 160)
            line += 228;
        maxLines += line;
    }
    if (si->MPlayMainHead != NULL)
        ((MPlayMainFunc)si->MPlayMainHead)(si->musicPlayerHead);
    si->CgbSound();

    ms.samplesPerVBlank = si->pcmSamplesPerVBlank;
    r5 = (u8 *)si->pcmBuffer;
    r4 = si->pcmDmaCounter;
    if (r4 > 1)
        r5 += (si->pcmDmaPeriod - (r4 - 1)) * ms.samplesPerVBlank;
    ms.mixPos = r5;
    ms.loopStart = NULL;
    ms.loopLen = 0;
    // The mixer runs as on the GBA's separate IRQ stack: the game-stack bytes its
    // frames use are restored afterwards, so that it leaves no residue there (the
    // game stack is environment state, and game code reads stale stack bytes
    // through uninitialized locals). Both mixer modes therefore leave identical
    // stacks. Everything that depends on the mode happens inside MixIsolated,
    // whose arguments are mode-independent; no call between the save and the
    // restore except that one (the copies are inline).
    {
        u32 sp, lo;
        __asm__ volatile("movl %%esp, %0" : "=r"(sp));
        lo = ((sp - MIX_STACK_BYTES) & ~15u) - MIX_STACK_CANARY;
        if (Port_IsOnGameStack((const void *)lo) && Port_IsOnGameStack((const void *)(sp - 1)))
        {
            const u32 *canary = (const u32 *)lo;
            CopyWordsInline(sMixStackSave, (const void *)lo, (MIX_STACK_BYTES + MIX_STACK_CANARY) / 4);
            MixIsolated(si, &ms, r4, maxLines);
            if ((canary[0] ^ sMixStackSave[0]) | (canary[1] ^ sMixStackSave[1]) | (canary[2] ^ sMixStackSave[2])
                | (canary[3] ^ sMixStackSave[3]))
                Port_Fatal("sound mixer: stack deeper than MIX_STACK_BYTES (platform/sound.c)");
            CopyWordsInline((void *)(lo + MIX_STACK_CANARY), sMixStackSave + MIX_STACK_CANARY / 4,
                            MIX_STACK_BYTES / 4);
            ClearScratchRegisters();
        }
        else
        {
            if (Port_IsOnGameStack((const void *)(sp - 1)))
                Port_Fatal("game stack overflow (sound mixer)");
            MixIsolated(si, &ms, r4, maxLines); // host stack (tests)
        }
    }
}

void Port_TestSoundMix(int mode)
{
    struct SoundInfo *si = &gSoundInfo;
    struct MixState ms;
    u8 saved = sSoundMix;
    u8 *r5;
    u32 r4;

    ms.samplesPerVBlank = si->pcmSamplesPerVBlank;
    r5 = (u8 *)si->pcmBuffer;
    r4 = si->pcmDmaCounter;
    if (r4 > 1)
        r5 += (si->pcmDmaPeriod - (r4 - 1)) * ms.samplesPerVBlank;
    ms.mixPos = r5;
    ms.loopStart = NULL;
    ms.loopLen = 0;
    sSoundMix = mode == PORT_SOUND_MIX_FULL ? PORT_SOUND_MIX_FULL : PORT_SOUND_MIX_LOGIC;
    MixIsolated(si, &ms, r4, 0);
    sSoundMix = saved;
}

// SoundMainBTM: clears 64 bytes (used as Clear64byte through the jump table).
void SoundMainBTM(void *x)
{
    memset(x, 0, 64);
}

// ---------------------------------------------------------------------------
// Channel list handling

void RealClearChain(void *x)
{
    struct SoundChannel *chan = x;
    struct MusicPlayerTrack *track = chan->track;
    if (track != NULL)
    {
        struct SoundChannel *next = chan->nextChannelPointer;
        struct SoundChannel *prev = chan->prevChannelPointer;
        if (prev != NULL)
            prev->nextChannelPointer = next;
        else
            track->chan = next;
        if (next != NULL)
            next->prevChannelPointer = prev;
        chan->track = NULL;
    }
}

// ---------------------------------------------------------------------------
// Command stream access (ld_r3_tp_adr_i, chk_adr_r2)

// chk_adr_r2: reads from the BIOS area are rejected (value 0) except for the
// jump table template. Native addresses are never below 0x2000000, so only
// stray pointers are affected; they read 0 instead of crashing.
static inline bool32 AddrOk(u32 addr)
{
    if ((addr >> 25) != 0)
        return TRUE;
    if (addr < (u32)gMPlayJumpTableTemplate)
        return FALSE;
    return (addr >> 14) == 0;
}

static inline u32 ReadCmdChecked(struct MusicPlayerTrack *track)
{
    u8 *p = track->cmdPtr;
    track->cmdPtr = p + 1;
    return AddrOk((u32)p) ? *p : 0;
}

void MPlayJumpTableCopy(MPlayFunc *mplayJumpTable)
{
    u32 i;
    for (i = 0; i < 0x24; i++)
    {
        void *const *src = &gMPlayJumpTableTemplate[i];
        mplayJumpTable[i] = AddrOk((u32)src) ? (MPlayFunc)*src : (MPlayFunc)NULL;
    }
}

// ---------------------------------------------------------------------------
// ply_* commands

void ply_fine(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    struct SoundChannel *chan = track->chan;
    while (chan != NULL)
    {
        if (chan->statusFlags & SOUND_CHANNEL_SF_ON)
            chan->statusFlags |= SOUND_CHANNEL_SF_STOP;
        RealClearChain(chan);
        chan = chan->nextChannelPointer;
    }
    track->flags = 0;
}

void ply_goto(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u8 *p = track->cmdPtr;
    u32 v = ((u32)p[3] << 24) | ((u32)p[2] << 16) | ((u32)p[1] << 8);
    v |= AddrOk((u32)p) ? p[0] : 0; // ldrb_r3_r2 + chk_adr_r2 on the low byte
    track->cmdPtr = (u8 *)v;
}

void ply_patt(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u8 level = track->patternLevel;
    if (level >= 3)
    {
        ply_fine(mplayInfo, track);
        return;
    }
    track->patternStack[level] = track->cmdPtr + 4;
    track->patternLevel = level + 1;
    ply_goto(mplayInfo, track);
}

void ply_pend(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    if (track->patternLevel != 0)
    {
        u8 level = track->patternLevel - 1;
        track->patternLevel = level;
        track->cmdPtr = track->patternStack[level];
    }
}

void ply_rept(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u8 *p = track->cmdPtr;
    if (*p == 0)
    {
        track->cmdPtr = p + 1;
        ply_goto(mplayInfo, track);
        return;
    }
    {
        u8 n = track->repN + 1;
        u32 target;
        track->repN = n;
        target = ReadCmdChecked(track); // r2 = old cmdPtr, cmdPtr += 1
        if (n < target)
        {
            ply_goto(mplayInfo, track);
        }
        else
        {
            track->repN = 0;
            track->cmdPtr = p + 5;
        }
    }
}

void ply_prio(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->priority = ReadCmdChecked(track);
}

void ply_tempo(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u32 t = ReadCmdChecked(track) << 1;
    mplayInfo->tempoD = t;
    mplayInfo->tempoI = (u16)((t * mplayInfo->tempoU) >> 8);
}

void ply_keysh(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->keyShift = ReadCmdChecked(track);
    track->flags |= 0xC;
}

void ply_voice(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u8 *p = track->cmdPtr;
    u32 voice = *p;
    const u32 *tone;
    u32 *dst = (u32 *)&track->tone;

    track->cmdPtr = p + 1;
    tone = (const u32 *)((u32)mplayInfo->tone + voice * 12);
    PcmAliasCheck((u32)tone, 12, "voice");
    // Three word copies, each checked with chk_adr_r2 (r2 = source address).
    dst[0] = AddrOk((u32)&tone[0]) ? tone[0] : 0;
    dst[1] = AddrOk((u32)&tone[1]) ? tone[1] : 0;
    dst[2] = AddrOk((u32)&tone[2]) ? tone[2] : 0;
}

void ply_vol(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->vol = ReadCmdChecked(track);
    track->flags |= MPT_FLG_VOLCHG;
}

void ply_pan(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->pan = (s8)(ReadCmdChecked(track) - C_V);
    track->flags |= MPT_FLG_VOLCHG;
}

void ply_bend(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->bend = (s8)(ReadCmdChecked(track) - C_V);
    track->flags |= MPT_FLG_PITCHG;
}

void ply_bendr(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->bendRange = ReadCmdChecked(track);
    track->flags |= MPT_FLG_PITCHG;
}

void ply_lfodl(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->lfoDelay = ReadCmdChecked(track);
}

void ply_modt(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u32 t = ReadCmdChecked(track);
    if (track->modT != t)
    {
        track->modT = t;
        track->flags |= MPT_FLG_VOLCHG | MPT_FLG_PITCHG;
    }
}

void ply_tune(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    track->tune = (s8)(ReadCmdChecked(track) - C_V);
    track->flags |= MPT_FLG_PITCHG;
}

void ply_port(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    // Writes the next byte to a sound register: REG_SOUND1CNT_L + offset.
    u8 *p = track->cmdPtr;
    u32 off = *p;
    u32 v;
    track->cmdPtr = p + 1;
    v = ReadCmdChecked(track);
    *(vu8 *)(REG_BASE + REG_OFFSET_SOUND1CNT_L + off) = (u8)v;
}

static void clear_modM(struct MusicPlayerTrack *track)
{
    track->modM = 0;
    track->lfoSpeedC = 0;
    track->flags |= (track->modT == 0) ? MPT_FLG_PITCHG : MPT_FLG_VOLCHG;
}

void ply_lfos(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u8 *p = track->cmdPtr; // ld_r3_tp_adr_i_unchecked
    track->cmdPtr = p + 1;
    track->lfoSpeed = *p;
    if (*p == 0)
        clear_modM(track);
}

void ply_mod(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u8 *p = track->cmdPtr;
    track->cmdPtr = p + 1;
    track->mod = *p;
    if (*p == 0)
        clear_modM(track);
}

void ply_endtie(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    u8 *p = track->cmdPtr;
    u32 key = *p;
    struct SoundChannel *chan;

    if (key < 0x80)
    {
        track->key = key;
        track->cmdPtr = p + 1;
    }
    else
    {
        key = track->key;
    }
    for (chan = track->chan; chan != NULL; chan = chan->nextChannelPointer)
    {
        u32 f = chan->statusFlags;
        if ((f & (SOUND_CHANNEL_SF_START | SOUND_CHANNEL_SF_ENV)) && !(f & SOUND_CHANNEL_SF_STOP)
         && chan->midiKey == key)
        {
            chan->statusFlags = f | SOUND_CHANNEL_SF_STOP;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// TrackStop, ChnVolSetAsm, ply_note

void TrackStop(struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    struct SoundChannel *chan;
    if (!(track->flags & MPT_FLG_EXIST))
        return;
    chan = track->chan;
    while (chan != NULL)
    {
        if (chan->statusFlags != 0)
        {
            u8 cgb = chan->type & TONEDATA_TYPE_CGB;
            if (cgb)
                ((CgbOscOffF)SOUND_INFO_PTR->CgbOscOff)(cgb);
            chan->statusFlags = 0;
        }
        chan->track = NULL;
        chan = chan->nextChannelPointer;
    }
    track->chan = NULL;
}

static void ChnVolSetAsm(struct SoundChannel *chan, struct MusicPlayerTrack *track)
{
    s32 velocity = chan->velocity;
    s32 pan = (s8)chan->rhythmPan;
    s32 v;

    v = (((0x80 + pan) * velocity) * track->volMR) >> 14;
    if ((u32)v > 0xFF)
        v = 0xFF;
    chan->rightVolume = v;
    v = (((0x7F - pan) * velocity) * track->volML) >> 14;
    if ((u32)v > 0xFF)
        v = 0xFF;
    chan->leftVolume = v;
}

void ply_note(u32 note_cmd, struct MusicPlayerInfo *mplayInfo, struct MusicPlayerTrack *track)
{
    struct SoundInfo *si = SOUND_INFO_PTR;
    u8 *p;
    u32 key;          // [sp+0x8]
    s32 rhythmPan;    // [sp+0x14]
    u32 priority;     // [sp+0x10]
    u32 cgbType;      // [sp+0xC]
    struct ToneData *tone; // r9
    struct SoundChannel *chan;
    struct WaveData *wav;
    s32 transposed;

    track->gateTime = gClockTable[note_cmd];
    p = track->cmdPtr;
    if (*p < 0x80)
    {
        track->key = *p;
        p++;
        if (*p < 0x80)
        {
            track->velocity = *p;
            p++;
            if (*p < 0x80)
            {
                track->gateTime += *p;
                p++;
            }
        }
        track->cmdPtr = p;
    }

    rhythmPan = 0;
    if (track->tone.type & (TONEDATA_TYPE_RHY | TONEDATA_TYPE_SPL))
    {
        u32 k = track->key;
        u32 idx;
        if (track->tone.type & TONEDATA_TYPE_SPL)
        {
            const u8 *keySplit = *(const u8 *const *)&track->tone.attack; // keySplitTable aliases attack..release
            PcmAliasCheck((u32)keySplit + k, 1, "key split table");
            idx = keySplit[k];
        }
        else
            idx = k;
        tone = (struct ToneData *)((u32)track->tone.wav + idx * 12);
        PcmAliasCheck((u32)tone, 12, "tone");
        if (tone->type & (TONEDATA_TYPE_SPL | TONEDATA_TYPE_RHY))
            return;
        if (track->tone.type & TONEDATA_TYPE_RHY)
        {
            u32 ps = tone->pan_sweep;
            if (ps & 0x80)
                rhythmPan = (s32)(ps - TONEDATA_P_S_PAN) << 1;
            k = tone->key;
        }
        key = k;
    }
    else
    {
        tone = &track->tone;
        key = track->key;
    }

    priority = mplayInfo->priority + track->priority;
    if (priority > 0xFF)
        priority = 0xFF;

    cgbType = tone->type & TONEDATA_TYPE_CGB;
    if (cgbType != 0)
    {
        if (si->cgbChans == NULL)
            return;
        chan = (struct SoundChannel *)((u32)si->cgbChans + ((cgbType - 1) << 6));
        if ((chan->statusFlags & SOUND_CHANNEL_SF_ON) && !(chan->statusFlags & SOUND_CHANNEL_SF_STOP))
        {
            if (chan->priority > priority)
                return;
            if (chan->priority == priority && (u32)chan->track < (u32)track)
                return;
        }
    }
    else
    {
        // _081DDBEC: choose a direct-sound channel
        u32 p6 = priority;
        u32 r7 = (u32)track;
        u32 r2 = 0; // found a stopping channel
        struct SoundChannel *best = NULL;
        struct SoundChannel *c = si->chans;
        s32 n = si->maxChans;

        for (;;)
        {
            u32 f = c->statusFlags;
            if (!(f & SOUND_CHANNEL_SF_ON))
            {
                best = c;
                goto found;
            }
            if (f & SOUND_CHANNEL_SF_STOP)
            {
                if (r2 == 0)
                {
                    r2 = 1;
                    p6 = c->priority;
                    r7 = (u32)c->track;
                    best = c;
                    goto cont;
                }
            }
            else if (r2 != 0)
            {
                goto cont;
            }
            // _081DDC18
            if (c->priority < p6)
            {
                p6 = c->priority;
                r7 = (u32)c->track;
                best = c;
            }
            else if (c->priority == p6)
            {
                if ((u32)c->track > r7)
                {
                    r7 = (u32)c->track;
                    best = c;
                }
                else if ((u32)c->track == r7)
                {
                    best = c;
                }
            }
        cont:
            c++;
            if (--n <= 0)
                break;
        }
        if (best == NULL)
            return;
    found:
        chan = best;
    }

    // _081DDC40
    RealClearChain(chan);
    chan->prevChannelPointer = NULL;
    chan->nextChannelPointer = track->chan;
    if (track->chan != NULL)
        ((struct SoundChannel *)track->chan)->prevChannelPointer = chan;
    track->chan = chan;
    chan->track = track;
    track->lfoDelayC = track->lfoDelay;
    if (track->lfoDelay != 0)
        clear_modM(track);
    TrkVolPitSet(mplayInfo, track);

    // ldr/str of 4 bytes: gateTime, key, velocity, runningStatus -> gateTime,
    // midiKey, velocity, priority (the last one is overwritten just below).
    memcpy(&chan->gateTime, &track->gateTime, 4);
    chan->priority = priority;
    chan->key = key;
    chan->rhythmPan = rhythmPan;
    chan->type = tone->type;
    wav = tone->wav;
    chan->wav = wav;
    if (cgbType == 3)
        PcmAliasCheck((u32)wav, 16, "CGB wave"); // CgbSound copies 16 bytes to the wave RAM
    else if (cgbType == 0)
        WaveAliasCheck(wav);
    memcpy(&chan->attack, &tone->attack, 4); // attack, decay, sustain, release
    memcpy(&chan->pseudoEchoVolume, &track->pseudoEchoVolume, 2); // + pseudoEchoLength
    ChnVolSetAsm(chan, track);

    transposed = (s32)chan->key + (s8)track->keyM;
    if (transposed < 0)
        transposed = 0;

    if (cgbType != 0)
    {
        struct CgbChannel *cgb = (struct CgbChannel *)chan;
        u32 ps = tone->pan_sweep;
        cgb->length = tone->length;
        if ((ps & 0x80) || !(ps & 0x70))
            ps = 8;
        cgb->sweep = ps;
        chan->frequency = ((MidiKeyToCgbFreqF)si->MidiKeyToCgbFreq)(cgbType, transposed, track->pitM);
    }
    else
    {
        chan->count = track->unk_3C;
        chan->frequency = MidiKeyToFreq(wav, transposed, track->pitM);
    }
    chan->statusFlags = SOUND_CHANNEL_SF_START;
    track->flags &= 0xF0;
}

// ---------------------------------------------------------------------------
// MPlayMain

void MPlayMain(struct MusicPlayerInfo *mplayInfo)
{
    struct SoundInfo *si;
    struct MusicPlayerTrack *track;
    s32 trackCount;
    u32 bit, active;
    u32 tempoC;

    if (mplayInfo->ident != ID_NUMBER)
        return;
    mplayInfo->ident = ID_NUMBER + 1;

    // The players form a chain: run the next one first (MPlayOpen inserted
    // every new player at the head, so the first opened runs first).
    if (mplayInfo->MPlayMainNext != NULL)
        ((MPlayMainFunc)mplayInfo->MPlayMainNext)(mplayInfo->musicPlayerNext);

    if ((s32)mplayInfo->status < 0)
        goto done;
    si = SOUND_INFO_PTR;
    FadeOutBody(mplayInfo);
    if ((s32)mplayInfo->status < 0)
        goto done;

    tempoC = (u32)mplayInfo->tempoC + mplayInfo->tempoI;
    for (;;)
    {
        mplayInfo->tempoC = (u16)tempoC; // _081DD9BC
        if ((tempoC & 0xFFFF) < 150)
            break;

        // _081DD874: one tick
        trackCount = mplayInfo->trackCount;
        track = mplayInfo->tracks;
        bit = 1;
        active = 0;
        for (;;)
        {
            if (track->flags & MPT_FLG_EXIST)
            {
                struct SoundChannel *chan;
                active |= bit;
                for (chan = track->chan; chan != NULL; chan = chan->nextChannelPointer)
                {
                    u32 f = chan->statusFlags;
                    if (f & SOUND_CHANNEL_SF_ON)
                    {
                        if (chan->gateTime != 0 && --chan->gateTime == 0)
                            chan->statusFlags = f | SOUND_CHANNEL_SF_STOP;
                    }
                    else
                    {
                        ClearChain(chan);
                    }
                }
                if (track->flags & MPT_FLG_START)
                {
                    Clear64byte(track);
                    track->flags = MPT_FLG_EXIST;
                    track->bendRange = 2;
                    track->volX = 0x40;
                    track->lfoSpeed = 0x16;
                    track->tone.type = 1;
                }
                // _081DD938
                while (track->wait == 0)
                {
                    u8 *p = track->cmdPtr;
                    u32 cmd;
                    PcmAliasCheck((u32)p, 8, "command"); // a command and its arguments (memacc + goto: 8 bytes)
                    cmd = *p;
                    if (cmd < 0x80)
                    {
                        cmd = track->runningStatus;
                    }
                    else
                    {
                        track->cmdPtr = p + 1;
                        if (cmd >= 0xBD)
                            track->runningStatus = cmd;
                    }
                    if (cmd >= 0xCF)
                    {
                        ((PlyNoteFunc)si->plynote)(cmd - 0xCF, mplayInfo, track);
                    }
                    else if (cmd > 0xB0)
                    {
                        mplayInfo->cmd = cmd - 0xB1;
                        ((TrackFunc)si->MPlayJumpTable[cmd - 0xB1])(mplayInfo, track);
                        if (track->flags == 0)
                            goto next_track;
                    }
                    else
                    {
                        track->wait = gClockTable[cmd - 0x80];
                    }
                }
                track->wait--;
                if (track->lfoSpeed != 0 && track->mod != 0)
                {
                    if (track->lfoDelayC != 0)
                    {
                        track->lfoDelayC--;
                    }
                    else
                    {
                        // _081DD95A: triangle LFO; r1 keeps the full 9-bit sum
                        u32 sum = (u32)track->lfoSpeedC + track->lfoSpeed;
                        s32 r2;
                        track->lfoSpeedC = (u8)sum;
                        if (((sum - 0x40) & 0x80) == 0)
                            r2 = 0x80 - (s32)sum;
                        else
                            r2 = (s8)sum;
                        r2 = ((s32)track->mod * r2) >> 6;
                        if ((u8)(track->modM ^ r2) != 0)
                        {
                            track->modM = (u8)r2;
                            track->flags |= (track->modT == 0) ? MPT_FLG_PITCHG : MPT_FLG_VOLCHG;
                        }
                    }
                }
            }
        next_track: // _081DD998
            if (--trackCount <= 0)
                break;
            track++;
            bit <<= 1;
        }
        // _081DD9A4
        mplayInfo->clock++;
        if (active == 0)
        {
            mplayInfo->status = MUSICPLAYER_STATUS_PAUSE;
            goto done;
        }
        mplayInfo->status = active;
        tempoC = (u32)mplayInfo->tempoC - 150;
    }

    // _081DD9C4: apply volume/pitch changes to the channels
    trackCount = mplayInfo->trackCount;
    track = mplayInfo->tracks;
    for (;;)
    {
        u32 f = track->flags;
        if ((f & 0x80) && (f & (MPT_FLG_VOLCHG | MPT_FLG_PITCHG)))
        {
            struct SoundChannel *chan;
            TrkVolPitSet(mplayInfo, track);
            for (chan = track->chan; chan != NULL; chan = chan->nextChannelPointer)
            {
                u32 cgbType;
                if (!(chan->statusFlags & SOUND_CHANNEL_SF_ON))
                {
                    ClearChain(chan);
                    continue;
                }
                cgbType = chan->type & TONEDATA_TYPE_CGB;
                if (track->flags & MPT_FLG_VOLCHG)
                {
                    ChnVolSetAsm(chan, track);
                    if (cgbType)
                        ((struct CgbChannel *)chan)->modify |= CGB_CHANNEL_MO_VOL;
                }
                if (track->flags & MPT_FLG_PITCHG)
                {
                    s32 k = (s32)chan->key + (s8)track->keyM;
                    if (k < 0)
                        k = 0;
                    if (cgbType)
                    {
                        ((struct CgbChannel *)chan)->frequency =
                            ((MidiKeyToCgbFreqF)si->MidiKeyToCgbFreq)(cgbType, k, track->pitM);
                        ((struct CgbChannel *)chan)->modify |= CGB_CHANNEL_MO_PIT;
                    }
                    else
                    {
                        chan->frequency = MidiKeyToFreq(chan->wav, k, track->pitM);
                    }
                }
            }
            track->flags &= 0xF0;
        }
        if (--trackCount <= 0)
            break;
        track++;
    }

done: // _081DDA6C
    mplayInfo->ident = ID_NUMBER;
}

// ---------------------------------------------------------------------------
// m4aSoundVSync (VCountIntr): restarts the sound FIFO DMAs at the end of the
// PCM buffer.

void m4aSoundVSync(void)
{
    struct SoundInfo *si = SOUND_INFO_PTR;
    u8 counter;

    if (si->ident - ID_NUMBER > 1)
        return;
    counter = si->pcmDmaCounter - 1;
    si->pcmDmaCounter = counter;
    if ((s8)counter > 0)
        return;
    si->pcmDmaCounter = si->pcmDmaPeriod;

    if (REG_DMA1CNT & (DMA_REPEAT << 16))
        REG_DMA1CNT = ((DMA_ENABLE | DMA_START_NOW | DMA_32BIT | DMA_SRC_INC | DMA_DEST_FIXED) << 16) | 4;
    if (REG_DMA2CNT & (DMA_REPEAT << 16))
        REG_DMA2CNT = ((DMA_ENABLE | DMA_START_NOW | DMA_32BIT | DMA_SRC_INC | DMA_DEST_FIXED) << 16) | 4;
    REG_DMA1CNT_H = DMA_32BIT;
    REG_DMA2CNT_H = DMA_32BIT;
    REG_DMA1CNT_H = DMA_ENABLE | DMA_START_SPECIAL | DMA_32BIT | DMA_REPEAT;
    REG_DMA2CNT_H = DMA_ENABLE | DMA_START_SPECIAL | DMA_32BIT | DMA_REPEAT;
}

// ---------------------------------------------------------------------------

void Port_SoundPowerOn(void)
{
    if (!sSegDone)
        FindReadableSegments();
    // SoundMainRAM keeps a 2-bit counter in the top bits of the PCM buffer
    // address; the buffer must therefore lie below 0x40000000.
    if ((u32)&gSoundInfo >= 0x40000000u)
        Port_Fatal("gSoundInfo at %p: the PCM mixer needs addresses below 0x40000000", (void *)&gSoundInfo);
}
