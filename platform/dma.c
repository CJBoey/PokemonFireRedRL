// fireredsim platform layer: the four DMA channels.
//
// Reference: mGBA src/gba/dma.c. Writing the control register with the enable
// bit going 0->1 latches source/destination/count. "Immediate" transfers run
// synchronously inside Port_DmaSet; VBlank transfers run at the VBlank point of
// the frame (Port_DmaRunVBlank); HBlank transfers (scanline effects) and the
// sound FIFO transfers (special timing) only change video/audio output and are
// not executed in state mode (docs/HBLANK_INVENTORY.md). Their register state
// is kept so that a renderer can reproduce them.
#include "port_internal.h"

struct PortDma
{
    u32 nextSource;
    u32 nextDest;
    u32 count;        // units per transfer (CNT_L, 0 means 0x4000 / 0x10000)
};

static struct PortDma sDma[4];
static u32 sDmaTransferRegister; // last value moved (sources of address 0 reuse it)

#define DMA_REG_SAD(n)   (REG_OFFSET_DMA0SAD + 12 * (n))
#define DMA_REG_DAD(n)   (REG_OFFSET_DMA0DAD + 12 * (n))
#define DMA_REG_CNT_L(n) (REG_OFFSET_DMA0CNT_L + 12 * (n))
#define DMA_REG_CNT_H(n) (REG_OFFSET_DMA0CNT_H + 12 * (n))

#define DMA_TIMING(cnt_h) (((cnt_h) >> 12) & 3)
#define DMA_TIMING_NOW     0
#define DMA_TIMING_VBLANK  1
#define DMA_TIMING_HBLANK  2
#define DMA_TIMING_SPECIAL 3

static const s32 sDmaOffset[] = { 1, -1, 0, 1 };

void Port_DmaPowerOn(void)
{
    u32 i;
    for (i = 0; i < 4; i++)
    {
        sDma[i].nextSource = 0;
        sDma[i].nextDest = 0;
        sDma[i].count = (i == 3) ? 0x10000 : 0x4000;
    }
    sDmaTransferRegister = 0;
}

// Transfers `info->count` units with the channel's current control bits.
static void DmaService(u32 n, u16 cntH)
{
    struct PortDma *info = &sDma[n];
    u32 width = (cntH & DMA_32BIT) ? 4 : 2;
    s32 sourceOffset = sDmaOffset[(cntH >> 7) & 3] * (s32)width;
    s32 destOffset = sDmaOffset[(cntH >> 5) & 3] * (s32)width;
    u32 source = info->nextSource;
    u32 dest = info->nextDest;
    u32 units = info->count;

    if (width == 4 && sourceOffset == 4 && destOffset == 4 && source)
    {
        // Common case: incrementing word copy (memmove gives the forward-copy
        // result unless the destination lies inside the source range ahead).
        if (!(dest > source && dest < source + units * 4))
        {
            memmove((void *)dest, (const void *)source, units * 4);
            sDmaTransferRegister = *(const u32 *)(source + (units - 1) * 4);
            source += units * 4;
            dest += units * 4;
            units = 0;
        }
    }
    else if (width == 2 && sourceOffset == 2 && destOffset == 2 && source)
    {
        if (!(dest > source && dest < source + units * 2))
        {
            u16 last;
            memmove((void *)dest, (const void *)source, units * 2);
            last = *(const u16 *)(source + (units - 1) * 2);
            sDmaTransferRegister = last | ((u32)last << 16);
            source += units * 2;
            dest += units * 2;
            units = 0;
        }
    }
    while (units)
    {
        if (width == 4)
        {
            if (source)
                sDmaTransferRegister = *(const u32 *)source;
            *(u32 *)dest = sDmaTransferRegister;
        }
        else
        {
            if (source)
            {
                sDmaTransferRegister = *(const u16 *)source;
                sDmaTransferRegister |= sDmaTransferRegister << 16;
            }
            *(u16 *)dest = (u16)sDmaTransferRegister;
        }
        if (source)
            source += sourceOffset;
        dest += destOffset;
        units--;
    }
    info->nextSource = source;
    info->nextDest = dest;
}

// End of a transfer: clear the enable bit unless repeating, reload the
// destination for "increment/reload".
static void DmaFinish(u32 n, u16 cntH)
{
    bool32 noRepeat = !(cntH & DMA_REPEAT) || DMA_TIMING(cntH) == DMA_TIMING_NOW;
    if (noRepeat)
        PORT_IO16(DMA_REG_CNT_H(n)) &= 0x7FE0;
    if (((cntH >> 5) & 3) == 3)
        sDma[n].nextDest = PORT_IO32(DMA_REG_DAD(n)) & ~1u;
}

void Port_DmaSet(u32 n, u32 src, u32 dest, u32 control)
{
    // The DmaSet macro is three 32-bit stores: SAD, DAD, CNT (CNT_L + CNT_H).
    u16 oldCntH = PORT_IO16(DMA_REG_CNT_H(n));
    u16 cntL = control & 0xFFFF;
    u16 cntH = control >> 16;
    u32 width;

    if (n > 3)
        Port_Fatal("DmaSet: bad channel %u", (unsigned)n);
    PORT_IO32(DMA_REG_SAD(n)) = src;
    PORT_IO32(DMA_REG_DAD(n)) = dest;
    PORT_IO16(DMA_REG_CNT_L(n)) = cntL;
    sDma[n].count = cntL ? cntL : (n == 3 ? 0x10000 : 0x4000);
    cntH &= (n < 3) ? 0xF7E0 : 0xFFE0;
    PORT_IO16(DMA_REG_CNT_H(n)) = cntH;

    if ((oldCntH & DMA_ENABLE) || !(cntH & DMA_ENABLE))
        return;

    // Enable 0->1: latch the addresses, aligned to the transfer width. A source
    // below EWRAM (BIOS/unmapped, e.g. NULL) is invalid: mGBA latches 0 and
    // the transfer repeats the last value on the bus (sDmaTransferRegister).
    // Note: the GBA always increments a source in cartridge ROM even when
    // "fixed"/"decrement" is selected; FireRed never DMAs that way.
    width = (cntH & DMA_32BIT) ? 4 : 2;
    sDma[n].nextSource = (src < 0x02000000) ? 0 : (src & -width);
    sDma[n].nextDest = dest & -width;

    switch (DMA_TIMING(cntH))
    {
    case DMA_TIMING_NOW:
        DmaService(n, cntH);
        DmaFinish(n, cntH);
        break;
    case DMA_TIMING_VBLANK:
    case DMA_TIMING_HBLANK:
    case DMA_TIMING_SPECIAL:
        break; // triggered later (VBlank) or presentation-only (HBlank, FIFO)
    }
}

void Port_DmaRunVBlank(void)
{
    u32 n;
    for (n = 0; n < 4; n++)
    {
        u16 cntH = PORT_IO16(DMA_REG_CNT_H(n));
        if ((cntH & DMA_ENABLE) && DMA_TIMING(cntH) == DMA_TIMING_VBLANK)
        {
            DmaService(n, cntH);
            DmaFinish(n, cntH);
        }
    }
}
