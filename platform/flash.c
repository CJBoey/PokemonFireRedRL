// fireredsim platform layer: the 1 Mbit (128 KiB) flash save chip.
//
// The chip's contents are plain memory at 0x0E000000..0x0E01FFFF (both 64 KiB
// banks back to back; mirror layout, see FLASH_BASE in gba/flash_internal.h),
// one of the regions of an environment's state, so saves are per environment
// and are carried by snapshots. The chip identifies itself like mGBA's FLASH1M chip
// (manufacturer Sanyo, ID 0x1362), so IdentifyFlash() selects the same
// LE26FV10N1TS descriptor as on the oracle. Erase and program complete
// instantly; on hardware they take milliseconds (lag frames during saving, see
// docs/DEVLOG.md) and the game polls the status until they finish.
#include "port_internal.h"

#define FLASH_CHIP ((u8 *)PORT_FLASH_ADDR) // the whole chip (FLASH_BASE adds the bank offset)
#define FLASH_CHIP_SIZE PORT_FLASH_SIZE
unsigned char gPortFlashBank;
static u8 sFlashCmdState;

#define FLASH_ID_SANYO 0x1362

void Port_FlashPowerOn(const u8 *image)
{
    if (image)
        memcpy(FLASH_CHIP, image, FLASH_CHIP_SIZE);
    else
        memset(FLASH_CHIP, 0xFF, FLASH_CHIP_SIZE);
    gPortFlashBank = 0;
    sFlashCmdState = 0;
}

void Port_ReadFlashImage(uint8_t *out)
{
    memcpy(out, FLASH_CHIP, FLASH_CHIP_SIZE);
}

u16 Port_FlashReadId(void)
{
    return FLASH_ID_SANYO;
}

// Command writes through FLASH_WRITE. With the chip operations handled by the
// Port_Flash* functions below, the only sequences the library still issues are
// the bank switch (AA/55/B0, then the bank number to 0x0000) and the reset to
// read mode (F0) on its timeout paths.
void Port_FlashCommand(u32 addr, u8 data)
{
    switch (sFlashCmdState)
    {
    case 0:
        if (addr == 0x5555 && data == 0xAA)
            sFlashCmdState = 1;
        else if (data == 0xF0)
            sFlashCmdState = 0; // reset / read array
        else
            Port_Fatal("flash: unexpected write %02X to %04X", data, (unsigned)addr);
        return;
    case 1:
        if (addr == 0x2AAA && data == 0x55)
            sFlashCmdState = 2;
        else
            Port_Fatal("flash: broken command sequence (%02X to %04X)", data, (unsigned)addr);
        return;
    case 2:
        if (addr == 0x5555 && data == 0xB0)
            sFlashCmdState = 3;
        else if (addr == 0x5555 && data == 0xF0)
            sFlashCmdState = 0;
        else
            Port_Fatal("flash: unsupported command %02X (chip operations go through Port_Flash*)", data);
        return;
    case 3:
        if (addr != 0)
            Port_Fatal("flash: bank select written to %04X", (unsigned)addr);
        gPortFlashBank = data & 1;
        sFlashCmdState = 0;
        return;
    }
}

static void CheckFlashAddr(const u8 *p, u32 size, const char *what)
{
    if (p < FLASH_CHIP || p + size > FLASH_CHIP + FLASH_CHIP_SIZE)
        Port_Fatal("flash: %s outside the chip", what);
}

void Port_FlashEraseChip(void)
{
    memset(FLASH_CHIP, 0xFF, FLASH_CHIP_SIZE);
}

void Port_FlashEraseSector(u8 *sectorAddr)
{
    u8 *start = FLASH_CHIP + ((sectorAddr - FLASH_CHIP) & ~0xFFF);
    CheckFlashAddr(start, 0x1000, "sector erase");
    memset(start, 0xFF, 0x1000);
}

void Port_FlashProgramByte(u8 *addr, u8 data)
{
    CheckFlashAddr(addr, 1, "byte program");
    // Programming can only clear bits. If the result is not the requested
    // value the game's status polling would only end through the flash timer
    // interrupt (timeout), which the simulator does not model.
    if ((*addr & data) != data)
        Port_Fatal("flash: programming a byte that is not erased (%02X over %02X)", data, *addr);
    *addr &= data;
}
