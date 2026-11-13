// fireredsim platform layer: the GBA memory-mapped hardware as plain memory.
//
// Mirror layout (docs/MIRROR_LAYOUT.md): the I/O registers (0x04000000),
// palette RAM (0x05000000), VRAM (0x06000000), OAM (0x07000000) and the flash
// chip (0x0E000000) are plain memory at their GBA addresses (NOLOAD sections of
// the generated linker script, tools/mirror/gen_ld.py), as are EWRAM and IWRAM
// with every game variable at its ROM address. REG_BASE, PLTT, VRAM, OAM,
// SOUND_INFO_PTR, INTR_CHECK and INTR_VECTOR are pret's literal definitions.
#include "port_internal.h"
#include "malloc.h"

// gHeap is the first 0x1C000 bytes of EWRAM in the ROM build
// (ld_script*.ld: gHeap = .; . = 0x1C000;); placed at 0x02000000.
EWRAM_DATA ALIGNED(4) u8 gHeap[HEAP_SIZE];

// ---- ROM-resident symbols the game references ------------------------------

// rom_header.s
const char RomHeaderGameCode[GAME_CODE_LENGTH] = "BPRE";
const char RomHeaderSoftwareVersion = 0;

// Machine-code images that the ROM copies into RAM before running them:
// InitIntrHandlers copies 0x800 bytes from intr_main into IntrMain_Buffer and
// m4aSoundInit copies 0x800 bytes from SoundMainRAM into SoundMainRAM_Buffer.
// Native code is never executed from RAM (platform/frame.c dispatches
// interrupts, platform/sound.c mixes), so the copies only need a deterministic
// source; the buffers' contents differ from the ROM's ARM code (mask them in
// differential RAM comparisons).
const u32 intr_main[0x200] ALIGNED(4) = {0};
char SoundMainRAM[0x800] ALIGNED(4); // declared non-const in gba/m4a_internal.h

// ---- power-on ---------------------------------------------------------------

void Port_SetScanline(u16 line)
{
    u16 dispstat = PORT_IO16(REG_OFFSET_DISPSTAT) & ~(DISPSTAT_VBLANK | DISPSTAT_HBLANK | DISPSTAT_VCOUNT);
    if (line >= 160 && line < 227)
        dispstat |= DISPSTAT_VBLANK;
    if ((dispstat >> 8) == line)
        dispstat |= DISPSTAT_VCOUNT;
    PORT_IO16(REG_OFFSET_DISPSTAT) = dispstat;
    PORT_IO16(REG_OFFSET_VCOUNT) = line;
}

void Port_PowerOn(const uint8_t *flash)
{
    // The region is all zero here (pristine template). Set the I/O registers
    // that are non-zero at power-on, as mGBA does when it skips the BIOS
    // (GBAIOInit + GBASkipBIOS): forced blank, serial RCNT, keys released,
    // sound bias, identity affine matrices, POSTFLG, VCOUNT 0x7E.
    PORT_IO16(REG_OFFSET_DISPCNT) = DISPCNT_FORCED_BLANK;
    PORT_IO16(REG_OFFSET_RCNT) = 0x8000;
    PORT_IO16(REG_OFFSET_KEYINPUT) = KEYS_MASK;
    PORT_IO16(REG_OFFSET_SOUNDBIAS) = 0x200;
    PORT_IO16(REG_OFFSET_BG2PA) = 0x100;
    PORT_IO16(REG_OFFSET_BG2PD) = 0x100;
    PORT_IO16(REG_OFFSET_BG3PA) = 0x100;
    PORT_IO16(REG_OFFSET_BG3PD) = 0x100;
    *(vu8 *)(REG_BASE + 0x300) = 1; // POSTFLG
    Port_SetScanline(0x7E);

    Port_DmaPowerOn();
    Port_FlashPowerOn(flash);
    Port_SoundPowerOn();
    Port_GameStackPowerOn();
}

// AgbMain (MODERN): after RegisterRamReset(RESET_ALL & ~RESET_IWRAM) an ARM
// loop clears IWRAM from 0x03000000 to 0x03007E00 (what the retail build's
// RegisterRamReset(RESET_ALL) does to IWRAM).
void Port_ClearIwramVariables(void)
{
    memset((void *)IWRAM_START, 0, PORT_IWRAM_RESET_SIZE);
}

const uint8_t *Port_IoRegs(void) { return (const uint8_t *)REG_BASE; }
const uint8_t *Port_Palette(void) { return (const uint8_t *)PLTT; }
const uint8_t *Port_Vram(void) { return (const uint8_t *)VRAM; }
const uint8_t *Port_Oam(void) { return (const uint8_t *)OAM; }
