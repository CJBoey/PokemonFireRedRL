// fireredsim platform layer: internal declarations shared by platform/*.c.
//
// Platform C files are compiled with exactly the game's flags and defines
// (game/Makefile.native), include the game headers and therefore see the same
// struct layouts. Every writable variable they define is moved into .pnstate
// (native-only state, part of every snapshot) by game/native/mirror_sections.py,
// except variables marked HOST_STATE, which hold host-only bookkeeping (never
// game-visible, never snapshotted), and EWRAM_DATA/IWRAM_DATA variables, which
// port a ROM RAM variable and are placed at its ROM address (gHeap, the m4a
// decoding buffer).
#ifndef FIREREDSIM_PORT_INTERNAL_H
#define FIREREDSIM_PORT_INTERNAL_H

#include "global.h"
#include "main.h"
#include "gba/m4a_internal.h"
#include "gba/flash_internal.h"
#include "fireredsim_port.h"

#define HOST_STATE __attribute__((section(".host.bss")))

// ---- hw.c -----------------------------------------------------------------
// The I/O register block as halfwords.
#define PORT_IO16(off) (*(vu16 *)(REG_BASE + (off)))
#define PORT_IO32(off) (*(vu32 *)(REG_BASE + (off)))

// Sets the read-only status registers (VCOUNT, DISPSTAT status bits) for the
// given scanline, as the PPU would.
void Port_SetScanline(u16 line);

extern u32 gPortBiosPrefetch;

// ---- dma.c ----------------------------------------------------------------
void Port_DmaRunVBlank(void);
void Port_DmaPowerOn(void);

// ---- frame.c --------------------------------------------------------------
// Delivers one interrupt (INTR_FLAG_*) through gIntrTable exactly like the
// ROM's intr_main dispatcher, if IME/IE allow it. Returns TRUE if dispatched.
bool32 Port_RaiseIrq(u16 flag);
// Non-local exit used by SoftReset(): never returns.
void Port_RequestSoftReset(void) __attribute__((noreturn));
// Called from the PORTABLE versions of the hardware busy-wait loops that run
// with interrupts enabled during AgbMain_Init (wireless adapter probe).
void Port_BootIdle(void);
// Fills the game stack with its guard pattern (Port_PowerOn).
void Port_GameStackPowerOn(void);
// TRUE if p points into the game stack (IWRAM on the GBA).
int Port_IsOnGameStack(const void *p);

// ---- libc_port.c ------------------------------------------------------------
// Platform code calls memcpy/memset/memmove; game/native/mirror_sections.py
// redirects those references to these (no host libc inside a frame).
void *Port_memcpy(void *dst0, const void *src0, size_t len0);
void *Port_memmove(void *dst0, const void *src0, size_t len0);
void *Port_memset(void *m, int c, size_t n);

// ---- flash.c --------------------------------------------------------------
extern u8 gPortFlashBank;
void Port_FlashPowerOn(const u8 *image);

// ---- sound.c --------------------------------------------------------------
void Port_SoundPowerOn(void);

#endif // FIREREDSIM_PORT_INTERNAL_H
