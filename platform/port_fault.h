// fireredsim platform layer: internal interface between the frame driver
// (frame.c) and the hardware-fault emulation (memfault.c). No game headers:
// memfault.c includes system headers that must keep the host ABI.
#ifndef FIREREDSIM_PORT_FAULT_H
#define FIREREDSIM_PORT_FAULT_H

#include <stdint.h>

// The stack game code runs on (frame.c, bottom of .pnstate).
#define PORT_GAME_STACK_SIZE  0x8000
#define PORT_GAME_STACK_GUARD 0x400
extern uint32_t gPortGameStack[PORT_GAME_STACK_SIZE / 4];

// Nonzero while game code runs on the game stack (frame.c, host state).
extern volatile int gPortOnGameStack;
// Incremented whenever game code starts running (a frame or the boot);
// the hang watchdog compares it between two ticks (host state).
extern volatile uint32_t gPortGameRunSeq;

// BIOS open-bus value (bios.c, .pnstate).
extern uint32_t gPortBiosPrefetch;
// Flash bank selected by the game (flash.c, .pnstate).
extern uint8_t gPortFlashBank;

// memfault.c ------------------------------------------------------------------
// Installs the fault handlers, the alternate signal stack and the hang
// watchdog for the calling thread if that has not been done yet. Cheap after
// the first call (one comparison). Called before game code runs.
void Port_FaultPrepareThread(void);
// Reports fault sites seen for the first time (stderr / the host's callback).
// Called after game code returned, on the host stack.
void Port_FaultFlushReports(void);
// Records an open-bus or other inexact read requested explicitly by a PORTABLE
// patch (Port_BiosOpenBus*), for the per-site report.
void Port_FaultNoteExplicit(uint32_t kind, uint32_t addr, uint32_t value, const void *caller);

// frame.c ----------------------------------------------------------------------
// Records the environment's terminal status (first one wins). Safe to call
// from a signal handler that interrupted game code (no C library calls).
void Port_SetTerminalRaw(uint32_t kind, uint32_t eip, uint32_t addr, const char *message);
// Heuristic backtrace: return addresses (words that point just after a call
// instruction in the executable's code) on the game stack from `esp` up.
// Returns how many were found (at most `max`). memfault.c.
int Port_GameBacktrace(uint32_t esp, uint32_t *out, int max);
// Fills the terminal status's backtrace from the game stack at `esp` (frame.c).
void Port_SetTerminalBacktrace(uint32_t esp);
// Ends the current frame with PORT_FRAME_TERMINAL (non-local exit to the frame
// driver). Only valid while game code runs; never returns.
void Port_TerminalExit(void) __attribute__((noreturn));
// Function a signal handler makes the interrupted game code "call" to end the
// frame: it runs on a rescue stack and calls Port_TerminalExit.
void Port_TerminalThunk(void) __attribute__((noreturn));
// Ends the environment with the given PORT_TERMINAL_* kind (inside game code).
void Port_Terminal(int kind, const char *fmt, ...) __attribute__((noreturn, format(printf, 2, 3)));
// Set by Port_SetTerminalRaw; Port_FaultFlushReports reports and clears it.
extern volatile int gPortTerminalPending;

// Site kinds (also the public PORT_FAULT_* values in fireredsim_port.h).
enum {
    PN_SITE_BIOS_READ = 1,      // read (or read-modify-write) of the BIOS region: BIOS open bus
    PN_SITE_BIOS_WRITE = 2,     // write to the BIOS region: ignored
    PN_SITE_UNMAPPED_WRITE = 3, // write to unmapped GBA memory: ignored
    PN_SITE_MIRROR = 4,         // access to a mirror of EWRAM/IWRAM/PLTT/VRAM/OAM/flash
    PN_SITE_ROM_WRITE = 5,      // write to the cartridge ROM: ignored
    PN_SITE_DIV_ZERO = 6,       // integer division by zero: agbcc's libgcc result (0)
    PN_SITE_DIV_OVERFLOW = 7,   // INT_MIN / -1: agbcc's libgcc result (INT_MIN, remainder 0)
    PN_SITE_OPEN_BUS = 8,       // read of unmapped memory (CPU open bus): not reproducible -> terminal
    PN_SITE_TERMINAL = 9,       // other fault that ended the environment
    PN_SITE_EXPLICIT_BIOS = 10, // Port_BiosOpenBus* called by a PORTABLE patch (exact)
    PN_SITE_SPIN_WAIT = 11,     // a busy-wait on interrupt-only state: VBlank delivered inside the body
    PN_SITE_EXPLICIT_OPEN_BUS = 12, // Port_LoadWithOpenBus: open bus with the retail opcode (exact)
};

#endif // FIREREDSIM_PORT_FAULT_H
