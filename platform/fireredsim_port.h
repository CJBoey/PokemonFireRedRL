// fireredsim platform layer: host-facing C API.
//
// This header is the only interface between the host side (sim/, cli/, tests,
// written in C++20) and the game + hardware replacements in game/ and
// platform/. It deliberately includes no game header: the host never touches
// game structs, only the state regions' bytes, symbol addresses and the
// functions below.
//
// Frame model (ARCHITECTURE.md section 4), one call of Port_RunFrame:
//
//   [VBlank: pending VBlank DMAs, VBlankIntr]   if vblank_pending (the IRQ raised
//                                               at the end of the previous frame)
//   REG_KEYINPUT = ~keys
//   AgbMain_RunFrame()                          ReadKeys, callbacks, ... (VCOUNT=160)
//   [VCountIntr]                                line 150, if enabled
//   -> frame boundary: VBlank (line 160) raised, not yet serviced
//
// Everything mutable that the game can observe lives in the state regions
// below (mirror layout, docs/MIRROR_LAYOUT.md): the GBA's EWRAM, IWRAM, I/O
// registers, palette RAM, VRAM, OAM and flash chip at their GBA addresses,
// every game variable at its retail ROM address, plus .pnstate, the
// native-only platform state (DMA/flash bookkeeping, the timer 1 value
// sequence, the stack the game code runs on: Port_BootInit and Port_RunFrame
// switch to it). Host-side bookkeeping (frame counter, vblank_pending, ...) is
// passed in and out explicitly.
#ifndef FIREREDSIM_PORT_H
#define FIREREDSIM_PORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The state regions: NOLOAD sections of the generated linker script
// (game/build/native/mirror.ld, tools/mirror/gen_ld.py) at the GBA's addresses.
// Nothing else is mapped near them, so an access that leaves a region faults
// (except inside the 4 KiB page of the 1 KiB I/O, palette and OAM regions).
#define PORT_EWRAM_ADDR   0x02000000u
#define PORT_EWRAM_SIZE   0x40000u
#define PORT_IWRAM_ADDR   0x03000000u
#define PORT_IWRAM_SIZE   0x8000u
#define PORT_IO_ADDR      0x04000000u
#define PORT_IO_SIZE      0x400u
#define PORT_PLTT_ADDR    0x05000000u
#define PORT_PLTT_SIZE    0x400u
#define PORT_VRAM_ADDR    0x06000000u
#define PORT_VRAM_SIZE    0x18000u
#define PORT_OAM_ADDR     0x07000000u
#define PORT_OAM_SIZE     0x400u
#define PORT_FLASH_ADDR   0x0E000000u
#define PORT_FLASH_SIZE   0x20000u
// .pnstate: native-only state (no GBA address); the game stack is at its
// bottom, so a stack overflow faults instead of corrupting state.
#define PORT_PNSTATE_ADDR 0x10000000u
// IWRAM bytes cleared by RegisterRamReset(RESET_IWRAM); the top 0x200 bytes
// hold the GBA's stacks and the SOUND_INFO_PTR/INTR_CHECK/INTR_VECTOR words.
#define PORT_IWRAM_RESET_SIZE 0x7E00u

// Section bounds defined by the linker script (checked against the constants
// above by the host at startup).
extern unsigned char __gba_ewram_start[], __gba_ewram_end[];
extern unsigned char __gba_iwram_start[], __gba_iwram_end[];
extern unsigned char __gba_io_start[], __gba_io_end[];
extern unsigned char __gba_pltt_start[], __gba_pltt_end[];
extern unsigned char __gba_vram_start[], __gba_vram_end[];
extern unsigned char __gba_oam_start[], __gba_oam_end[];
extern unsigned char __gba_flash_start[], __gba_flash_end[];
extern unsigned char __pnstate_start[], __pnstate_end[];

// Result flags of Port_BootInit / Port_RunFrame.
enum {
    PORT_FRAME_VBLANK_PENDING = 1u << 0, // a VBlank IRQ is raised at the boundary
    PORT_FRAME_SOFT_RESET     = 1u << 1, // the game executed SoftReset(); the host must
                                         // power the environment on again (flash kept)
    PORT_FRAME_TERMINAL       = 1u << 2, // the environment has ended (Port_GetTerminal);
                                         // it stays ended until it is powered on again or
                                         // a snapshot is restored (docs/ROBUSTNESS.md)
};

// ---- terminal status (docs/ROBUSTNESS.md) ------------------------------------------
//
// An environment ends, instead of crashing the process or spinning forever,
// when the game reaches a state the simulator cannot continue exactly. The
// frame in which that happens is abandoned at that point (the state regions
// hold whatever the frame had written so far: discard the environment, i.e.
// reset it or restore a snapshot); every later Port_RunFrame returns
// PORT_FRAME_TERMINAL without running game code. The status is part of the
// environment's state (.pnstate): snapshots, clones and resets carry it.
enum {
    PORT_TERMINAL_NONE = 0,
    PORT_TERMINAL_HUNG = 1,        // the real game hangs forever here (endless Halt loop on
                                   // gNullScriptPtr, BIOS Div by zero, a wait no interrupt can
                                   // end) or the native hang watchdog fired
    PORT_TERMINAL_CRASHED = 2,     // the real game crashes / runs garbage here (call through a
                                   // pointer read from unmapped memory, failed AGB_ASSERT,
                                   // illegal instruction)
    PORT_TERMINAL_UNSUPPORTED = 3, // a feature the simulator does not emulate (link cable,
                                   // wireless adapter, e-Reader, multiboot)
    PORT_TERMINAL_INEXACT = 4,     // the next state depends on hardware behaviour that cannot be
                                   // reproduced natively (read of unmapped memory = CPU open bus)
    PORT_TERMINAL_INTERNAL = 5,    // simulator error (game stack overflow, Port_Fatal, ...)
};
#define PORT_BACKTRACE_MAX 8
struct PortTerminal
{
    uint32_t kind;     // PORT_TERMINAL_*
    uint32_t eip;      // native code address where it happened (0 if not applicable)
    uint32_t addr;     // GBA address involved (0 if not applicable)
    uint32_t reserved;
    uint32_t backtrace[PORT_BACKTRACE_MAX]; // return addresses found on the game stack (0 = none)
    char message[208]; // NUL-terminated description
};
// Copies the environment's terminal status; returns its kind (0 while it runs).
uint32_t Port_GetTerminal(struct PortTerminal *out);
const char *Port_TerminalKindName(uint32_t kind);

// ---- retail hardware-fault emulation (docs/ROBUSTNESS.md) ---------------------------
//
// Retail code reads and writes through NULL and stale pointers, divides by
// zero, etc. On the GBA these accesses have defined results (BIOS open bus,
// ignored writes, mirrors, agbcc's x/0 == 0); natively they fault. The
// platform handles SIGSEGV/SIGBUS/SIGFPE/SIGILL raised by game code (on an
// alternate signal stack): the faulting instruction is re-executed with its
// memory operand redirected to the GBA's value (exact), or the environment
// ends with a terminal status. Faults outside game code go to the handler that
// was installed before (sim's diagnostics). Installed automatically when game
// code first runs on a thread. Every distinct site is reported once per
// process (stderr by default) and counted.
enum {
    PORT_FAULT_BIOS_READ = 1,      // read of the BIOS region 0..0x3FFF: BIOS open bus (exact)
    PORT_FAULT_BIOS_WRITE = 2,     // write to the BIOS region: ignored (exact)
    PORT_FAULT_UNMAPPED_WRITE = 3, // write to unmapped GBA memory: ignored (exact)
    PORT_FAULT_MIRROR = 4,         // EWRAM/IWRAM/palette/VRAM/OAM/flash mirror (exact)
    PORT_FAULT_ROM_WRITE = 5,      // write to the cartridge ROM: ignored (exact)
    PORT_FAULT_DIV_ZERO = 6,       // x / 0, x % 0: 0 as agbcc's libgcc (exact)
    PORT_FAULT_DIV_OVERFLOW = 7,   // INT_MIN / -1: INT_MIN, remainder 0 (exact)
    PORT_FAULT_OPEN_BUS = 8,       // read of unmapped memory: terminal (inexact)
    PORT_FAULT_TERMINAL = 9,       // any other fault in game code: terminal
    PORT_FAULT_EXPLICIT_BIOS = 10, // BIOS open bus requested by a PORTABLE patch (exact)
    PORT_FAULT_SPIN_WAIT = 11,     // body spun on interrupt-only state: VBlank delivered there (exact)
    PORT_FAULT_EXPLICIT_OPEN_BUS = 12, // open bus from the retail opcode, by a PORTABLE patch (exact)
};
struct PortFaultSite
{
    uint32_t eip;          // native instruction address (the site)
    uint32_t kind;         // PORT_FAULT_*
    uint32_t addr;         // GBA address of the first occurrence
    uint32_t value;        // value read at the first occurrence (BIOS open bus), if any
    uint64_t count;        // occurrences in this process
    uint32_t backtrace[PORT_BACKTRACE_MAX]; // callers at the first occurrence (game stack scan)
    char where[96];        // "function+0xoff"
};
// "function+0xoff" for a native code address (the executable's symbol table).
void Port_Symbolize(uint32_t addr, char *buf, uint32_t size);
// Copies up to `max` distinct sites seen by this process; returns how many exist.
int Port_GetFaultSites(struct PortFaultSite *out, int max);
// Number of distinct sites so far (cheap; for noticing new ones).
int Port_FaultSiteCount(void);
// Called once per new site, outside game code. NULL restores the default
// (one line on stderr per new site). Mode 0 silences the default report.
typedef void (*PortFaultReportFn)(const struct PortFaultSite *site);
void Port_SetFaultReport(PortFaultReportFn fn, int mode);
const char *Port_FaultKindName(uint32_t kind);
// Hang watchdog: a frame (or the boot) that runs longer than about
// `cpu_ms` .. 2 x `cpu_ms` milliseconds of the thread's CPU time ends the
// environment with PORT_TERMINAL_HUNG (a native loop that cannot end; real
// frames take microseconds). Default 2000; 0 disables. Takes effect for
// threads that run game code afterwards.
void Port_SetHangWatchdog(uint32_t cpu_ms);

// Puts the hardware into its power-on state. Must be called on state regions
// that have just been zeroed (the GBA's RAM after RegisterRamReset). `flash` points to
// PORT_FLASH_SIZE bytes of initial flash contents, or NULL for an erased chip.
void Port_PowerOn(const uint8_t *flash);

// Runs AgbMain's initialisation (everything before the main loop), including
// the VBlank/VCount interrupts that occur during it on hardware (see
// Port_SetBootVBlanks). Returns PORT_FRAME_* flags.
uint32_t Port_BootInit(void);

// One logical frame. `keys` uses the KEYINPUT layout, active high
// (A=1 B=2 SELECT=4 START=8 RIGHT=0x10 LEFT=0x20 UP=0x40 DOWN=0x80 R=0x100 L=0x200).
uint32_t Port_RunFrame(int vblank_pending, uint16_t keys);

// Values of the free-running timer 1 counter (REG_TM1CNT_L) as seen by
// successive reads: read k (k = 0 first; SeedRngAndSetTrainerId on the title
// screen and when the naming screen closes) returns values[min(k, count - 1)].
// They seed the RNG and give the trainer ID. Stored in .pnstate with
// the read position `next` (0..count); call after Port_PowerOn. count 1..16.
#define PORT_TIMER1_VALUES_MAX 16
void Port_SetTimer1Values(const uint16_t *values, int count, int next);
// Returns count; copies the values (up to 16) and the read position.
int Port_GetTimer1Values(uint16_t *values, int *next);

// Deepest use of the game stack since power-on, in bytes (diagnostics).
uint32_t Port_GameStackHighWater(void);

// Number of VBlank interrupts delivered during AgbMain_Init (default 12,
// measured on the mGBA oracle: initialisation takes 12 frames, mostly the
// wireless-adapter probe in InitRFU and ReadFlashId's delay loops). 13: 12
// during the init plus one pending when the first main-loop body starts (what
// mGBA shows after some soft resets, depending on the CPU-cycle phase of the
// reset; differential testing only, docs/ROBUSTNESS.md). Used by every later
// Port_BootInit (power-on, reset, soft reset) until changed.
void Port_SetBootVBlanks(int n);
int Port_GetBootVBlanks(void);

// Differential testing only: the next Port_RunFrame delivers `extra_vblanks`
// VCount+VBlank interrupt pairs right after the frame's callbacks, emulating a
// frame that lags on hardware (see --lag-profile in cli/). Default 0.
void Port_SetFrameLag(int extra_vblanks);
// Differential testing only, after Port_SetFrameLag: deliver the first `count`
// of those extra VBlanks inside the body where the hardware took them: VBlank i
// where the body's gRngValue equals rngValues[i] and, as far as possible,
// gMain.vblankCallback equals vblankCallbacks[i] (a native address; 0xFFFFFFFF
// or a NULL array: any): the oracle's values at the frame's i-th mid-body
// boundary (platform/frame.c, tools/fuzz/oracle_compare.py).
void Port_SetFrameLagAnchors(const uint32_t *rngValues, const uint32_t *vblankCallbacks, int count);
// Differential testing only, after Port_SetFrameLagAnchors: function-call
// anchors (docs/LAG.md). For i < count with functions[i] != 0, anchor i is
// delivered at the entries[i]-th entry of the game function at native address
// functions[i] in this body instead of by its RNG value; totals[i] (optional,
// 0 = unknown) is how often the oracle's body entered it, checked at the end
// of the body. Entries are only observed in the differential-test build
// (game code compiled with -finstrument-functions, tools/fuzz/build_difftest.sh);
// in the normal build (no game function entry seen since process start) call
// anchors are ignored and the RNG anchors apply (counted as callIgnored).
void Port_SetFrameLagCallAnchors(const uint32_t *functions, const uint32_t *entries, const uint32_t *totals,
                                 int count);
// Differential testing only, after Port_SetFrameLagCallAnchors: where at its
// function entry call anchor i is delivered (docs/COVERAGE.md, "Random() and the
// VBlank"). PORT_CALL_ANCHOR_ENTRY (default): when the entry begins.
// PORT_CALL_ANCHOR_RNG_LOST: at the entry, and gRngValue is put back afterwards
// (the hardware's VBlank interrupted Random() between its load and its store of
// gRngValue, so the store discarded the VBlank's own Random() step).
// PORT_CALL_ANCHOR_RETURN: when that call of the function returns (the hardware's
// VBlank came after Random()'s store).
#define PORT_CALL_ANCHOR_ENTRY 0
#define PORT_CALL_ANCHOR_RNG_LOST 1
#define PORT_CALL_ANCHOR_RETURN 2
void Port_SetFrameLagCallAnchorModes(const uint8_t *modes, int count);
// Differential testing only, after Port_SetFrameLagAnchors: the sound engine's
// lock state the hardware's VBlank i found (docs/COVERAGE.md, "m4a locks"). For
// locks[i] & PORT_LAG_LOCKS_PRESENT, VBlank i runs with gMPlayInfo_BGM/SE1/SE2/SE3
// (bits 0..3) and gSoundInfo (bit 4) locked (ident = ID_NUMBER + 1) or unlocked
// (ID_NUMBER) as given; their idents are restored afterwards. A VBlank that
// interrupts e.g. m4aMPlayPanpotControl between its lock and unlock stores skips
// that player (and the players chained after it) in SoundMain; a function-call
// anchor can only place it at a function entry, outside the locked section.
#define PORT_LAG_LOCKS_PRESENT 0x80
void Port_SetFrameLagAnchorLocks(const uint8_t *locks, int count);
// Cumulative counts since process start (all environments).
struct PortLagAnchorStats {
    uint64_t rngDelivered;      // anchored VBlanks delivered by RNG anchors
    uint64_t callDelivered;     // ... by function-call anchors
    uint64_t rngMissed;         // RNG anchors not reached (delivered after callback2)
    uint64_t callMissed;        // call anchors not reached (delivered after callback2)
    uint64_t callCountMismatch; // call anchors whose function's entries in the body differ from the oracle's
    uint64_t callIgnored;       // call anchors given to a normal build (not instrumented): RNG anchor used
    uint64_t lateDelivered;     // anchors delivered at a later call anchor's point (included above)
};
void Port_GetLagAnchorStats(struct PortLagAnchorStats *out);
// After a body with function-call anchors: for each of its anchors, how often
// that body entered the anchor's function (0 for RNG anchors). Returns the
// number of anchors of that body.
int Port_GetCallAnchorEntries(uint32_t *entries, int max);

// Testing only (tests/fuzz): runs fn on the game stack exactly like a frame
// body (fault emulation, terminal statuses and the watchdog apply). Returns
// PORT_FRAME_TERMINAL if the environment ended.
uint32_t Port_TestRunGameCode(void (*fn)(void));

// ---- sound mixer mode ------------------------------------------------------------
//
// The m4a mixer (platform/sound.c, SoundMainRAM) synthesizes PCM samples into
// gSoundInfo.pcmBuffer, which only the sound FIFO DMA reads (audio output; not
// executed natively), and advances the channel state the sequencer and the game
// read (sample position, end of sample, envelope, status flags, the BDPCM decoding
// buffer). PORT_SOUND_MIX_LOGIC (the default) updates all of that exactly as the
// full mixer does but synthesizes no samples: the PCM buffer is left untouched
// (zero from power-on), which is the only difference from PORT_SOUND_MIX_FULL
// (bit-exact PCM output as on hardware, for audio or A/B checks). Host setting of
// the process (not part of the state); hosts that use pn::Simulator get it from
// the environment variable FIREREDSIM_SOUND_MIX=logic|full (sim/sim.cpp).
enum {
    PORT_SOUND_MIX_LOGIC = 0,
    PORT_SOUND_MIX_FULL = 1,
};
void Port_SetSoundMix(int mode);
int Port_GetSoundMix(void);
// Cumulative counts since process start (all environments).
struct PortSoundMixStats {
    uint64_t mixesLogic;      // SoundMain mixes in logic mode
    uint64_t mixesFull;       // ... by the full mixer (PORT_SOUND_MIX_FULL, or a frame whose full mix would
                              // write outside the PCM buffer)
    uint64_t channelsLogic;   // channel mixes done in closed form (logic mode)
    uint64_t channelsFull;    // channel mixes in logic mode that took the full code path (preconditions of the
                              // closed form not met; PCM output to a scratch buffer)
    uint64_t channelsStopped; // closed-form channel mixes in which the sample ended
    // closed-form channel mixes by path: 0 resampling, 1 resampling with a loop wrap,
    // 2 no resampling (FIX), 3 FIX with a loop wrap, 4 reversed, 5 compressed,
    // 6 compressed with a loop wrap, 7 compressed reversed
    uint64_t paths[8];
};
void Port_GetSoundMixStats(struct PortSoundMixStats *out);
// Address and size of gSoundInfo.pcmBuffer (both halves): the audio-output-only
// state bytes, the only ones in which the two modes differ.
void Port_SoundPcmBuffer(uint32_t *addr, uint32_t *size);
// Testing only (tests/fuzz/test_sound_mix.cpp): runs the mixer part of SoundMain
// (SoundMainRAM, after the sequencer) once on gSoundInfo in the given mode.
void Port_TestSoundMix(int mode);

// Copies the flash chip contents out (PORT_FLASH_SIZE bytes).
void Port_ReadFlashImage(uint8_t *out);

// Message of the last fatal condition (Port_Fatal/Port_Unsupported). The
// default handler prints it and aborts the process; a host may install its own
// handler, which must not return.
typedef void (*PortFatalHandler)(const char *message);
void Port_SetFatalHandler(PortFatalHandler handler);

// Read-only snapshot of a few game values for probes, tests and the adaptive
// driver, filled by the game's own data structures (platform/probe.c). Pointer
// values are native addresses (resolve names with the executable's symbols).
struct PortProbe
{
    uint32_t callback1;          // gMain.callback1
    uint32_t callback2;          // gMain.callback2
    uint32_t vblankCallback;     // gMain.vblankCallback
    uint32_t vblankCounter2;     // gMain.vblankCounter2
    uint32_t rngValue;           // gRngValue
    uint32_t heapUsed;           // bytes in allocated gHeap blocks (headers included)
    uint32_t heapBlocks;         // number of allocated gHeap blocks
    uint32_t saveBlock1;         // gSaveBlock1Ptr
    int16_t x, y;                // gSaveBlock1Ptr->pos
    int8_t mapGroup, mapNum;     // gSaveBlock1Ptr->location
    uint8_t facing;              // player object event facingDirection (0 if none)
    uint8_t mainState;           // gMain.state
    uint8_t textPrinterActive;   // any text printer active
    uint8_t scriptEnabled;       // ScriptContext_IsEnabled()
    uint8_t fieldControlsLocked; // ArePlayerFieldControlsLocked()
    uint8_t inBattle;            // gMain.inBattle
    uint16_t heldKeys;           // gMain.heldKeys
    uint16_t newKeys;            // gMain.newKeys
    uint32_t taskFuncs[16];      // gTasks[i].func of active tasks (0 otherwise)
};
void Port_Probe(struct PortProbe *out);

// Overworld interactions (observer v6, reward v0.14; platform/interact.c). A PORTABLE hook in
// game/src/field_control_avatar.c reports each interaction whose script the game chooses (A on an
// object event or a background event, a scripted sign read by walking into it; not during Quest
// Log playback). The platform keeps the FIRST one since the last Port_TakeInteraction plus a
// count. The latch is host memory (HOST_STATE, outside every state region): it is not saved,
// restored, snapshotted or hashed, it is not per environment, and nothing in the game reads it,
// so a host that hosts several environments takes it around each operation that runs frames.
struct PortInteraction
{
    uint8_t kind;       // 0 none, else PORT_INTERACT_* (game/include/gba/portable_abi.h) =
                        // enum PnInteractKind (obs/observation.h): 1 object, 2 sign, 3 hidden item,
                        // 4 sign walked into
    uint8_t count;      // interactions since the last take (saturates at 255)
    uint8_t mapGroup;   // OBJECT: gObjectEvents[].mapGroup / mapNum; else the current map
    uint8_t mapNum;
    uint8_t localId;    // OBJECT: gObjectEvents[].localId; else 0
    uint8_t reserved[3];
    int16_t x, y;       // OBJECT: the object's map coordinates; else the background event's
};
// Copies the latch into *out (if out != NULL) and clears it.
void Port_TakeInteraction(struct PortInteraction *out);

// Consumable item uses (observer skill events use_consumable, 2026-10-28, and use_battle_item,
// 2026-10-29; platform/interact.c). The paths of the game that remove an item BECAUSE the player used
// it successfully (item_use.c RemoveUsedItem: Repel, Escape Rope, in battle the Poke Doll / Fluffy
// Tail; party_menu.c: medicine and status heals, PP restorers, Rare Candy, Sacred Ash, evolution
// stones, berries from the berry pouch through the medicine path, the medicine and PP paths also in
// battle; item_use.c in battle: X items, Guard Spec, Dire Hit) report the item id
// (PORT_NOTE_ITEM_USE, game/include/gba/portable_abi.h), counting the ones made in battle apart. A
// refused use (nothing removed), reusable items (flutes, key items, HMs), TMs, toss / sell / give /
// deposit and Poke Balls report nothing. The same latch rules as PortInteraction: host memory, not
// per environment, taken around each operation that runs frames.
struct PortItemUse
{
    uint16_t item;      // the FIRST item consumed since the last take (0 none)
    uint8_t count;      // consumptions since the last take (saturates at 255)
    uint8_t battle;     // of those, the ones made while gMain.inBattle
};
// Copies the latch into *out (if out != NULL) and clears it.
void Port_TakeItemUse(struct PortItemUse *out);

// Video state for an optional renderer (read only; the GBA addresses).
const uint8_t *Port_IoRegs(void);   // 0x400 bytes
const uint8_t *Port_Palette(void);  // 0x400 bytes
const uint8_t *Port_Vram(void);     // 0x18000 bytes
const uint8_t *Port_Oam(void);      // 0x400 bytes

#ifdef __cplusplus
}
#endif

#endif // FIREREDSIM_PORT_H
