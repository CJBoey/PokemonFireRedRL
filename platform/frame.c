// fireredsim platform layer: frame driver and interrupt delivery.
//
// Implements the frame model of ARCHITECTURE.md section 4. The GBA main loop
// is AgbMain's for(;;); the game's own code runs unmodified except that
// AgbMain is split into AgbMain_Init and AgbMain_RunFrame (game/src/main.c).
// Interrupts are delivered at the points where they occur on hardware when the
// main loop does not lag:
//
//   line 160  VBlank: VBlank-timed DMA, VBlankIntr        (start of next call)
//             main loop body (VCOUNT reads 160)
//   line 150  VCount: VCountIntr (m4aSoundVSync)
//   line 160  frame boundary, VBlank raised but not serviced (= mGBA runFrame)
//
// Dispatch mirrors the ROM's intr_main (game/src/crt0.s): the IRQ is taken only
// if IME and the IE bit allow it, IF is acknowledged, IME/IE are adjusted for
// the handler and restored afterwards, and the handler is looked up in
// gIntrTable. HBlank, timer, serial, keypad and Game Pak interrupts are never
// raised (HBlank effects are presentation only, see docs/HBLANK_INVENTORY.md;
// there is no link partner and timers do not run inside a frame).
#include "port_internal.h"
#include "port_fault.h"
#include "librfu.h"

// Host-only state: never part of the state regions or of a snapshot.
HOST_STATE static void *sFrameJmp[5];   // __builtin_setjmp buffer (game stack)
HOST_STATE static int sInGameCode;      // a setjmp target is live
HOST_STATE static uint32_t sJumpResult; // why sFrameJmp was taken (PORT_FRAME_SOFT_RESET/TERMINAL)
HOST_STATE static int sInBoot;
HOST_STATE static int sBootIdleDone;
HOST_STATE static int sBootVBlanksDone;
HOST_STATE static int sBootMPlayOpens;
HOST_STATE static int sBootVBlanks;
HOST_STATE static int sBootVBlanksSet;
HOST_STATE static PortFatalHandler sFatalHandler;
HOST_STATE static int sFrameLag;
HOST_STATE static int sIdleVBlanksThisFrame;
HOST_STATE static uint32_t sSpinPolls; // "still waiting" answers since the last VBlank (Port_SpinPoll)
HOST_STATE static int sInIrq;          // an interrupt handler runs (Port_RaiseIrq)
// Anchored lag (differential testing only, Port_SetFrameLagAnchors).
#define PORT_LAG_ANCHORS_MAX 1024
HOST_STATE uint8_t gPortLagAnchorsActive;
HOST_STATE static uint32_t sLagAnchors[PORT_LAG_ANCHORS_MAX];
HOST_STATE static int sLagAnchorCount, sLagAnchorNext;

// Number of VBlank interrupts that occur during AgbMain's initialisation, i.e.
// before the first ReadKeys. Measured on the mGBA oracle
// (reference/traces/mgba_new_game_to_pallet.events.txt): the first ReadKeys is
// in emulator frame 12 and gRngValue at its end is the 12th value of the LCG
// from 0, i.e. VBlankIntr ran 12 times (once per frame 0..11) during init.
#define PORT_DEFAULT_BOOT_VBLANKS 12

void Port_SetBootVBlanks(int n)
{
    sBootVBlanks = n;
    sBootVBlanksSet = 1;
}

int Port_GetBootVBlanks(void)
{
    return sBootVBlanksSet ? sBootVBlanks : PORT_DEFAULT_BOOT_VBLANKS;
}

// ---- fatal conditions and terminal status -----------------------------------------
//
// Inside game code (a frame or the boot), a condition the simulator cannot
// continue from ends the environment (docs/ROBUSTNESS.md): the status is
// recorded in the state and the frame is abandoned through sFrameJmp, so the
// process and every other environment keep running. Outside game code (host
// misuse of this API) the old behaviour applies: the host's fatal handler or
// abort().

// The environment's terminal status (.pnstate: part of every snapshot; zero
// = running after power-on).
static struct PortTerminal sTerminal;

void Port_SetFatalHandler(PortFatalHandler handler)
{
    sFatalHandler = handler;
}

static void CopyMessage(char *dst, uint32_t size, const char *src)
{
    uint32_t i = 0;
    if (src)
        for (; i + 1 < size && src[i]; i++)
            dst[i] = src[i];
    dst[i] = 0;
}

void Port_SetTerminalRaw(uint32_t kind, uint32_t eip, uint32_t addr, const char *message)
{
    if (sTerminal.kind != PORT_TERMINAL_NONE)
        return; // the first cause wins
    sTerminal.kind = kind;
    sTerminal.eip = eip;
    sTerminal.addr = addr;
    sTerminal.reserved = 0;
    CopyMessage(sTerminal.message, sizeof(sTerminal.message), message);
    gPortTerminalPending = 1;
}

// Backtrace of the game code that ended the environment (game stack scan from
// `esp`; frame.c's callers pass their own stack position).
void Port_SetTerminalBacktrace(uint32_t esp)
{
    u32 i;
    for (i = 0; i < PORT_BACKTRACE_MAX; i++)
        sTerminal.backtrace[i] = 0;
    if (gPortOnGameStack)
        Port_GameBacktrace(esp, sTerminal.backtrace, PORT_BACKTRACE_MAX);
}

uint32_t Port_GetTerminal(struct PortTerminal *out)
{
    if (out)
        *out = sTerminal;
    return sTerminal.kind;
}

const char *Port_TerminalKindName(uint32_t kind)
{
    switch (kind)
    {
    case PORT_TERMINAL_NONE: return "NONE";
    case PORT_TERMINAL_HUNG: return "HUNG";
    case PORT_TERMINAL_CRASHED: return "CRASHED";
    case PORT_TERMINAL_UNSUPPORTED: return "UNSUPPORTED";
    case PORT_TERMINAL_INEXACT: return "INEXACT";
    case PORT_TERMINAL_INTERNAL: return "INTERNAL";
    }
    return "?";
}

void Port_TerminalExit(void)
{
    if (!sInGameCode)
    {
        fflush(stdout);
        fprintf(stderr, "fireredsim: FATAL: %s: %s\n", Port_TerminalKindName(sTerminal.kind), sTerminal.message);
        fflush(stderr);
        abort();
    }
    sJumpResult = PORT_FRAME_TERMINAL;
    __builtin_longjmp(sFrameJmp, 1);
}

// Runs on the rescue stack that a signal handler set up (memfault.c).
void Port_TerminalThunk(void)
{
    Port_TerminalExit();
}

HOST_STATE volatile int gPortTerminalPending; // a terminal status was set; reported after the frame

static void __attribute__((noreturn)) TerminalMessage(uint32_t kind, void *caller, const char *fmt, va_list ap)
{
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    if (!sInGameCode)
    {
        // Not inside a frame: host misuse of the API. Fail as before.
        if (sFatalHandler)
            sFatalHandler(buf);
        fflush(stdout);
        fprintf(stderr, "fireredsim: FATAL: %s\n", buf);
        fflush(stderr);
        abort();
    }
    if (sTerminal.kind == PORT_TERMINAL_NONE)
    {
        Port_SetTerminalRaw(kind, (uint32_t)caller, 0, buf);
        Port_SetTerminalBacktrace((uint32_t)__builtin_frame_address(0));
    }
    Port_TerminalExit();
}

void Port_Fatal(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    TerminalMessage(PORT_TERMINAL_INTERNAL, __builtin_return_address(0), fmt, ap);
}

static void __attribute__((noreturn)) TerminalString(uint32_t kind, void *caller, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    TerminalMessage(kind, caller, fmt, ap);
}

void Port_Unsupported(const char *what)
{
    TerminalString(PORT_TERMINAL_UNSUPPORTED, __builtin_return_address(0), "UNSUPPORTED: %s", what);
}

void Port_Terminal(int kind, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    TerminalMessage((uint32_t)kind, __builtin_return_address(0), fmt, ap);
}

// Game-side spellings (gba/portable_abi.h declares them; no shared constants).
void Port_Hang(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    TerminalMessage(PORT_TERMINAL_HUNG, __builtin_return_address(0), fmt, ap);
}

void Port_Crash(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    TerminalMessage(PORT_TERMINAL_CRASHED, __builtin_return_address(0), fmt, ap);
}

void Port_Inexact(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    TerminalMessage(PORT_TERMINAL_INEXACT, __builtin_return_address(0), fmt, ap);
}

// ---- interrupts -----------------------------------------------------------------

static int IntrTableIndex(u16 flag)
{
    switch (flag)
    {
    case INTR_FLAG_VCOUNT:  return 0;
    case INTR_FLAG_SERIAL:  return 1;
    case INTR_FLAG_TIMER3:  return 2;
    case INTR_FLAG_HBLANK:  return 3;
    case INTR_FLAG_VBLANK:  return 4;
    case INTR_FLAG_TIMER0:  return 5;
    case INTR_FLAG_TIMER1:  return 6;
    case INTR_FLAG_TIMER2:  return 7;
    case INTR_FLAG_DMA0:    return 8;
    case INTR_FLAG_DMA1:    return 9;
    case INTR_FLAG_DMA2:    return 10;
    case INTR_FLAG_DMA3:    return 11;
    case INTR_FLAG_KEYPAD:  return 12;
    case INTR_FLAG_GAMEPAK: return 13;
    }
    Port_Fatal("bad interrupt flag %04X", flag);
}

bool32 Port_RaiseIrq(u16 flag)
{
    u16 ie, ime, allowed, timerSelect;
    IntrFunc handler;

    // The request is latched in IF; it is serviced only if IME and IE allow it.
    PORT_IO16(REG_OFFSET_IF) |= flag;
    ime = PORT_IO16(REG_OFFSET_IME);
    ie = PORT_IO16(REG_OFFSET_IE);
    if (!(ime & 1) || !(ie & flag))
        return FALSE;

    // intr_main: IME=0; acknowledge IF; for everything but VCount re-enable
    // IME with IE restricted to the interrupts that may nest; call
    // gIntrTable[i]; restore IE and IME.
    PORT_IO16(REG_OFFSET_IME) = 0;
    PORT_IO16(REG_OFFSET_IF) &= ~flag;
    if (flag != INTR_FLAG_VCOUNT)
        PORT_IO16(REG_OFFSET_IME) = 1;
    timerSelect = gSTWIStatus ? gSTWIStatus->timerSelect : 0;
    allowed = (INTR_FLAG_TIMER0 << timerSelect) | INTR_FLAG_GAMEPAK
            | INTR_FLAG_SERIAL | INTR_FLAG_TIMER3 | INTR_FLAG_VCOUNT | INTR_FLAG_HBLANK;
    PORT_IO16(REG_OFFSET_IE) = allowed & ie & ~flag;

    handler = gIntrTable[IntrTableIndex(flag)];
    if (handler == NULL)
        Port_Terminal(PORT_TERMINAL_CRASHED, "interrupt %04X dispatched to a NULL handler (the GBA "
                      "would jump to the BIOS reset vector)", flag);
    // The BIOS IRQ vector code runs before and after the handler (BIOS open bus).
    gPortBiosPrefetch = 0xE25EF004;
    sInIrq++;
    handler();
    sInIrq--;
    gPortBiosPrefetch = 0xE55EC002;

    PORT_IO16(REG_OFFSET_IE) = ie;
    PORT_IO16(REG_OFFSET_IME) = ime;
    return TRUE;
}

// REG_VCOUNT while the main-loop body runs (STATUS.md deviation 2). On hardware
// the body starts right after VBlankIntr, i.e. inside VBlank, so SetGpuReg
// writes the I/O register at once (161 <= VCOUNT <= 225) unless the body runs
// past line 225. Default 161: measured on 21,332 frames against mGBA
// (docs/RENDER_PARITY.md), native frames are pixel-identical in 99.7% of frames
// with 161 vs 96.0% with 160, with identical game logic (cb2, RNG, map, position)
// under both. 160 makes every body write buffered until the next VBlankIntr.
// Also used when a body resumes after a VBlank delivered inside it (nested
// VBlank wait, lag profile). Host state, not part of an environment's state.
HOST_STATE uint8_t gPortBodyScanline = 161;

// Line 160 at the start of the next frame: the VBlank IRQ raised at the
// previous boundary is serviced (VBlank DMAs run first: DMA has priority).
static void ServiceVBlank(void)
{
    Port_SetScanline(160);
    Port_DmaRunVBlank();
    // The PPU requests the IRQ (sets IF) only if DISPSTAT enables it.
    if (PORT_IO16(REG_OFFSET_DISPSTAT) & DISPSTAT_VBLANK_INTR)
        Port_RaiseIrq(INTR_FLAG_VBLANK);
}

// Line 150, after the main loop body.
static void ServiceVCount(void)
{
    u16 dispstat;
    Port_SetScanline(150);
    dispstat = PORT_IO16(REG_OFFSET_DISPSTAT);
    if ((dispstat & DISPSTAT_VCOUNT_INTR) && (dispstat >> 8) != 150)
        Port_Terminal(PORT_TERMINAL_UNSUPPORTED, "VCount interrupt at line %u (the frame model assumes "
                      "line 150)", dispstat >> 8);
    if (dispstat & DISPSTAT_VCOUNT_INTR)
        Port_RaiseIrq(INTR_FLAG_VCOUNT);
}

// Line 160 at the end of the frame: raise (not service) the VBlank IRQ.
static uint32_t RaiseVBlankAtBoundary(void)
{
    Port_SetScanline(160);
    if (PORT_IO16(REG_OFFSET_DISPSTAT) & DISPSTAT_VBLANK_INTR)
    {
        PORT_IO16(REG_OFFSET_IF) |= INTR_FLAG_VBLANK;
        return PORT_FRAME_VBLANK_PENDING;
    }
    return 0;
}

// ---- soft reset -------------------------------------------------------------------

void Port_RequestSoftReset(void)
{
    if (!sInGameCode)
        Port_Fatal("SoftReset outside of a frame");
    sJumpResult = PORT_FRAME_SOFT_RESET;
    __builtin_longjmp(sFrameJmp, 1);
}

// ---- the game stack -----------------------------------------------------------------
//
// On the GBA the stack is part of IWRAM. The game stores pointers to stack
// variables in RAM (e.g. CreateSprite(&localTemplate) leaves
// gSprites[i].template pointing into the stack) and reads uninitialised locals
// and stale stack contents. To keep that deterministic and inside the
// snapshot, game code runs on this stack at the bottom of .pnstate (native-only
// state at 0x10000000, part of every snapshot), at a fixed address
// (ARCHITECTURE.md section 5: no pointer to host memory in the state). It is
// deliberately not in the GBA's IWRAM stack area (0x03007480..0x03007FFF):
// retail code reads there (DestroyTask(TASK_NONE) reads 0x0300786C and relies
// on it being zero, as on mGBA), and IWRAM must stay comparable with the
// oracle's dumps.
//
// Port_CallOnGameStack switches to it with every callee-saved register zeroed,
// so no host value (host stack or heap address) can be spilled into it, and
// switches back afterwards. Code running on it must not call into the host C
// library (game/native/mirror_sections.py enforces that for memcpy & co.),
// except on the fatal-error path. The lowest PORT_GAME_STACK_GUARD bytes hold a
// pattern written at power-on; Port_RunFrame checks it after every frame.
#define PORT_GAME_STACK_FILL  0x57A2C3E1u
__attribute__((section(".pnstate.stack"), aligned(64))) u32 gPortGameStack[PORT_GAME_STACK_SIZE / 4];

HOST_STATE u32 gPortHostEsp; // host stack pointer while the game stack is in use
HOST_STATE volatile int gPortOnGameStack;
HOST_STATE volatile uint32_t gPortGameRunSeq;

void Port_CallOnGameStack(void (*fn)(void), void *stackTop);
__asm__(
    "    .text\n"
    "    .p2align 4\n"
    "    .globl Port_CallOnGameStack\n"
    "    .type Port_CallOnGameStack, @function\n"
    "Port_CallOnGameStack:\n"
    "    pushl %ebp\n"
    "    pushl %ebx\n"
    "    pushl %esi\n"
    "    pushl %edi\n"
    "    movl 20(%esp), %eax\n"        // fn
    "    movl 24(%esp), %ecx\n"        // stackTop (16-byte aligned)
    "    movl %esp, gPortHostEsp\n"
    "    movl %ecx, %esp\n"
    "    xorl %ebp, %ebp\n"
    "    xorl %ebx, %ebx\n"
    "    xorl %esi, %esi\n"
    "    xorl %edi, %edi\n"
    "    xorl %ecx, %ecx\n"
    "    xorl %edx, %edx\n"
    "    call *%eax\n"
    "    movl gPortHostEsp, %esp\n"
    "    popl %edi\n"
    "    popl %esi\n"
    "    popl %ebx\n"
    "    popl %ebp\n"
    "    ret\n"
    "    .size Port_CallOnGameStack, .-Port_CallOnGameStack\n");

// Leaves the game stack from a body's setjmp receiver (terminal status, soft
// reset) through Port_CallOnGameStack's epilogue, i.e. straight back to its
// caller on the host stack, without the body's own frame and return address.
// Game code can overwrite every frame up to the top of the game stack (retail
// stack smash: a trainer battle against TRAINER_LINK_OPPONENT outside a link
// ends in AddOpponentLinkBattleRecord's StringCopy of an unterminated link
// name; the GBA then returns to address 0, the reset vector): returning
// through FrameBody's smashed return address jumped to 0 outside game code
// and aborted the process instead of ending the environment.
void Port_LeaveGameStack(void) __attribute__((noreturn));
__asm__(
    "    .text\n"
    "    .p2align 4\n"
    "    .globl Port_LeaveGameStack\n"
    "    .type Port_LeaveGameStack, @function\n"
    "Port_LeaveGameStack:\n"
    "    movl gPortHostEsp, %esp\n"
    "    popl %edi\n"
    "    popl %esi\n"
    "    popl %ebx\n"
    "    popl %ebp\n"
    "    ret\n"
    "    .size Port_LeaveGameStack, .-Port_LeaveGameStack\n");

static void RunOnGameStack(void (*fn)(void))
{
    if (gPortOnGameStack)
        Port_Fatal("re-entered the game stack");
    // Fault handlers, alternate signal stack and hang watchdog for this thread
    // (memfault.c; one comparison once they are in place).
    Port_FaultPrepareThread();
    gPortGameRunSeq++;
    gPortOnGameStack = 1;
    Port_CallOnGameStack(fn, (u8 *)gPortGameStack + sizeof(gPortGameStack));
    gPortOnGameStack = 0;
}

void Port_GameStackPowerOn(void)
{
    u32 i;
    for (i = 0; i < PORT_GAME_STACK_SIZE / 4; i++)
        gPortGameStack[i] = PORT_GAME_STACK_FILL;
}

// After game code returned (host stack): a damaged guard area ends the
// environment (the overflow itself faults below the stack, memfault.c).
// PN_STACK_GUARD=0 (CMake -DPN_STACK_GUARD=OFF, part of PN_FAST) skips the check
// (1.6 % of a frame, docs/SPS_PLAN.md section 2.8): an overflow deep enough to
// leave the stack still ends the environment through memfault.c, one that only
// reaches into the guard area goes unnoticed. The guard pattern is still written
// at power-on, so the state is the same either way. Differential-test builds keep
// the default (on).
#ifndef PN_STACK_GUARD
#define PN_STACK_GUARD 1
#endif
static uint32_t CheckGameStack(void)
{
#if !PN_STACK_GUARD
    return 0;
#endif
    // Every guard word, 64 bytes per step (the stack is 64-aligned) into four
    // independent accumulators, one test at the end (was: a word-by-word loop,
    // 3.4-7.5 % of a frame).
    typedef u32 GuardVec __attribute__((vector_size(16)));
    const GuardVec *p = (const GuardVec *)gPortGameStack;
    const GuardVec fill = {PORT_GAME_STACK_FILL, PORT_GAME_STACK_FILL, PORT_GAME_STACK_FILL, PORT_GAME_STACK_FILL};
    GuardVec d0 = {0, 0, 0, 0}, d1 = d0, d2 = d0, d3 = d0;
    u32 i;
    for (i = 0; i < PORT_GAME_STACK_GUARD / 16; i += 4)
    {
        d0 |= p[i] ^ fill;
        d1 |= p[i + 1] ^ fill;
        d2 |= p[i + 2] ^ fill;
        d3 |= p[i + 3] ^ fill;
    }
    d0 |= d1 | d2 | d3;
    if (d0[0] | d0[1] | d0[2] | d0[3])
    {
        Port_SetTerminalRaw(PORT_TERMINAL_INTERNAL, 0, 0, "game stack overflow (guard area overwritten; "
                            "raise PORT_GAME_STACK_SIZE in platform/port_fault.h)");
        return PORT_FRAME_TERMINAL;
    }
    return 0;
}

uint32_t Port_GameStackHighWater(void)
{
    u32 i;
    for (i = 0; i < PORT_GAME_STACK_SIZE / 4; i++)
        if (gPortGameStack[i] != PORT_GAME_STACK_FILL)
            break;
    return PORT_GAME_STACK_SIZE - i * 4;
}

int Port_IsOnGameStack(const void *p)
{
    const u8 *q = p;
    return q >= (const u8 *)gPortGameStack && q < (const u8 *)gPortGameStack + sizeof(gPortGameStack);
}

// ---- boot -------------------------------------------------------------------------

// AgbMain's initialisation takes 12 frames on hardware with interrupts
// enabled, measured on the mGBA oracle (main-thread PC and CPU registers at
// emulator boundaries 0..11, docs/DEVLOG.md "m1b agent"):
//
//   frame 0 (starts at line 126) ends inside m4aSoundInit's m4aSoundMode, with
//     the sound engine lock held (soundInfo->ident = ID_NUMBER + 1): the first
//     VBlankIntr runs, its SoundMain returns at once;
//   m4aSoundMode's SampleFreqSet then waits for line 159 of frame 1 (no VCount
//     interrupt: EnableVCountIntrAtLine150 comes after m4aSoundInit);
//   frame 1 ends inside m4aSoundInit's third MPlayOpen (gMPlayInfo_SE2), lock
//     held again: second VBlankIntr, SoundMain skipped;
//   frames 2..11 (wireless adapter probe in InitRFU: AgbRFU_checkID busy-waits
//     64 x 32 ticks of a 1024-cycle timer; ReadFlashId's DELAY loops) each take
//     a VCount and a VBlank interrupt (10 pairs).
//
// Port_BootSoundLocked delivers the first two (the m4a.c hooks run with the
// lock held, which is all that matters for their effect); Port_BootIdle
// delivers the remaining VCount+VBlank pairs at the first probe wait. Their
// position among the init steps between there and InitMainCallbacks does not
// matter (the handlers and those steps touch disjoint state). The sound
// engine's phase counters (pcmDmaCounter: VCounts, c15: SoundMain calls) are
// what makes this schedule observable.
void Port_BootSoundLocked(int site)
{
    if (!sInBoot)
        return;
    if (site == 1 && ++sBootMPlayOpens != 3)
        return;
    if (sBootVBlanksDone != site || sBootVBlanksDone >= Port_GetBootVBlanks())
        return;
    ServiceVBlank();
    sBootVBlanksDone++;
}

// Boot VBlanks beyond the measured 12 (Port_SetBootVBlanks(13)): on mGBA the
// init after a soft reset sometimes spans one more VBlank, depending on the
// CPU-cycle phase of the reset. The 13th falls after the initialisation (the
// VBlank is pending when the first main-loop body starts), see Port_BootInit.
#define PORT_BOOT_VBLANKS_IN_INIT 12

void Port_BootIdle(void)
{
    int n;
    if (!sInBoot || sBootIdleDone)
        return;
    sBootIdleDone = 1;
    n = Port_GetBootVBlanks();
    if (n > PORT_BOOT_VBLANKS_IN_INIT)
        n = PORT_BOOT_VBLANKS_IN_INIT;
    while (sBootVBlanksDone < n)
    {
        ServiceVCount();
        ServiceVBlank();
        sBootVBlanksDone++;
    }
    Port_SetScanline(160);
}

HOST_STATE static uint32_t sBodyResult;
HOST_STATE static int sArgVBlankPending;
HOST_STATE static uint16_t sArgKeys;

// Runs on the game stack.
static void BootBody(void)
{
    sInBoot = 1;
    sSpinPolls = 0;
    sBootIdleDone = 0;
    sBootVBlanksDone = 0;
    sBootMPlayOpens = 0;
    sInGameCode = 1;
    if (__builtin_setjmp(sFrameJmp))
    {
        sInGameCode = 0;
        sInBoot = 0;
        sBodyResult = sJumpResult;
        Port_LeaveGameStack();
    }
    AgbMain_Init();
    sInGameCode = 0;
    sInBoot = 0;
    if (!sBootIdleDone && Port_GetBootVBlanks() > 0)
        Port_Fatal("boot: the wireless adapter probe did not run");
    // The first main-loop iteration starts right after initialisation, without
    // waiting for a VBlank: no VBlank is pending for frame 0 (after power-on;
    // after a soft reset the init can end with one pending, see Port_BootIdle).
    Port_SetScanline(160);
    sBodyResult = Port_GetBootVBlanks() > PORT_BOOT_VBLANKS_IN_INIT ? RaiseVBlankAtBoundary() : 0;
}

uint32_t Port_BootInit(void)
{
    if (sTerminal.kind != PORT_TERMINAL_NONE)
        return PORT_FRAME_TERMINAL;
    RunOnGameStack(BootBody);
    if (!(sBodyResult & PORT_FRAME_TERMINAL))
        sBodyResult |= CheckGameStack();
    Port_FaultFlushReports();
    return (sBodyResult & PORT_FRAME_TERMINAL) ? PORT_FRAME_TERMINAL : sBodyResult;
}

// ---- lag emulation (differential testing only) ------------------------------------------

void Port_SetFrameLag(int extra_vblanks)
{
    sFrameLag = extra_vblanks;
}

// Anchored lag (differential testing): the next frame's extra VBlanks are
// delivered where the hardware took them inside the body. Anchor i is the
// oracle's gRngValue and gMain.vblankCallback at the frame's i-th mid-body
// boundary, where that VBlank had not run yet (tools/fuzz/oracle_compare.py).
// It is delivered at the latest point that keeps both: before the next body
// Random() call would step the generator away from that value, or right
// before SetVBlankCallback replaces the callback the hardware's VBlank ran; a
// nested wait (Port_IdleUntilVBlank) whose VBlank comes at that value counts
// as it. So the VBlank lies between the same two body Random() calls as on
// hardware, with the same callback. Extra VBlanks whose anchor is not reached
// are delivered by Port_LagHook as before.
#define PORT_ANCHOR_ANY_CALLBACK 0xFFFFFFFFu
HOST_STATE static uint32_t sLagAnchorCallbacks[PORT_LAG_ANCHORS_MAX];

// Function-call anchors (Port_SetFrameLagCallAnchors, docs/LAG.md): anchor i
// is delivered at the sLagAnchorEntries[i]-th entry of the game function
// sLagAnchorFunctions[i] in this body (0: an RNG anchor). Entries are seen by
// __cyg_profile_func_enter, i.e. only in the differential-test build whose
// game code is compiled with -finstrument-functions
// (tools/fuzz/build_difftest.sh); the normal build never calls it.
#define PORT_CALL_ANCHOR_FUNCS 32
HOST_STATE static uint32_t sLagAnchorFunctions[PORT_LAG_ANCHORS_MAX];
HOST_STATE static uint32_t sLagAnchorEntries[PORT_LAG_ANCHORS_MAX];
HOST_STATE static uint32_t sLagAnchorTotals[PORT_LAG_ANCHORS_MAX];
HOST_STATE static uint32_t sLagAnchorNative[PORT_LAG_ANCHORS_MAX]; // entries of the function in the last body
HOST_STATE static int sLastCallAnchorCount;
HOST_STATE static uint32_t sCallFns[PORT_CALL_ANCHOR_FUNCS];
HOST_STATE static uint32_t sCallCounts[PORT_CALL_ANCHOR_FUNCS];
HOST_STATE static int sCallFnCount;
HOST_STATE static uint8_t sLagAnchorModes[PORT_LAG_ANCHORS_MAX]; // PORT_CALL_ANCHOR_* (Port_SetFrameLagCallAnchorModes)
HOST_STATE static uint8_t sLagAnchorLocks[PORT_LAG_ANCHORS_MAX]; // Port_SetFrameLagAnchorLocks
HOST_STATE static uint32_t sExitAnchorFn;    // a PORT_CALL_ANCHOR_RETURN anchor waits for this function's return
HOST_STATE static uint32_t sExitAnchorEntry; // ... of its sExitAnchorEntry-th entry
HOST_STATE static uint8_t sCallAnchorsArmed; // count entries in this body
HOST_STATE static uint8_t sInCallHook;
HOST_STATE static uint8_t sCallHookSeen; // game code is instrumented (difftest build)
HOST_STATE static struct PortLagAnchorStats sAnchorStats;

void Port_SetFrameLagAnchors(const uint32_t *rngValues, const uint32_t *vblankCallbacks, int count)
{
    int i;
    if (count < 0 || count > PORT_LAG_ANCHORS_MAX)
        Port_Fatal("lag anchors: %d anchors (0..%d)", count, PORT_LAG_ANCHORS_MAX);
    for (i = 0; i < count; i++)
    {
        sLagAnchors[i] = rngValues[i];
        sLagAnchorCallbacks[i] = vblankCallbacks ? vblankCallbacks[i] : PORT_ANCHOR_ANY_CALLBACK;
        sLagAnchorFunctions[i] = 0;
        sLagAnchorModes[i] = PORT_CALL_ANCHOR_ENTRY;
        sLagAnchorLocks[i] = 0;
    }
    sExitAnchorFn = 0;
    sLagAnchorCount = count;
    sLagAnchorNext = 0;
    gPortLagAnchorsActive = count > 0;
    sCallFnCount = 0;
    sCallAnchorsArmed = 0;
}

void Port_SetFrameLagCallAnchors(const uint32_t *functions, const uint32_t *entries, const uint32_t *totals,
                                 int count)
{
    int i, j;
    if (count < 0 || count > sLagAnchorCount)
        Port_Fatal("lag call anchors: %d anchors, %d set by Port_SetFrameLagAnchors", count, sLagAnchorCount);
    for (i = 0; i < count; i++)
    {
        if (functions[i] == 0 || entries[i] == 0)
            continue;
        if (!sCallHookSeen)
        {
            // Normal build: function entries are not observable, the RNG
            // anchor applies.
            sAnchorStats.callIgnored++;
            continue;
        }
        for (j = 0; j < sCallFnCount && sCallFns[j] != functions[i]; j++)
            ;
        if (j == sCallFnCount)
        {
            if (sCallFnCount == PORT_CALL_ANCHOR_FUNCS)
                continue; // stays an RNG anchor
            sCallFns[sCallFnCount] = functions[i];
            sCallCounts[sCallFnCount++] = 0;
        }
        sLagAnchorFunctions[i] = functions[i];
        sLagAnchorEntries[i] = entries[i];
        sLagAnchorTotals[i] = totals ? totals[i] : 0;
        sCallAnchorsArmed = 1;
    }
}

void Port_SetFrameLagCallAnchorModes(const uint8_t *modes, int count)
{
    int i;
    if (count < 0 || count > sLagAnchorCount)
        Port_Fatal("lag call anchor modes: %d anchors, %d set by Port_SetFrameLagAnchors", count, sLagAnchorCount);
    for (i = 0; i < count; i++)
    {
        if (modes[i] > PORT_CALL_ANCHOR_RETURN)
            Port_Fatal("lag call anchor mode %u", modes[i]);
        sLagAnchorModes[i] = modes[i];
    }
}

void Port_SetFrameLagAnchorLocks(const uint8_t *locks, int count)
{
    int i;
    if (count < 0 || count > sLagAnchorCount)
        Port_Fatal("lag anchor locks: %d anchors, %d set by Port_SetFrameLagAnchors", count, sLagAnchorCount);
    for (i = 0; i < count; i++)
        sLagAnchorLocks[i] = locks[i];
}

extern struct MusicPlayerInfo gMPlayInfo_BGM, gMPlayInfo_SE1, gMPlayInfo_SE2, gMPlayInfo_SE3;
extern struct SoundInfo gSoundInfo;

// The sound engine locks of the hardware's VBlank (Port_SetFrameLagAnchorLocks): applied
// to the initialised players/sound info for the duration of the VBlank, then restored.
static void SetSoundLocks(uint8_t locks, u32 *saved)
{
    u32 *idents[5] = {&gMPlayInfo_BGM.ident, &gMPlayInfo_SE1.ident, &gMPlayInfo_SE2.ident,
                      &gMPlayInfo_SE3.ident, &gSoundInfo.ident};
    int i;
    for (i = 0; i < 5; i++)
    {
        saved[i] = *idents[i];
        if (*idents[i] == ID_NUMBER || *idents[i] == ID_NUMBER + 1)
            *idents[i] = (locks >> i) & 1 ? ID_NUMBER + 1 : ID_NUMBER;
    }
}

static void RestoreSoundLocks(const u32 *saved)
{
    u32 *idents[5] = {&gMPlayInfo_BGM.ident, &gMPlayInfo_SE1.ident, &gMPlayInfo_SE2.ident,
                      &gMPlayInfo_SE3.ident, &gSoundInfo.ident};
    int i;
    for (i = 0; i < 5; i++)
        *idents[i] = saved[i];
}

void Port_GetLagAnchorStats(struct PortLagAnchorStats *out)
{
    *out = sAnchorStats;
}

int Port_GetCallAnchorEntries(uint32_t *entries, int max)
{
    int i;
    for (i = 0; i < sLastCallAnchorCount && i < max; i++)
        entries[i] = sLagAnchorNative[i];
    return sLastCallAnchorCount;
}

// The head anchor is an RNG anchor whose value the generator has reached.
static int AnchorDue(void)
{
    extern u32 gRngValue;
    return sLagAnchorNext < sLagAnchorCount && sLagAnchorFunctions[sLagAnchorNext] == 0
        && sLagAnchors[sLagAnchorNext] == gRngValue && sFrameLag > 0;
}

// The head anchor is a function-call anchor.
static int CallAnchorPending(void)
{
    return sLagAnchorNext < sLagAnchorCount && sLagAnchorFunctions[sLagAnchorNext] != 0 && sFrameLag > 0;
}

static void DeliverAnchoredVBlankAt(int ownPoint);
static void DeliverAnchoredVBlank(void)
{
    DeliverAnchoredVBlankAt(0);
}

// ownPoint: the head anchor is delivered at its own function-call point (not late,
// at a later anchor's point or after callback2), where its mode applies.
static void DeliverAnchoredVBlankAt(int ownPoint)
{
    extern u32 gRngValue;
    // PORT_CALL_ANCHOR_RNG_LOST: the hardware's VBlank came inside Random() after it
    // had loaded gRngValue and before it stored the new value, so the store discarded
    // VBlankIntr's own Random() step (a lost update). Delivered at Random's entry,
    // the VBlank's other effects stay and the generator is put back to the value
    // Random loaded: Random then computes and stores exactly what the hardware's did.
    int lost = ownPoint && sLagAnchorFunctions[sLagAnchorNext]
        && sLagAnchorModes[sLagAnchorNext] == PORT_CALL_ANCHOR_RNG_LOST;
    u32 rng = gRngValue;
    uint8_t locks = sLagAnchorLocks[sLagAnchorNext];
    u32 savedIdents[5];
    if (sLagAnchorFunctions[sLagAnchorNext])
        sAnchorStats.callDelivered++;
    else
        sAnchorStats.rngDelivered++;
    sLagAnchorNext++;
    sFrameLag--;
    if (locks & PORT_LAG_LOCKS_PRESENT)
        SetSoundLocks(locks, savedIdents);
    ServiceVCount();
    ServiceVBlank();
    if (locks & PORT_LAG_LOCKS_PRESENT)
        RestoreSoundLocks(savedIdents);
    if (lost)
        gRngValue = rng;
    sSpinPolls = 0;
    Port_SetScanline(gPortBodyScanline);
    if (sLagAnchorNext >= sLagAnchorCount)
        gPortLagAnchorsActive = 0;
}

// Random() (game/src/random.c), before it steps the generator.
void Port_RandomAnchor(void)
{
    if (sInIrq)
        return; // VBlankIntr's own Random() calls are not body calls
    while (AnchorDue())
        DeliverAnchoredVBlank();
}

// Entry of every game function in the differential-test build (game code
// compiled with -finstrument-functions; GCC also calls it for inlined
// instances). Never called in the normal build.
__attribute__((no_instrument_function)) void __cyg_profile_func_enter(void *fn, void *site)
{
    int i, j;
    uint32_t n;
    (void)site;
    if (!sCallAnchorsArmed)
    {
        sCallHookSeen = 1;
        return;
    }
    if (sInIrq || sInCallHook)
        return;
    for (i = 0; i < sCallFnCount && sCallFns[i] != (uint32_t)fn; i++)
        ;
    if (i == sCallFnCount)
        return;
    n = ++sCallCounts[i];
    // The first pending anchor at this point. Earlier pending anchors (an
    // RNG anchor whose value did not come up, a call anchor whose function
    // this body enters fewer times than the oracle's) are delivered here too:
    // the hardware took them before this point.
    for (j = sLagAnchorNext; j < sLagAnchorCount; j++)
        if (sLagAnchorFunctions[j] == (uint32_t)fn && sLagAnchorEntries[j] == n)
            break;
    if (j == sLagAnchorCount)
        return;
    sInCallHook = 1;
    while (sLagAnchorNext < j && sFrameLag > 0)
    {
        sAnchorStats.lateDelivered++;
        DeliverAnchoredVBlank();
    }
    while (CallAnchorPending() && sLagAnchorFunctions[sLagAnchorNext] == (uint32_t)fn
           && sLagAnchorEntries[sLagAnchorNext] == n)
    {
        if (sLagAnchorModes[sLagAnchorNext] == PORT_CALL_ANCHOR_RETURN)
        {
            // delivered when this call returns (__cyg_profile_func_exit)
            sExitAnchorFn = (uint32_t)fn;
            sExitAnchorEntry = n;
            break;
        }
        DeliverAnchoredVBlankAt(1);
    }
    sInCallHook = 0;
}

__attribute__((no_instrument_function)) void __cyg_profile_func_exit(void *fn, void *site)
{
    int i;
    (void)site;
    if (!sExitAnchorFn || sExitAnchorFn != (uint32_t)fn || sInIrq || sInCallHook)
        return;
    for (i = 0; i < sCallFnCount && sCallFns[i] != (uint32_t)fn; i++)
        ;
    // the return of the anchored entry (no later entry of the function began: Random
    // and the other anchor functions of this mode do not recurse)
    if (i == sCallFnCount || sCallCounts[i] != sExitAnchorEntry)
        return;
    sExitAnchorFn = 0;
    sInCallHook = 1;
    while (CallAnchorPending() && sLagAnchorFunctions[sLagAnchorNext] == (uint32_t)fn
           && sLagAnchorEntries[sLagAnchorNext] == sExitAnchorEntry
           && sLagAnchorModes[sLagAnchorNext] == PORT_CALL_ANCHOR_RETURN)
        DeliverAnchoredVBlankAt(1);
    sInCallHook = 0;
}

// End of a body with function-call anchors: anchors not reached and anchor
// functions entered a different number of times than on the oracle are
// counted (the body did not follow the oracle's call sequence).
static void CheckCallAnchors(void)
{
    int i, j;
    sLastCallAnchorCount = sLagAnchorCount;
    for (i = 0; i < sLagAnchorCount; i++)
    {
        sLagAnchorNative[i] = 0;
        if (sLagAnchorFunctions[i] == 0)
            continue;
        if (i >= sLagAnchorNext)
            sAnchorStats.callMissed++;
        for (j = 0; j < sCallFnCount && sCallFns[j] != sLagAnchorFunctions[i]; j++)
            ;
        if (j < sCallFnCount)
            sLagAnchorNative[i] = sCallCounts[j];
        if (sLagAnchorTotals[i] && j < sCallFnCount && sCallCounts[j] != sLagAnchorTotals[i])
            sAnchorStats.callCountMismatch++;
    }
    for (i = sLagAnchorNext; i < sLagAnchorCount; i++)
        if (sLagAnchorFunctions[i] == 0)
            sAnchorStats.rngMissed++;
}

// SetVBlankCallback (game/src/main.c), before it installs `next`.
void Port_VBlankCallbackAnchor(const void *next)
{
    uint32_t cb;
    if (sInIrq)
        return;
    while (AnchorDue())
    {
        cb = sLagAnchorCallbacks[sLagAnchorNext];
        if (cb == PORT_ANCHOR_ANY_CALLBACK || cb != (uint32_t)gMain.vblankCallback || cb == (uint32_t)next)
            break; // not the last chance for this anchor's callback
        DeliverAnchoredVBlank();
    }
}

// A busy-wait inside the frame body that only an interrupt can end (see
// Port_IdleUntilVBlank in gba/portable_abi.h): time passes until the next
// VBlank, so the VCount (line 150) and VBlank (line 160) interrupts are
// delivered, then the loop resumes "inside VBlank". It counts towards a lag
// profile's extra VBlanks of the current frame.
void Port_IdleUntilVBlank(void)
{
    static const int kMaxIdleVBlanks = 600;
    if (++sIdleVBlanksThisFrame > kMaxIdleVBlanks)
        Port_Terminal(PORT_TERMINAL_HUNG, "a loop waited for interrupts for %d frames inside one frame "
                      "body (it can never end: the real game hangs)", kMaxIdleVBlanks);
    if (gPortLagAnchorsActive && (AnchorDue() || CallAnchorPending()))
    {
        // This nested-wait VBlank is the anchored one: with an RNG anchor, the
        // generator is at its value; a function-call anchor not reached yet
        // is the next mid-body VBlank of the hardware's body, which must lie
        // in this wait (the loop ends only after an interrupt; polls have no
        // other effect, so where in the wait it lands does not matter).
        DeliverAnchoredVBlank();
        return;
    }
    if (sFrameLag > 0)
        sFrameLag--;
    sSpinPolls = 0;
    ServiceVCount();
    ServiceVBlank();
    Port_SetScanline(gPortBodyScanline);
}

// Busy-wait detection for predicates whose answer only an interrupt handler
// can change (docs/ROBUSTNESS.md, "nested waits"). Such a predicate calls this
// each time it answers "still waiting" (IsDma3ManagerBusyWithBgCopy: the DMA3
// queue is drained only by VBlankIntr's ProcessDma3Requests). Inside one frame
// body nothing else can change the answer, so a caller that asks
// PORT_SPIN_POLLS times without a VBlank in between is a loop spinning until
// the next interrupt: on hardware the VBlank (and the line-150 VCount) fire
// during that spin, i.e. the frame lags. They are delivered here, then the
// predicate is asked again. Spin iterations that only poll have no other
// effect, so where in the spin the interrupts land does not matter: the
// result equals the hardware's. Non-blocking callers ask once per frame and
// never reach the threshold (the count restarts with every frame and VBlank).
#define PORT_SPIN_POLLS 64

// With a function-call anchor pending (differential testing), the hardware's
// VBlank of this wait comes where that anchor fires (its function entry, e.g.
// the k-th poll of the caller's loop): the spin goes on until then, so the
// body polls as often as the hardware's (bounded, in case it never fires).
#define PORT_SPIN_POLLS_ANCHORED_MAX 65536

// A blocking loop's iteration that made no progress (a side-effect-free poll
// that only an interrupt can change: DoMapLoadLoop): normally the VBlank is
// delivered at once; with a function-call anchor pending the loop polls again,
// as often as the hardware's, until the anchor delivers it (bounded).
void Port_PollUntilVBlank(void)
{
    if (gPortLagAnchorsActive && CallAnchorPending() && ++sSpinPolls < PORT_SPIN_POLLS_ANCHORED_MAX)
        return;
    Port_IdleUntilVBlank();
}

int Port_SpinPoll(const void *site)
{
    if (++sSpinPolls < PORT_SPIN_POLLS)
        return 0;
    if (gPortLagAnchorsActive && CallAnchorPending() && sSpinPolls < PORT_SPIN_POLLS_ANCHORED_MAX)
        return 0;
    Port_FaultNoteExplicit(PN_SITE_SPIN_WAIT, 0, 0, site);
    Port_IdleUntilVBlank();
    return 1;
}

// Called by AgbMain_RunFrame right after the callbacks (callback2 returned),
// before PlayTimeCounter_Update/MapMusicMain. On a frame whose body is longer
// than one frame on hardware, the VCount and VBlank interrupts fire inside the
// body; the lag profile (from the mGBA oracle) says how many times.
void Port_LagHook(void)
{
    // Function anchors may lie in PlayTimeCounter_Update/MapMusicMain or at
    // WaitForVBlank, after this hook. Let those entries deliver their VBlanks;
    // flushing here would discard the anchor's placement and sound locks.
    if (sCallAnchorsArmed && sLagAnchorNext < sLagAnchorCount)
        return;
    int i, n = sFrameLag;
    sFrameLag = 0;
    for (i = 0; i < n; i++)
    {
        ServiceVCount();
        ServiceVBlank();
        sSpinPolls = 0;
    }
    if (n > 0)
        Port_SetScanline(gPortBodyScanline);
}

// ---- one frame ----------------------------------------------------------------------



// Runs on the game stack.
static void FrameBody(void)
{
    sInGameCode = 1;
    if (__builtin_setjmp(sFrameJmp))
    {
        sInGameCode = 0;
        sBodyResult = sJumpResult;
        Port_LeaveGameStack();
    }

    sIdleVBlanksThisFrame = 0;
    sSpinPolls = 0;
    if (sArgVBlankPending)
        ServiceVBlank();

    // KEYINPUT is active low.
    PORT_IO16(REG_OFFSET_KEYINPUT) = ~sArgKeys & KEYS_MASK;
    Port_SetScanline(gPortBodyScanline);

    AgbMain_RunFrame();
    if (sLagAnchorCount)
        CheckCallAnchors();
    if (sCallAnchorsArmed)
    {
        // Record missed entries above before applying their approximate
        // end-of-body fallback. Honor recorded sound locks even for a miss.
        while (sFrameLag > 0 && sLagAnchorNext < sLagAnchorCount)
            DeliverAnchoredVBlank();
        sCallAnchorsArmed = 0;
        Port_LagHook();
    }
    sFrameLag = 0;
    gPortLagAnchorsActive = 0;
    sLagAnchorCount = 0;
    sCallAnchorsArmed = 0;
    sCallFnCount = 0;
    sExitAnchorFn = 0;

    ServiceVCount();
    sBodyResult = RaiseVBlankAtBoundary();
    sInGameCode = 0;
}

uint32_t Port_RunFrame(int vblank_pending, uint16_t keys)
{
    // An ended environment stays ended (docs/ROBUSTNESS.md).
    if (sTerminal.kind != PORT_TERMINAL_NONE)
        return PORT_FRAME_TERMINAL;
    sArgVBlankPending = vblank_pending;
    sArgKeys = keys;
    RunOnGameStack(FrameBody);
    if (!(sBodyResult & PORT_FRAME_TERMINAL))
        sBodyResult |= CheckGameStack();
    Port_FaultFlushReports();
    return (sBodyResult & PORT_FRAME_TERMINAL) ? PORT_FRAME_TERMINAL : sBodyResult;
}

// ---- testing: arbitrary code as game code -------------------------------------------

HOST_STATE static void (*sTestFn)(void);

// Runs on the game stack.
static void TestBody(void)
{
    sInGameCode = 1;
    if (__builtin_setjmp(sFrameJmp))
    {
        sInGameCode = 0;
        sBodyResult = sJumpResult;
        Port_LeaveGameStack();
    }
    sIdleVBlanksThisFrame = 0;
    sSpinPolls = 0;
    sTestFn();
    sInGameCode = 0;
    sBodyResult = 0;
}

uint32_t Port_TestRunGameCode(void (*fn)(void))
{
    if (sTerminal.kind != PORT_TERMINAL_NONE)
        return PORT_FRAME_TERMINAL;
    sTestFn = fn;
    RunOnGameStack(TestBody);
    if (!(sBodyResult & PORT_FRAME_TERMINAL))
        sBodyResult |= CheckGameStack();
    Port_FaultFlushReports();
    return (sBodyResult & PORT_FRAME_TERMINAL) ? PORT_FRAME_TERMINAL : sBodyResult;
}

// ---- timer 1 (REG_TM1CNT_L) ---------------------------------------------------------
//
// SeedRngAndSetTrainerId reads the free-running timer 1 (title screen START and
// naming screen exit; the second read sets the trainer ID). Its value is a
// host-supplied sequence: read k returns sTimer1Values[min(k, count - 1)]. The
// sequence and the read position are game-visible state, so they live in the
// state (.pnstate, part of snapshots).
#define PORT_TIMER1_VALUES_MAX 16
static u16 sTimer1Values[PORT_TIMER1_VALUES_MAX];
static u8 sTimer1Count;
static u8 sTimer1Next;

void Port_SetTimer1Values(const uint16_t *values, int count, int next)
{
    int i;
    if (count < 1 || count > PORT_TIMER1_VALUES_MAX)
        Port_Fatal("timer 1 value sequence: %d values (1..%d)", count, PORT_TIMER1_VALUES_MAX);
    if (next < 0 || next > count)
        Port_Fatal("timer 1 value sequence: bad read position %d", next);
    for (i = 0; i < PORT_TIMER1_VALUES_MAX; i++)
        sTimer1Values[i] = i < count ? values[i] : 0;
    sTimer1Count = count;
    sTimer1Next = next;
}

int Port_GetTimer1Values(uint16_t *values, int *next)
{
    int i;
    for (i = 0; i < sTimer1Count; i++)
        values[i] = sTimer1Values[i];
    if (next)
        *next = sTimer1Next;
    return sTimer1Count;
}

u16 Port_ReadTimer1Counter(void)
{
    u16 value;
    if (sTimer1Count == 0)
        Port_Fatal("timer 1 read before Port_SetTimer1Values");
    value = sTimer1Values[sTimer1Next < sTimer1Count ? sTimer1Next : sTimer1Count - 1];
    if (sTimer1Next < sTimer1Count)
        sTimer1Next++;
    // The caller stops the timer right after the read: the counter keeps
    // showing this value.
    PORT_IO16(REG_OFFSET_TM1CNT_L) = value;
    return value;
}
