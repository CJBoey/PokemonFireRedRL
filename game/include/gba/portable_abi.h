#ifndef GUARD_GBA_PORTABLE_ABI_H
#define GUARD_GBA_PORTABLE_ABI_H

// fireredsim: force-included (gcc -include) into every game and platform C file
// of the native build. Only active with PORTABLE; the ARM build never sees it.
//
// The retail ROM was compiled by agbcc for ARM with the old APCS ABI, where every
// struct and union is aligned to at least 4 bytes and its size is rounded up to a
// multiple of 4 (STRUCTURE_SIZE_BOUNDARY == 32; pret's MODERN build with
// -mabi=apcs-gnu behaves identically). i386 GCC uses natural alignment, so e.g.
// struct { u8 a, b, c; } is 3 bytes instead of 4 and union AffineAnimCmd is 6
// bytes instead of 8. Data tables emitted by the assembler (data/*.s), save data
// and code that walks memory by hand all assume the ARM layout, so the native
// build reproduces the ARM rule by attaching aligned(4) to every struct/union
// definition. For non-definitions (struct Foo *p) GCC ignores the attribute.
// __attribute__((packed)) definitions are exempt on ARM; those few definitions
// are wrapped in #pragma push_macro("struct") ... pop_macro in the headers.
//
// All system headers that game code uses are included here first so that libc
// structs keep the host ABI (their include guards make later includes no-ops).

#ifdef PORTABLE
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>

// Fatal conditions inside game code (platform/port.c). Used where the retail
// game would crash or hang forever, and for features the simulator does not
// support (link cable, wireless adapter, e-Reader, ...): fail loudly, never fake.
void Port_Fatal(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
void Port_Unsupported(const char *what) __attribute__((noreturn));
// Inside a frame these end the environment with a terminal status instead of
// the process (docs/ROBUSTNESS.md): Port_Fatal = simulator error,
// Port_Unsupported = feature not emulated, Port_Hang = the real game hangs
// forever here, Port_Crash = the real game crashes / runs garbage here,
// Port_Inexact = the result depends on hardware behaviour that cannot be
// reproduced natively (CPU open bus).
void Port_Hang(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
void Port_Crash(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
void Port_Inexact(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
// Hardware busy-wait with interrupts enabled during boot (platform/frame.c).
void Port_BootIdle(void);
// Boot interrupt schedule hooks in m4a.c, called with the sound engine lock
// held: site 0 = m4aSoundMode, site 1 = MPlayOpen (platform/frame.c).
void Port_BootSoundLocked(int site);
// Lag emulation hook in AgbMain_RunFrame after the callbacks (platform/frame.c).
void Port_LagHook(void);
// A loop inside a frame body that can only progress through the VBlank
// interrupt (e.g. waiting for the DMA3 queue that VBlankIntr processes): on
// hardware the loop keeps spinning while frames pass. Delivers the next
// VCount and VBlank interrupts (platform/frame.c).
void Port_IdleUntilVBlank(void);
// The same for an iteration of a polling loop that made no progress (the poll
// has no side effect): differential tests with function-call anchors let the
// loop poll as often as on hardware first (docs/LAG.md).
void Port_PollUntilVBlank(void);
// A predicate that only an interrupt handler can change answered "still
// waiting" (`site`: where it was asked). After PORT_SPIN_POLLS such answers
// without a VBlank in between (a busy-wait loop inside the frame body) the
// VCount and VBlank interrupts are delivered and 1 is returned: ask again.
// platform/frame.c, docs/ROBUSTNESS.md.
int Port_SpinPoll(const void *site);
// TRUE if p points into the stack the game code runs on (platform/frame.c; in
// IWRAM on the GBA, in native-only memory natively).
int Port_IsOnGameStack(const void *p);
// A read of the free-running timer 1 counter (REG_TM1CNT_L): the next value of
// the environment's host-supplied sequence (platform/frame.c).
unsigned short Port_ReadTimer1Counter(void);
// Value of a read from the GBA BIOS area (address < 0x4000, e.g. through a NULL
// struct pointer) while code runs outside the BIOS: the BIOS "open bus", i.e.
// the last opcode the CPU prefetched from the BIOS (0xE3A02004 after a BIOS
// call, 0xE25EF004 inside an interrupt handler, 0xE55EC002 after one, as in
// mGBA). platform/bios.c.
// A retail load of `size` (1/2/4) bytes through `ptr`, which may point into
// unmapped GBA memory (a pointer read from the BIOS region, e.g. via NULL). On
// the GBA the CPU open bus there is the Thumb opcode two instructions after the
// load instruction in the ROM (`thumbOpcodeAtPcPlus4`, from the retail
// disassembly); readable memory is read normally (platform/memfault.c).
unsigned int Port_LoadWithOpenBus(const void *ptr, unsigned int size, unsigned int thumbOpcodeAtPcPlus4);
unsigned int Port_BiosOpenBus32(unsigned int addr);
unsigned short Port_BiosOpenBus16(unsigned int addr);
// MPlayStart: in logic-only mixing (platform/sound.c) a song header (with the track pointers
// of up to playerTracks tracks) inside gSoundInfo.pcmBuffer ends the environment as INEXACT.
void Port_SoundSongCheck(const void *songHeader, unsigned int playerTracks);
// Other sound-engine reads in game code (CgbSound's wave copy, ply_memacc): the same check.
void Port_SoundReadCheck(const void *ptr, unsigned int size);
unsigned char Port_BiosOpenBus8(unsigned int addr);
// Observation hook (no game-visible effect): field_control_avatar.c reports the overworld
// interaction whose script it has just chosen (not during Quest Log playback) into
// gPortInteractionLatch (platform/interact.c; HOST_STATE: host memory outside the state regions,
// never read by game logic): the first one since the host's last Port_TakeInteraction
// (fireredsim_port.h) and a saturating count. Kinds: the values of obs/observation.h enum
// PnInteractKind (static_assert in obs/src/observe.c). OBJECT: the object's own map, local id and
// map-grid coordinates; SIGN / HIDDEN_ITEM / SIGN_WALK: the current map and the background event's
// map-grid coordinates, localId 0 (x, y: grid coordinates, i.e. map coordinates + MAP_OFFSET;
// Port_TakeInteraction subtracts it). PORT_NOTE_INTERACTION is plain stores, not a call: a call grew
// the hooked functions' stack frames (outgoing arguments, a spill), i.e. it would change the stale
// native-stack bytes that game code can copy into game memory (uninitialised locals); with the
// stores both functions keep the frames of the build without the hook (checked at -O2).
struct PortInteractionLatch
{
    unsigned char kind, count, mapGroup, mapNum, localId, reserved[3];
    short x, y;
};
extern struct PortInteractionLatch gPortInteractionLatch;
enum { PORT_INTERACT_OBJECT = 1, PORT_INTERACT_SIGN = 2, PORT_INTERACT_HIDDEN_ITEM = 3, PORT_INTERACT_SIGN_WALK = 4 };
#define PORT_NOTE_INTERACTION(k, group, num, id, px, py)                     \
    do {                                                                     \
        if (gPortInteractionLatch.count == 0)                                \
        {                                                                    \
            gPortInteractionLatch.kind = (unsigned char)(k);                 \
            gPortInteractionLatch.mapGroup = (unsigned char)(group);         \
            gPortInteractionLatch.mapNum = (unsigned char)(num);             \
            gPortInteractionLatch.localId = (unsigned char)(id);             \
            gPortInteractionLatch.x = (short)(px);                           \
            gPortInteractionLatch.y = (short)(py);                           \
        }                                                                    \
        if (gPortInteractionLatch.count != 255)                              \
            gPortInteractionLatch.count++;                                   \
    } while (0)

// Observation hook (no game-visible effect): the paths that CONSUME an item the player used
// (item_use.c RemoveUsedItem: Repel, Escape Rope, and in battle the Poke Doll / Fluffy Tail;
// party_menu.c: medicine, PP restorers, Rare Candy, Sacred Ash, evolution stones, the medicine and PP
// paths also from the battle's party menu; item_use.c in battle: X items, Guard Spec, Dire Hit: at
// their RemoveBagItem) report the item id into gPortItemUseLatch (platform/interact.c; HOST_STATE,
// never read by game logic): the first one since the host's last Port_TakeItemUse
// (fireredsim_port.h), a saturating count, and of those the ones made while gMain.inBattle (battle;
// expanded where main.h is included). Not reported: a use the game refused (nothing removed),
// reusable items (flutes, key items, HMs), TMs (the move learned is observed separately), toss /
// sell / give / deposit, and Poke Balls (2026-10-29, user). Plain stores, as above.
struct PortItemUseLatch
{
    unsigned short item;
    unsigned char count, battle;
};
extern struct PortItemUseLatch gPortItemUseLatch;
#define PORT_NOTE_ITEM_USE(id)                                               \
    do {                                                                     \
        if (gPortItemUseLatch.count == 0)                                    \
            gPortItemUseLatch.item = (unsigned short)(id);                   \
        if (gPortItemUseLatch.count != 255)                                  \
        {                                                                    \
            gPortItemUseLatch.count++;                                       \
            if (gMain.inBattle)                                              \
                gPortItemUseLatch.battle++;                                  \
        }                                                                    \
    } while (0)

// Narrow return values (game/NATIVE_CHANGES.md, "ABI / layout"). agbcc's callee
// extends a u8/s8/u16/s16 (bool8, bool16) return value to all of r0 (lsl+lsr or
// lsl+asr, or a load that extends); GCC/i386 only defines AL/AX and leaves the
// extension to the caller. A caller that calls such a function through a wider
// type (a script in assembly data names it and game code calls it through
// u32 (*)(void)) reads the upper bits of EAX, which hold garbage natively. Such
// dispatchers call through the function's real type instead: PORT_RET_KIND
// classifies a function by its declaration (so the compiler, not a hand-made
// list, decides the extension), Port_CallNativeExtended looks the function up in
// a table of PORT_NATIVE_FUNC entries and returns the value the ROM's r0 holds.
// A function missing from the table is a simulator error (Port_Fatal).
enum { PORT_RET_WORD, PORT_RET_U8, PORT_RET_S8, PORT_RET_U16, PORT_RET_S16, PORT_RET_VOID };
#define PORT_RET_KIND(fn) _Generic(&(fn),                                          \
    unsigned char (*)(void): PORT_RET_U8, signed char (*)(void): PORT_RET_S8,     \
    unsigned short (*)(void): PORT_RET_U16, short (*)(void): PORT_RET_S16,        \
    void (*)(void): PORT_RET_VOID, default: PORT_RET_WORD)
struct PortNativeFunc { void (*func)(void); unsigned int retKind; };
#define PORT_NATIVE_FUNC(fn) { (void (*)(void))(fn), PORT_RET_KIND(fn) }
// Not a game function: excluded from the differential-test build's function-entry
// hooks (-finstrument-functions, docs/LAG.md) like platform code.
__attribute__((no_instrument_function))
static inline unsigned int Port_CallNativeExtended(const struct PortNativeFunc *table, unsigned int count,
                                                   void (*func)(void), const char *what)
{
    unsigned int i;
    for (i = 0; i < count; i++)
    {
        if (table[i].func != func)
            continue;
        switch (table[i].retKind)
        {
        case PORT_RET_U8: return ((unsigned char (*)(void))func)();
        case PORT_RET_S8: return (unsigned int)(int)((signed char (*)(void))func)();
        case PORT_RET_U16: return ((unsigned short (*)(void))func)();
        case PORT_RET_S16: return (unsigned int)(int)((short (*)(void))func)();
        // r0 after a void function is whatever its last instruction left there
        // (not reproducible in general); callers of such entries ignore it.
        case PORT_RET_VOID: func(); return 0;
        default: return ((unsigned int (*)(void))func)();
        }
    }
    Port_Fatal("%s: native function %p is not in the dispatcher's table", what, (void *)func);
}

#define struct struct __attribute__((aligned(4)))
#define union union __attribute__((aligned(4)))
#endif // PORTABLE

#endif // GUARD_GBA_PORTABLE_ABI_H
