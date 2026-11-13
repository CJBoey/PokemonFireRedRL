// fireredsim platform layer: C port of game/src/libgcnmultiboot.s for the
// situation the simulator models: no GameCube connected to the link port.
//
// The copyright screen calls GameCubeMultiBoot_Init once and _Main every
// frame for about 140 frames, then _Quit. Without a GameCube no JOY Bus command
// ever arrives, the serial interrupt never fires and the state machine only
// runs its idle path, which is ported exactly below (counters, the periodic
// re-initialisation, register writes). Every path that requires data from a
// GameCube (handshake, image download, boot) is unreachable without a serial
// interrupt and is reported as unsupported if it is ever entered.
//
// Like game/native/pokefirered.patch, this file is derived from the game's own
// code (through pret's disassembly) and is not covered by this repository's
// MIT licence (THIRD_PARTY.md).
#include "port_internal.h"
#include "libgcnmultiboot.h"

#define GCMB_STRUCT_COUNTER1    0x00
#define GCMB_STRUCT_COUNTER2    0x01
#define GCMB_STRUCT_MBPROGRESS  0x02
#define GCMB_STRUCT_SAVEDVCOUNT 0x03
#define GCMB_STRUCT_BASE_DEST   0x20
#define GCMB_STRUCT_CUR_DEST    0x24
#define GCMB_STRUCT_HANDLER     0x28

#define MBPROGRESS_NONE          0
#define MBPROGRESS_READY_TO_BOOT 2

#define FIELD8(mb, off)  (((u8 *)(mb))[off])
#define FIELD32(mb, off) (*(u32 *)((u8 *)(mb) + (off)))

// JOYCNT: bits 0-2 are acknowledged by writing 1, bit 6 is the IRQ enable.
static void WriteJoyCnt(u16 value)
{
    u16 old = PORT_IO16(REG_OFFSET_JOYCNT);
    PORT_IO16(REG_OFFSET_JOYCNT) = (value & 0x40) | (old & ~(value & 0x7) & ~0x40);
}

// GcMbIntrHandler_Stop. Its address is stored in the struct as the serial
// handler, like the ROM stores the address of its Thumb routine (pointer
// fields differ between ROM and native by construction; see docs/DEVLOG.md).
static void GcMbIntrHandler_Stop(void)
{
    Port_Unsupported("GameCube multiboot serial handler (no GameCube connected)");
}

void GameCubeMultiBoot_Init(struct GcmbStruct *mb)
{
    u16 ime = PORT_IO16(REG_OFFSET_IME);
    u8 savedVCount, counter2;
    u32 i;

    PORT_IO16(REG_OFFSET_IME) = 0;
    FIELD32(mb, GCMB_STRUCT_HANDLER) = (u32)GcMbIntrHandler_Stop;
    savedVCount = FIELD8(mb, GCMB_STRUCT_SAVEDVCOUNT);
    counter2 = FIELD8(mb, GCMB_STRUCT_COUNTER2);
    // Clear all but the last three words (base/cur dest pointers, handler).
    for (i = 0; i < GCMB_STRUCT_BASE_DEST; i += 4)
        FIELD32(mb, i) = 0;
    FIELD8(mb, GCMB_STRUCT_SAVEDVCOUNT) = counter2 >> 1;
    FIELD8(mb, GCMB_STRUCT_COUNTER2) = savedVCount;

    PORT_IO16(REG_OFFSET_RCNT) = 0x8000; // JOY Bus off ((REG_BASE + 0x120) << 10)
    PORT_IO16(REG_OFFSET_RCNT) = 0xC000; // JOY Bus on
    WriteJoyCnt(0x47);
    PORT_IO16(REG_OFFSET_JOYSTAT) = 0;
    PORT_IO16(REG_OFFSET_IF) &= ~INTR_FLAG_SERIAL; // acknowledge (write-1-to-clear)
    PORT_IO16(REG_OFFSET_IE) |= INTR_FLAG_SERIAL;
    PORT_IO16(REG_OFFSET_IME) = ime;
}

void GameCubeMultiBoot_Main(struct GcmbStruct *mb)
{
    bool32 reinit;

    if (FIELD32(mb, GCMB_STRUCT_HANDLER) == 0)
    {
        // cmp r1, 0 -> carry set -> bcs GameCubeMultiBoot_Init
        reinit = TRUE;
    }
    else
    {
        u16 ime;
        FIELD8(mb, GCMB_STRUCT_COUNTER2)++;
        if (FIELD8(mb, GCMB_STRUCT_MBPROGRESS) == MBPROGRESS_READY_TO_BOOT)
            return;
        ime = PORT_IO16(REG_OFFSET_IME);
        PORT_IO16(REG_OFFSET_IME) = 0;
        // cmp r1, 0xA; bgt (skip increment, carry set) / adds (carry clear)
        if (FIELD8(mb, GCMB_STRUCT_COUNTER1) > 10)
        {
            reinit = TRUE;
        }
        else
        {
            FIELD8(mb, GCMB_STRUCT_COUNTER1)++;
            reinit = FALSE;
        }
        PORT_IO16(REG_OFFSET_IME) = ime;
    }
    if (reinit)
    {
        GameCubeMultiBoot_Init(mb);
        return;
    }
    if (FIELD8(mb, GCMB_STRUCT_MBPROGRESS) != MBPROGRESS_NONE)
        Port_Unsupported("GameCube multiboot image transfer");
    if (FIELD32(mb, GCMB_STRUCT_CUR_DEST) - FIELD32(mb, GCMB_STRUCT_BASE_DEST) == 0)
        return; // no image data received
    Port_Unsupported("GameCube multiboot image transfer");
}

void GameCubeMultiBoot_ExecuteProgram(struct GcmbStruct *mb)
{
    if (FIELD8(mb, GCMB_STRUCT_MBPROGRESS) != MBPROGRESS_READY_TO_BOOT)
        return;
    Port_Unsupported("GameCube multiboot program execution");
}

void GameCubeMultiBoot_HandleSerialInterrupt(struct GcmbStruct *mb)
{
    Port_Unsupported("GameCube multiboot serial interrupt");
}

void GameCubeMultiBoot_Quit(void)
{
    u16 ime = PORT_IO16(REG_OFFSET_IME);
    PORT_IO16(REG_OFFSET_IME) = 0;
    WriteJoyCnt(0x7); // acknowledge all JOYCNT flags
    PORT_IO16(REG_OFFSET_RCNT) = 0x8000;
    PORT_IO16(REG_OFFSET_IF) &= ~INTR_FLAG_SERIAL;
    PORT_IO16(REG_OFFSET_IE) &= ~INTR_FLAG_SERIAL;
    PORT_IO16(REG_OFFSET_IME) = ime;
}
