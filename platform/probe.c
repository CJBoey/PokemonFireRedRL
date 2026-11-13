// fireredsim platform layer: read-only probes of game state for the host
// (CLI --probe-every, tests, the adaptive input driver). Uses the game's own
// headers so that struct layouts and accessors are the game's.
#include "port_internal.h"
#include "malloc.h"
#include "random.h"
#include "text.h"
#include "task.h"
#include "script.h"
#include "field_player_avatar.h"

struct MemBlockHeader
{
    bool16 flag;
    u16 magic_number;
    u32 size;
    struct MemBlockHeader *prev;
    struct MemBlockHeader *next;
};

void Port_Probe(struct PortProbe *out)
{
    u32 i;
    struct MemBlockHeader *head, *pos;

    memset(out, 0, sizeof(*out));
    out->callback1 = (u32)gMain.callback1;
    out->callback2 = (u32)gMain.callback2;
    out->vblankCallback = (u32)gMain.vblankCallback;
    out->vblankCounter2 = gMain.vblankCounter2;
    out->rngValue = gRngValue;
    out->mainState = gMain.state;
    out->inBattle = gMain.inBattle;
    out->heldKeys = gMain.heldKeys;
    out->newKeys = gMain.newKeys;
    out->saveBlock1 = (u32)gSaveBlock1Ptr;
    if ((u32)gSaveBlock1Ptr >= EWRAM_START && (u32)gSaveBlock1Ptr + sizeof(struct SaveBlock1) <= EWRAM_END)
    {
        out->x = gSaveBlock1Ptr->pos.x;
        out->y = gSaveBlock1Ptr->pos.y;
        out->mapGroup = gSaveBlock1Ptr->location.mapGroup;
        out->mapNum = gSaveBlock1Ptr->location.mapNum;
    }
    if (gPlayerAvatar.objectEventId < OBJECT_EVENTS_COUNT && gObjectEvents[gPlayerAvatar.objectEventId].active)
        out->facing = gObjectEvents[gPlayerAvatar.objectEventId].facingDirection;
    for (i = 0; i < NUM_TEXT_PRINTERS; i++)
        if (IsTextPrinterActive(i))
            out->textPrinterActive = 1;
    out->scriptEnabled = ScriptContext_IsEnabled();
    out->fieldControlsLocked = ArePlayerFieldControlsLocked();
    for (i = 0; i < NUM_TASKS && i < 16; i++)
        if (gTasks[i].isActive)
            out->taskFuncs[i] = (u32)gTasks[i].func;

    // Walk the gHeap block list (malloc.c) without modifying it. This runs on
    // the host stack: never follow a link outside gHeap (a corrupted heap must
    // not crash the host; robustness, docs/ROBUSTNESS.md).
    head = (struct MemBlockHeader *)gHeap;
    pos = head;
    for (i = 0; i < 4096; i++)
    {
        if (pos->flag)
        {
            out->heapUsed += pos->size + sizeof(struct MemBlockHeader);
            out->heapBlocks++;
        }
        if (pos->next == head || pos->next == NULL)
            break;
        if ((u8 *)pos->next < gHeap || (u8 *)pos->next > gHeap + HEAP_SIZE - sizeof(struct MemBlockHeader)
            || ((u32)pos->next & 3))
            break;
        pos = pos->next;
    }
}
