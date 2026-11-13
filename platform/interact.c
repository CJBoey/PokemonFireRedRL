// fireredsim platform layer: the overworld-interaction latch (observer v6,
// reward v0.14; fireredsim_port.h Port_TakeInteraction) and the consumable
// item-use latch (skill events use_consumable, 2026-10-28, and use_battle_item,
// 2026-10-29; Port_TakeItemUse).
//
// game/src/field_control_avatar.c (PORTABLE) stores into gPortInteractionLatch
// (PORT_NOTE_INTERACTION, gba/portable_abi.h) when it has chosen the script of
// an object event, a background event or a scripted sign walked into. The latch
// is HOST_STATE: never part of the state regions (not saved, restored,
// snapshotted or hashed) and never read by game logic, so it cannot change the
// game's behaviour.
#include "port_internal.h"
#include "fieldmap.h"

HOST_STATE struct PortInteractionLatch gPortInteractionLatch;
HOST_STATE struct PortItemUseLatch gPortItemUseLatch;

void Port_TakeInteraction(struct PortInteraction *out)
{
    if (out != NULL)
    {
        out->kind = gPortInteractionLatch.kind;
        out->count = gPortInteractionLatch.count;
        out->mapGroup = gPortInteractionLatch.mapGroup;
        out->mapNum = gPortInteractionLatch.mapNum;
        out->localId = gPortInteractionLatch.localId;
        out->reserved[0] = out->reserved[1] = out->reserved[2] = 0;
        // the hook stores map-grid coordinates (fewer instructions in the hooked functions)
        out->x = gPortInteractionLatch.count ? (int16_t)(gPortInteractionLatch.x - MAP_OFFSET) : 0;
        out->y = gPortInteractionLatch.count ? (int16_t)(gPortInteractionLatch.y - MAP_OFFSET) : 0;
    }
    gPortInteractionLatch.kind = 0;
    gPortInteractionLatch.count = 0;
    gPortInteractionLatch.mapGroup = 0;
    gPortInteractionLatch.mapNum = 0;
    gPortInteractionLatch.localId = 0;
    gPortInteractionLatch.x = 0;
    gPortInteractionLatch.y = 0;
}

void Port_TakeItemUse(struct PortItemUse *out)
{
    if (out != NULL)
    {
        out->item = gPortItemUseLatch.count ? gPortItemUseLatch.item : 0;
        out->count = gPortItemUseLatch.count;
        out->battle = gPortItemUseLatch.battle;
    }
    gPortItemUseLatch.item = 0;
    gPortItemUseLatch.count = 0;
    gPortItemUseLatch.battle = 0;
}
