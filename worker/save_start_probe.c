// fireredsim worker: read-only probe for the save start driver (save_start_probe.h).
// Compiled with the game's headers and flags; only pure game functions are called and
// nothing in game memory is written.
#include "global.h"
#include "main.h"
#include "task.h"
#include "palette.h"
#include "save.h"
#include "script.h"
#include "overworld.h"
#include "quest_log.h"
#include "event_object_movement.h"
#include "field_player_avatar.h"

#include "save_start_probe.h"

void pn_save_start_probe(PnSaveStartProbe *out)
{
    u32 i;

    memset(out, 0, sizeof(*out));
    out->callback2 = (u32)gMain.callback2;
    for (i = 0; i < NUM_TASKS && i < PN_SS_TASKS; i++)
    {
        if (gTasks[i].isActive)
        {
            out->taskFunc[i] = (u32)gTasks[i].func;
            out->taskData1[i] = gTasks[i].data[1];
        }
    }
    out->saveFileStatus = gSaveFileStatus;
    out->heldKeys = gMain.heldKeys;
    out->paletteFadeActive = gPaletteFade.active ? 1 : 0;
    out->questLogState = gQuestLogState;
    out->scriptEnabled = ScriptContext_IsEnabled() ? 1 : 0;
    out->fieldControlsLocked = ArePlayerFieldControlsLocked() ? 1 : 0;
    out->playerObjectActive = gPlayerAvatar.objectEventId < OBJECT_EVENTS_COUNT
                              && gObjectEvents[gPlayerAvatar.objectEventId].active;
}

uint16_t pn_save_status_ok(void)
{
    return SAVE_STATUS_OK;
}
