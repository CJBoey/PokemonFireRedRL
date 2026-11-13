/* fireredsim worker: read-only game-state probe for the save start driver
 * (EnvPool, PN_TEMPLATE_SAVE: power-on with a .sav, title, main menu CONTINUE,
 * Quest Log recap skipped, overworld). Compiled with the game's headers and
 * flags (pn_add_game_reader_library, cmake/GameReader.cmake); reads the live
 * state regions, never writes game memory. */
#ifndef PN_SAVE_START_PROBE_H
#define PN_SAVE_START_PROBE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_SS_TASKS 16

typedef struct PnSaveStartProbe {
    uint32_t callback2;                /* gMain.callback2 */
    uint32_t taskFunc[PN_SS_TASKS];    /* gTasks[i].func of active tasks, 0 otherwise */
    int16_t taskData1[PN_SS_TASKS];    /* gTasks[i].data[1] (main menu: tCursorPos) */
    uint16_t saveFileStatus;           /* gSaveFileStatus (SAVE_STATUS_OK = 1) */
    uint16_t heldKeys;                 /* gMain.heldKeys */
    uint8_t paletteFadeActive;         /* gPaletteFade.active */
    uint8_t questLogState;             /* gQuestLogState (2, 3: Quest Log playback) */
    uint8_t scriptEnabled;             /* ScriptContext_IsEnabled() */
    uint8_t fieldControlsLocked;       /* ArePlayerFieldControlsLocked() */
    uint8_t playerObjectActive;        /* the player's object event exists */
    uint8_t pad[3];
} PnSaveStartProbe;

void pn_save_start_probe(PnSaveStartProbe *out);

/* Game constants the driver needs (from the game's headers). */
uint16_t pn_save_status_ok(void);       /* SAVE_STATUS_OK */

#ifdef __cplusplus
}
#endif

#endif /* PN_SAVE_START_PROBE_H */
