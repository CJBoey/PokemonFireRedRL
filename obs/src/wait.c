/* fireredsim observer: screen classification and input-wait derivation.
 *
 * Implements docs/OBS_SOURCES.md section 3.2 (screen_kind) and section 2.1 (pn_wait):
 * waiting_for_input = 1 is produced only by an explicit predicate below (whitelist);
 * every predicate contains the global gate G (section 2.9), the identity of
 * gMain.callback2, and for every task it names isActive && func == fn (never a cached
 * id). Unknown screens, task functions and states give 0. Only pure reads are used.
 *
 * Each predicate also sets input_keys: the keys the game acts on in that context.
 * A key in the set may still be inert in a particular cursor position (e.g. A facing
 * nothing on the field); the false-positive audit (obs/tests) requires that at least
 * one key of the set changes the game state.
 */
#include "global.h"
#include "main.h"
#include "task.h"
#include "text.h"
#include "script.h"
#include "palette.h"
#include "window.h"
#include "bg.h"
#include "link.h"
#include "menu.h"
#include "list_menu.h"
#include "event_data.h"
#include "event_object_movement.h"
#include "field_player_avatar.h"
#include "fieldmap.h"
#include "metatile_behavior.h"
#include "overworld.h"
#include "quest_log.h"
#include "help_system.h"
#include "item_menu.h"
#include "item.h"
#include "constants/item_menu.h"
#include "constants/party_menu.h"
#include "party_menu.h"
#include "pokemon_summary_screen.h"
#include "pokemon_storage_system_internal.h"
#include "battle.h"
#include "battle_main.h"
#include "battle_anim.h"
#include "battle_script_commands.h"
#include "pokemon_special_anim_internal.h"
#include "constants/quest_log.h"
#include "constants/event_object_movement.h"
#include "constants/event_objects.h"
#include "constants/trainer_types.h"
#include "constants/battle.h"
#include "constants/battle_setup.h"
#include "constants/species.h"
#include "constants/flags.h"
#include "constants/vars.h"
#include "constants/map_scripts.h"
#include "constants/global.h"
#include "constants/moves.h"

#include "../observation.h"
#include "obs_internal.h"

extern u16 *const gSpecialVars[]; /* data/event_scripts.s */
#define PN_SCRIPT_MODE_NATIVE 2 /* script.c enum: STOPPED, BYTECODE, NATIVE */
#define PN_NUM_BATTLE_SCRIPT_CMDS 0xF8 /* gBattleScriptingCommandsTable: Cmd_attackcanceler .. Cmd_finishturn */

/* ---- small helpers ----------------------------------------------------------- */

#define CB2 ((uint32_t)(uintptr_t)gMain.callback2)
#define CB1 ((uint32_t)(uintptr_t)gMain.callback1)
#define CB2_IS(id) (SYM(id) != 0 && CB2 == SYM(id))
#define TD(t, i) (gTasks[(t)].data[(i)])
#define FADE (gPaletteFade.active)
#define K_AB (PN_KEY_A | PN_KEY_B)

int pn_obs_task_find(uint32_t fn)
{
    int t;
    if (fn == 0)
        return -1;
    for (t = 0; t < NUM_TASKS; t++)
        if (gTasks[t].isActive && (uint32_t)(uintptr_t)gTasks[t].func == fn)
            return t;
    return -1;
}
#define TASK(id) pn_obs_task_find(SYM(id))
#define TASK_ON(id) (TASK(id) >= 0)

static const struct TextPrinter *printer(int w)
{
    const struct TextPrinter *p = SYMPTR(const struct TextPrinter *, sTextPrinters);
    if (p == NULL || w < 0 || w >= NUM_TEXT_PRINTERS)
        return NULL;
    return &p[w];
}

static int printer_active(int w)
{
    const struct TextPrinter *p = printer(w);
    return p != NULL && p->active;
}

/* printer w blocked on JOY_NEW(A|B) (RENDER_STATE_WAIT / CLEAR / SCROLL_START) and no
 * auto-scroll: meaningful only if something runs the printer in the next frame */
static int pr_wait(int w)
{
    const struct TextPrinter *p = printer(w);
    if (p == NULL || !p->active || gTextFlags.autoScroll)
        return 0;
    return p->state >= RENDER_STATE_WAIT && p->state <= RENDER_STATE_SCROLL_START;
}

/* struct Menu of menu.c (file-local; reference/layout_probes.tsv: 12 bytes, cursorPos @2) */
struct PnMenuMirror {
    u8 left, top;
    s8 cursorPos, minCursorPos, maxCursorPos;
    u8 windowId, fontId, optionWidth, optionHeight, columns, rows;
    bool8 APressMuted;
};
_Static_assert(sizeof(struct PnMenuMirror) == 12, "struct Menu mirror");

static const struct PnMenuMirror *smenu(void)
{
    return SYMPTR(const struct PnMenuMirror *, sMenu);
}

/* vertical Menu_* list: cursor and the keys that move it */
static uint16_t menu_keys(int wrap, uint8_t *cursor, uint8_t *count)
{
    const struct PnMenuMirror *m = smenu();
    uint16_t keys = PN_KEY_A;
    if (m == NULL)
        return keys;
    *cursor = (uint8_t)m->cursorPos;
    *count = (uint8_t)(m->maxCursorPos - m->minCursorPos + 1);
    if (wrap || m->cursorPos > m->minCursorPos)
        keys |= PN_KEY_UP;
    if (wrap || m->cursorPos < m->maxCursorPos)
        keys |= PN_KEY_DOWN;
    return keys;
}

static void set_wait(PnWait *w, int kind, int ctx, uint16_t keys)
{
    w->waiting = 1;
    w->kind = (uint8_t)kind;
    w->ctx = (uint8_t)ctx;
    w->keys = keys;
}

static void set_menu(PnWait *w, int open, uint8_t cursor, uint8_t count, uint8_t id)
{
    w->menu_open = (uint8_t)open;
    w->menu_cursor = cursor;
    w->menu_count = count;
    w->menu_id = id;
}

/* Menu_ProcessInputNoWrapAround / ...ClearOnChoose YES/NO (cursor 0 = YES) */
static void yesno_menu(PnWait *w, int ctx, int open)
{
    uint8_t cur = 0, cnt = 2;
    uint16_t keys = menu_keys(0, &cur, &cnt) | PN_KEY_B;
    set_wait(w, PN_INPUT_YESNO, ctx, keys);
    set_menu(w, open, cur, cnt, 255);
}

/* v7: the move a learn-move prompt offers and the party slot of the mon concerned (offered_move /
 * offered_slot). Both or neither: a move outside 1..MOVES_COUNT-1 or a slot outside the party is "none". */
/* v8: the Summary moves page cursor (0..4) as 1..5 (summary_cursor; 0: unknown) */
static void set_move_cursor(PnWait *w)
{
    const u8 *pos = SYMPTR(const u8 *, sMoveSelectionCursorPos_pss);
    if (pos != NULL && *pos < 5)
        w->summary_cursor = (uint8_t)(*pos + 1);
}

static void set_offer(PnWait *w, unsigned move, int slot)
{
    if (move == MOVE_NONE || move >= MOVES_COUNT || slot < 0 || slot >= PARTY_SIZE)
        return;
    w->offered_move = (uint16_t)move;
    w->offered_slot = (uint8_t)slot;
}

/* ---- v4: menu action lists (docs/OBSERVATION.md "Menu actions") ------------------------ */

static void set_actions(PnWait *w, const uint8_t *acts, int n)
{
    int i;
    if (n < 0)
        n = 0;
    if (n > PN_OBS_MENU_ACTIONS)
        n = PN_OBS_MENU_ACTIONS;
    w->menu_action_count = (uint8_t)n;
    for (i = 0; i < n; i++)
        w->menu_actions[i] = acts[i];
    for (; i < PN_OBS_MENU_ACTIONS; i++)
        w->menu_actions[i] = PN_ACTION_NONE;
}

/* bag context menu entries (constants/item_menu.h ITEMMENUACTION_*) */
static uint8_t bag_action(u8 a)
{
    switch (a) {
    case ITEMMENUACTION_USE: case ITEMMENUACTION_BATTLE_USE: case ITEMMENUACTION_CHECK:
    case ITEMMENUACTION_OPEN: case ITEMMENUACTION_OPEN_BERRIES: case ITEMMENUACTION_WALK:
        return PN_ACTION_USE;
    case ITEMMENUACTION_TOSS: return PN_ACTION_TOSS;
    case ITEMMENUACTION_REGISTER: case ITEMMENUACTION_DESELECT: return PN_ACTION_REGISTER;
    case ITEMMENUACTION_GIVE: return PN_ACTION_GIVE;
    case ITEMMENUACTION_CANCEL: return PN_ACTION_CANCEL;
    case ITEMMENUACTION_DUMMY: return PN_ACTION_NONE;
    default: return PN_ACTION_OTHER;
    }
}

/* party menu action submenu entries: enum CURSOR_OPTION_* of game/src/data/party_menu.h (SUMMARY 0,
 * SWITCH, CANCEL1, ITEM, GIVE, TAKE_ITEM, MAIL, TAKE_MAIL, READ, CANCEL2, SHIFT, SEND_OUT, ENTER,
 * NO_ENTRY, STORE, REGISTER, TRADE1, TRADE2, FIELD_MOVES = 18 + the sFieldMoves index) */
static uint8_t party_action(u8 a)
{
    static const uint8_t base[18] = {
        PN_ACTION_SUMMARY, PN_ACTION_SWITCH, PN_ACTION_CANCEL, PN_ACTION_ITEM, PN_ACTION_GIVE, PN_ACTION_TAKE,
        PN_ACTION_MAIL, PN_ACTION_TAKE, PN_ACTION_READ, PN_ACTION_CANCEL, PN_ACTION_SEND_OUT, PN_ACTION_SEND_OUT,
        PN_ACTION_ENTER, PN_ACTION_NO_ENTRY, PN_ACTION_DEPOSIT, PN_ACTION_REGISTER, PN_ACTION_TRADE, PN_ACTION_TRADE,
    };
    if (a < 18)
        return base[a];
    if (a - 18 < PN_ACTION_SWEET_SCENT - PN_ACTION_FLASH + 1)
        return (uint8_t)(PN_ACTION_FLASH + (a - 18));
    return PN_ACTION_OTHER;
}

/* Pokemon Storage System menus: struct StorageMenu.textId (MENU_TEXT_*) */
static uint8_t storage_action(int textId)
{
    switch (textId) {
    case MENU_TEXT_CANCEL: return PN_ACTION_CANCEL;
    case MENU_TEXT_STORE: return PN_ACTION_DEPOSIT;
    case MENU_TEXT_WITHDRAW: return PN_ACTION_WITHDRAW;
    case MENU_TEXT_MOVE: return PN_ACTION_MOVE;
    case MENU_TEXT_SHIFT: return PN_ACTION_SHIFT;
    case MENU_TEXT_PLACE: return PN_ACTION_PLACE;
    case MENU_TEXT_SUMMARY: return PN_ACTION_SUMMARY;
    case MENU_TEXT_RELEASE: return PN_ACTION_RELEASE;
    case MENU_TEXT_MARK: return PN_ACTION_MARK;
    case MENU_TEXT_JUMP: return PN_ACTION_JUMP;
    case MENU_TEXT_WALLPAPER: return PN_ACTION_WALLPAPER;
    case MENU_TEXT_NAME: return PN_ACTION_NAME;
    case MENU_TEXT_TAKE: return PN_ACTION_TAKE;
    case MENU_TEXT_GIVE: case MENU_TEXT_GIVE2: return PN_ACTION_GIVE;
    case MENU_TEXT_SWITCH: return PN_ACTION_SWITCH;
    case MENU_TEXT_BAG: return PN_ACTION_BAG;
    case MENU_TEXT_INFO: return PN_ACTION_INFO;
    default: return textId >= MENU_TEXT_SCENERY_1 ? PN_ACTION_WALLPAPER_SET : PN_ACTION_OTHER;
    }
}

/* the "Someone's PC" / "Bill's PC" multichoice (script_menu.c ScriptMenu_CreatePCMultichoice):
 * the storage system, the player's PC, PROF. OAK'S PC with the Pokedex, HALL OF FAME after the
 * game clear, LOG OFF */
static void someones_pc_actions(PnWait *w)
{
    uint8_t acts[5];
    int n = 0;
    acts[n++] = PN_ACTION_PC_STORAGE;
    acts[n++] = PN_ACTION_PC_PLAYER;
    if (pn_obs_raw_flag(FLAG_SYS_POKEDEX_GET) || pn_obs_raw_flag(FLAG_SYS_GAME_CLEAR))
        acts[n++] = PN_ACTION_PC_OAK;
    if (pn_obs_raw_flag(FLAG_SYS_GAME_CLEAR))
        acts[n++] = PN_ACTION_PC_HALL_OF_FAME;
    acts[n++] = PN_ACTION_CANCEL;
    set_actions(w, acts, n);
}

/* ---- gates (section 2.9) --------------------------------------------------------- */

/* struct HelpSystemVideoState.state (help_system_util.c, @0x15): 0 closed; 1-7 open/close
 * animation and the menu (5); 8 = last closing step: RunHelpSystemCallback restores the
 * vblank/hblank callbacks, sets the state to 0 and returns 0, so the screen's callbacks
 * run in that same frame: treated as closed. */
static u8 help_state(void)
{
    const u8 *vs = SYMPTR(const u8 *, sVideoState);
    return vs != NULL ? vs[0x15] : 0;
}

static int help_open(void)
{
    u8 st = help_state();
    return st != 0 && st != 8;
}

static int save_failed(void)
{
    const bool32 *in = SYMPTR(const bool32 *, sIsInSaveFailedScreen);
    const u8 *st = SYMPTR(const u8 *, sSaveFailedScreenState);
    return (in != NULL && *in) || (st != NULL && *st != 0);
}

static int link_active(void)
{
    return (SYM(CB1_UpdateLinkState) != 0 && CB1 == SYM(CB1_UpdateLinkState)) || gReceivedRemoteLinkPlayers == 1;
}

int pn_obs_quest_log_playback(void)
{
    return gQuestLogState == QL_STATE_PLAYBACK || gQuestLogState == QL_STATE_PLAYBACK_LAST
        || QL_GetPlaybackState() == QL_PLAYBACK_STATE_RUNNING;
}

/* FIELD: callback2 == CB2_Overworld outside Quest Log playback */
int pn_obs_field_gate(void)
{
    return CB2_IS(CB2_Overworld) && gQuestLogState != QL_STATE_PLAYBACK && gQuestLogState != QL_STATE_PLAYBACK_LAST;
}

/* INGAME: the play-time counter runs (started only by CB2_NewGame / continue / end of
 * Quest Log) and no Quest Log playback */
int pn_obs_ingame(void)
{
    const u8 *st = SYMPTR(const u8 *, sPlayTimeCounterState);
    return st != NULL && *st != 0 && gQuestLogState != QL_STATE_PLAYBACK && gQuestLogState != QL_STATE_PLAYBACK_LAST;
}

/* BATTLE (section 1): battle structures allocated and the intro past its first stage */
int pn_obs_battle_gate(void)
{
    uint32_t f = (uint32_t)(uintptr_t)gBattleMainFunc;
    return gMain.inBattle && SYM(BattleMainCB1) != 0 && CB1 == SYM(BattleMainCB1) && gBattleStruct != NULL
        && gBattleResources != NULL && gBattleSpritesDataPtr != NULL
        && f != SYM(BeginBattleIntroDummy) && f != SYM(BeginBattleIntro);
}

int pn_obs_battle_mons_gate(void)
{
    uint32_t f = (uint32_t)(uintptr_t)gBattleMainFunc;
    return pn_obs_battle_gate() && f != SYM(BattleIntroGetMonsData) && f != SYM(BattleIntroPrepareBackgroundSlide)
        && f != SYM(BattleIntroDrawTrainersOrMonsSprites);
}

/* ---- screen classification (section 3.2) ------------------------------------------ */

static int warp_task_active(void)
{
    return TASK_ON(Task_Teleport2Warp) || TASK_ON(Task_TeleportWarp) || TASK_ON(Task_DoorWarp) || TASK_ON(Task_StairWarp)
        || TASK_ON(Task_ExitDoor) || TASK_ON(Task_ExitNonAnimDoor) || TASK_ON(Task_ExitNonDoor)
        || TASK_ON(Task_TeleportWarpIn) || TASK_ON(Task_ExitStairs);
}

static uint32_t start_menu_callback(void)
{
    const uint32_t *cb = SYMPTR(const uint32_t *, sStartMenuCallback);
    return cb != NULL ? *cb : 0;
}

static int someones_pc_menu(void)
{
    int t = TASK(Task_MultichoiceMenu_HandleInput);
    int win;
    if (t < 0 || TD(t, 7) != 255 /* MULTICHOICE_NONE */)
        return 0;
    win = TD(t, 6);
    /* script_menu.c CreateWindowFromRect(left, ...) places the window at tilemapLeft = left + 1: the PC menu
     * is drawn at left 0 (tilemapLeft 1), the other MULTICHOICE_NONE menu (the Seagallop ferry) at 17 (v4
     * fix: v2-v3 tested 0 and never matched, so the PC menu was reported as a plain MULTICHOICE) */
    return win >= 0 && win < WINDOWS_MAX && gWindows[win].window.tilemapLeft == 1;
}

static int player_pc_menu(void)
{
    return TASK_ON(Task_TopMenuHandleInput_pc) || TASK_ON(Task_TopMenu_ItemStorageSubmenu_HandleInput)
        || TASK_ON(Task_MailSubmenuHandleInput) || TASK_ON(Task_MailboxPcHandleInput)
        || TASK_ON(Task_MoveToBagYesNoMenuHandleInput);
}

static int region_map_type(void);

static int screen_kind(int sf, int help)
{
    int k;
    if (sf)
        return PN_SCREEN_SAVE_FAILED;
    if (help)
        return PN_SCREEN_HELP_SYSTEM;
    if (CB2_IS(CB2_Overworld)) {
        int t;
        if (gQuestLogState == QL_STATE_PLAYBACK || gQuestLogState == QL_STATE_PLAYBACK_LAST)
            return PN_SCREEN_QUEST_LOG;
        if (TASK_ON(Task_BattleStart))
            return PN_SCREEN_BATTLE_TRANSITION;
        if (warp_task_active())
            return PN_SCREEN_MAP_TRANSITION;
        t = TASK(Task_StartMenuHandleInput);
        if (t >= 0) {
            uint32_t cb = start_menu_callback();
            if (cb != 0 && (cb == SYM(StartCB_Save1) || cb == SYM(StartCB_Save2)))
                return PN_SCREEN_SAVE_MENU;
            return PN_SCREEN_START_MENU;
        }
        if (TASK_ON(task50_startmenu) || TASK_ON(Task_WaitFadeAndCreateStartMenuTask))
            return PN_SCREEN_START_MENU;
        if (TASK_ON(task50_save_game))
            return PN_SCREEN_SAVE_MENU;
        if (TASK_ON(Task_ShopMenu) || TASK_ON(Task_ReturnToShopMenu))
            return PN_SCREEN_MART;
        if (TASK_ON(Task_PCMainMenu) || player_pc_menu() || someones_pc_menu())
            return PN_SCREEN_PC_MENU;
        return PN_SCREEN_OVERWORLD;
    }
    if (CB2_IS(BattleMainCB2)) {
        if (TASK_ON(Task_DexScreen_RegisterMonToPokedex))
            return PN_SCREEN_POKEDEX;
        return PN_SCREEN_BATTLE;
    }
    k = pn_obs_screen_table_lookup(CB2);
    if (k == PN_SCREEN_REGION_MAP && region_map_type() == 2 /* REGIONMAP_TYPE_FLY */)
        k = PN_SCREEN_FLY_MAP;
    return k >= 0 ? k : PN_SCREEN_UNKNOWN;
}

static int field_state(void)
{
    if (CB2_IS(CB2_Overworld)) {
        if (gQuestLogState == QL_STATE_PLAYBACK || gQuestLogState == QL_STATE_PLAYBACK_LAST)
            return PN_FIELD_QUEST_LOG;
        if (TASK_ON(Task_BattleStart))
            return PN_FIELD_BATTLE_PENDING;
        if (warp_task_active())
            return PN_FIELD_WARP;
        return PN_FIELD_FREE;
    }
    if (CB2_IS(CB2_OverworldBasic))
        return PN_FIELD_BATTLE_PENDING;
    if (CB2_IS(CB2_LoadMap) || CB2_IS(CB2_DoChangeMap) || CB2_IS(CB2_ChangeMapMain) || CB2_IS(CB2_LoadMap2)
        || CB2_IS(CB2_ReturnToField) || CB2_IS(CB2_ReturnToFieldLocal))
        return PN_FIELD_WARP;
    return PN_FIELD_NONE;
}

/* ---- raw flag / var reads (section 0.2: FlagGet/VarGet have a Quest Log side effect) */

int pn_obs_raw_flag(u16 id)
{
    if (id == 0)
        return 0;
    if (id < FLAGS_COUNT)
        return gSaveBlock1Ptr != NULL && ((gSaveBlock1Ptr->flags[id >> 3] >> (id & 7)) & 1);
    if (id < 0x4000 || id > 0x407F)
        return 0;
    {
        const u8 *sf = SYMPTR(const u8 *, sSpecialFlags);
        return sf != NULL && ((sf[(id - 0x4000) >> 3] >> (id & 7)) & 1);
    }
}

static u16 raw_var(u16 id)
{
    if (id < VARS_START)
        return id;
    if (id <= VARS_END)
        return gSaveBlock1Ptr != NULL ? gSaveBlock1Ptr->vars[id - VARS_START] : 0;
    if (id < SPECIAL_VARS_START || id > SPECIAL_VARS_END)
        return 0;
    return *gSpecialVars[id - SPECIAL_VARS_START];
}

/* ---- field: free movement (section 2.3.1) ------------------------------------------ */

static int anim_stationary(u8 id)
{
    return id <= MOVEMENT_ACTION_FACE_RIGHT_FAST
        || (id >= MOVEMENT_ACTION_DELAY_1 && id <= MOVEMENT_ACTION_DELAY_16)
        || (id >= MOVEMENT_ACTION_WALK_IN_PLACE_SLOW_DOWN && id <= MOVEMENT_ACTION_WALK_IN_PLACE_FASTER_RIGHT)
        || (id >= MOVEMENT_ACTION_ACRO_WHEELIE_FACE_DOWN && id <= MOVEMENT_ACTION_ACRO_END_WHEELIE_FACE_RIGHT)
        || (id >= MOVEMENT_ACTION_ACRO_WHEELIE_IN_PLACE_DOWN && id <= MOVEMENT_ACTION_ACRO_WHEELIE_IN_PLACE_RIGHT);
}

/* MapHeaderCheckScriptTable(MAP_SCRIPT_ON_FRAME_TABLE) with raw var reads: 1 if an
 * on-frame script would start (TryRunOnFrameMapScript runs before any input) */
static int on_frame_script_pending(void)
{
    const u8 *ms = gMapHeader.mapScripts;
    const u8 *tab = NULL;
    int guard;
    if (gQuestLogState == QL_STATE_PLAYBACK_LAST)
        return 0;
    if (ms == NULL)
        return 0;
    for (guard = 0; guard < 64; guard++) {
        if (*ms == 0)
            return 0;
        if (*ms == MAP_SCRIPT_ON_FRAME_TABLE) {
            tab = (const u8 *)(uintptr_t)T2_READ_32(ms + 1);
            break;
        }
        ms += 5;
    }
    if (tab == NULL)
        return guard >= 64; /* malformed: assume pending */
    for (guard = 0; guard < 256; guard++) {
        u16 v1 = T1_READ_16(tab), v2;
        if (v1 == 0)
            return 0;
        v2 = T1_READ_16(tab + 2);
        if (raw_var(v1) == raw_var(v2))
            return 1;
        tab += 8;
    }
    return 1;
}

static const struct ObjectEventTemplate *object_template(const struct ObjectEvent *o)
{
    const struct ObjectEventTemplate *templates;
    int count, i;
    if (gSaveBlock1Ptr->location.mapNum == o->mapNum && gSaveBlock1Ptr->location.mapGroup == o->mapGroup) {
        templates = gSaveBlock1Ptr->objectEventTemplates;
        count = gMapHeader.events != NULL ? gMapHeader.events->objectEventCount : 0;
    } else {
        const struct MapHeader *h = Overworld_GetMapHeaderByGroupAndId(o->mapGroup, o->mapNum);
        if (h == NULL || h->events == NULL)
            return NULL;
        templates = h->events->objectEvents;
        count = h->events->objectEventCount;
    }
    if (templates == NULL)
        return NULL;
    for (i = 0; i < count && i < OBJECT_EVENT_TEMPLATES_COUNT; i++)
        if (templates[i].localId == o->localId)
            return &templates[i];
    return NULL;
}

/* ---- trainer sight: an exact, pure mirror of CheckForTrainersWantingBattle ---------------
 * (trainer_see.c:88; v3.1, docs/OBS_SOURCES.md section 2.3.1 term 8). ProcessPlayerFieldInput
 * calls it before any input (field_control_avatar.c:209); when it returns TRUE the frame is
 * consumed (ConfigureAndSetUpOneTrainerBattle sets up a script and locks the controls). The
 * game functions are re-implemented on pure reads (the originals are static, and
 * CheckPathBetweenTrainerAndPlayer temporarily writes the trainer's rangeX/rangeY); the global
 * grid accessors they call (MapGridGetCollisionAt, GetMapBorderIdAt, MapGridGetElevationAt,
 * MapGridGetMetatileBehaviorAt, CanCameraMoveInDirection, GetFirstInactiveObjectEventId,
 * MetatileBehavior_Is*Blocked) are pure and called as they are. Coordinates are border
 * coordinates (currentCoords, MAP_OFFSET included), as in the game. */

/* sDirectionToVectors (event_object_movement.c:903) for DIR_SOUTH..DIR_EAST */
static const s8 kSightDx[5] = {0, 0, 0, -1, 1};
static const s8 kSightDy[5] = {0, 1, -1, 0, 0};

/* IsElevationMismatchAt (event_object_movement.c:8351) */
static int sight_elevation_mismatch(u8 elevation, s16 x, s16 y)
{
    u8 e;
    if (elevation == 0)
        return 0;
    e = MapGridGetElevationAt(x, y);
    return e != 0 && e != 15 && e != elevation;
}

/* DoesObjectCollideWithObjectAt (event_object_movement.c:4904) with AreElevationsCompatible
 * (:8437) */
static int sight_object_at(const struct ObjectEvent *self, s16 x, s16 y)
{
    int i;
    for (i = 0; i < OBJECT_EVENTS_COUNT; i++) {
        const struct ObjectEvent *c = &gObjectEvents[i];
        if (!c->active || c == self)
            continue;
        if ((c->currentCoords.x == x && c->currentCoords.y == y) || (c->previousCoords.x == x && c->previousCoords.y == y)) {
            u8 a = self->currentElevation, b = c->currentElevation;
            if (a == 0 || b == 0 || a == b)
                return 1;
        }
    }
    return 0;
}

/* the IMPASSABLE part shared by GetCollisionAtCoords (event_object_movement.c:4835) and
 * GetCollisionFlagsAtCoords (:4851, flag 2): collision bit, outside the loaded map,
 * IsMetatileDirectionallyImpassable (:4894: gOppositeDirectionBlockedMetatileFuncs on the
 * object's own currentMetatileBehavior, gDirectionBlockedMetatileFuncs on the target,
 * :889-901), the camera object's edge */
static int sight_impassable(const struct ObjectEvent *o, s16 x, s16 y, u8 dir)
{
    u8 own = o->currentMetatileBehavior, b;
    if (MapGridGetCollisionAt(x, y) || GetMapBorderIdAt(x, y) == CONNECTION_INVALID)
        return 1;
    b = (u8)MapGridGetMetatileBehaviorAt(x, y);
    switch (dir) {
    case DIR_SOUTH:
        if (MetatileBehavior_IsSouthBlocked(own) || MetatileBehavior_IsNorthBlocked(b))
            return 1;
        break;
    case DIR_NORTH:
        if (MetatileBehavior_IsNorthBlocked(own) || MetatileBehavior_IsSouthBlocked(b))
            return 1;
        break;
    case DIR_WEST:
        if (MetatileBehavior_IsWestBlocked(own) || MetatileBehavior_IsEastBlocked(b))
            return 1;
        break;
    default: /* DIR_EAST */
        if (MetatileBehavior_IsEastBlocked(own) || MetatileBehavior_IsWestBlocked(b))
            return 1;
        break;
    }
    return o->trackedByCamera && !CanCameraMoveInDirection(dir);
}

/* GetTrainerApproachDistanceSouth/North/West/East (trainer_see.c:149-194): how far the player's
 * tile (px, py) lies in direction dir within range, 0 if not in that sight line */
static u8 sight_distance(const struct ObjectEvent *t, u8 dir, s16 range, s16 px, s16 py)
{
    s16 tx = t->currentCoords.x, ty = t->currentCoords.y;
    switch (dir) {
    case DIR_SOUTH:
        if (tx == px && py > ty && py <= ty + range) {
            /* :155: a long sight line needs a free object slot (for the camera object) */
            if (range > 3 && GetFirstInactiveObjectEventId() == OBJECT_EVENTS_COUNT)
                return 0;
            return (u8)(py - ty);
        }
        return 0;
    case DIR_NORTH:
        return (tx == px && py < ty && py >= ty - range) ? (u8)(ty - py) : 0;
    case DIR_WEST:
        return (ty == py && px < tx && px >= tx - range) ? (u8)(tx - px) : 0;
    default: /* DIR_EAST */
        return (ty == py && px > tx && px <= tx + range) ? (u8)(px - tx) : 0;
    }
}

/* CheckPathBetweenTrainerAndPlayer (trainer_see.c:198-233): distance if every tile from the
 * trainer's own up to the one before the player has none of the collision flags 2/4/8
 * (GetCollisionFlagsAtCoords & ~1: the movement-range flag 1 is ignored, :215), and the
 * player's tile, with the trainer's movement range cleared (:219-228, so never OUTSIDE_RANGE),
 * gives exactly COLLISION_OBJECT_EVENT: not impassable, no elevation mismatch, an object there
 * (the player); 0 otherwise */
static u8 sight_path(const struct ObjectEvent *t, u8 dist, u8 dir)
{
    s16 x = t->currentCoords.x, y = t->currentCoords.y;
    int i;
    if (dist == 0)
        return 0;
    for (i = 0; i <= dist - 1; i++, x += kSightDx[dir], y += kSightDy[dir]) {
        if (sight_impassable(t, x, y, dir) || sight_elevation_mismatch(t->currentElevation, x, y) || sight_object_at(t, x, y))
            return 0;
    }
    if (sight_impassable(t, x, y, dir) || sight_elevation_mismatch(t->currentElevation, x, y))
        return 0;
    return sight_object_at(t, x, y) ? dist : 0;
}

/* GetTrainerApproachDistance (trainer_see.c:123-147): NORMAL tests only its facing direction,
 * BURIED (and SEE_ALL_DIRECTIONS, never reached: the caller skips it) all four in the order
 * south, north, west, east. -1: a NORMAL trainer facing no cardinal direction (the game would
 * index sDirectionalApproachDistanceFuncs out of bounds; treated as "spots") */
static int sight_approach(const struct ObjectEvent *t, s16 px, s16 py)
{
    s16 range = t->trainerRange_berryTreeId;
    u8 d;
    if (t->trainerType == TRAINER_TYPE_NORMAL) {
        d = t->facingDirection;
        if (d < DIR_SOUTH || d > DIR_EAST)
            return -1;
        return sight_path(t, sight_distance(t, d, range, px, py), d);
    }
    for (d = DIR_SOUTH; d <= DIR_EAST; d++) {
        u8 r = sight_path(t, sight_distance(t, d, range, px, py), d);
        if (r)
            return r;
    }
    return 0;
}

/* GetMonsStateToDoubles (pokemon.c:3769) != PLAYER_HAS_TWO_USABLE_MONS, without its
 * CalculatePlayerPartyCount write (pokemon.c:3742): the party count recomputed from the
 * species, GetMonData on copies (it decrypts in place) */
static int sight_double_exempt(void)
{
    int n = 0, alive = 0, i;
    struct Pokemon m;
    while (n < PARTY_SIZE) {
        memcpy(&m, &gPlayerParty[n], sizeof(m));
        if (GetMonData(&m, MON_DATA_SPECIES, NULL) == SPECIES_NONE)
            break;
        n++;
    }
    if (n == 1)
        return 1; /* PLAYER_HAS_ONE_MON */
    for (i = 0; i < n; i++) {
        u32 sp;
        memcpy(&m, &gPlayerParty[i], sizeof(m));
        if (GetMonData(&m, MON_DATA_HP, NULL) == 0)
            continue;
        sp = GetMonData(&m, MON_DATA_SPECIES_OR_EGG, NULL);
        if (sp != SPECIES_NONE && sp != SPECIES_EGG)
            alive++;
    }
    return alive <= 1; /* PLAYER_HAS_ONE_USABLE_MON */
}

/* 1 if CheckForTrainersWantingBattle would return TRUE in the next frame (a trainer starts its
 * approach, before any input is read): an active NORMAL or BURIED trainer (trainer_see.c:94-97)
 * whose trainer flag is clear (CheckTrainer :105-109, GetTrainerFlagFromScriptPointer
 * battle_setup.c:841, raw flag read) and whose approach distance is nonzero (:111-112), unless it is
 * a double battle the player cannot fight (:114). The sight test runs first here (it is the
 * cheap part; the game reads the script first, which only matters for an object without a
 * template or script, where the game would read through NULL: treated as "spots"). */
static int trainer_may_spot(void)
{
    s16 px, py;
    int i;
    if (QL_IsTrainerSightDisabled())
        return 0;
    PlayerGetDestCoords(&px, &py);
    for (i = 0; i < OBJECT_EVENTS_COUNT; i++) {
        const struct ObjectEvent *o = &gObjectEvents[i];
        const struct ObjectEventTemplate *tmpl;
        const u8 *script;
        int dist;
        if (!o->active || (o->trainerType != TRAINER_TYPE_NORMAL && o->trainerType != TRAINER_TYPE_BURIED))
            continue;
        dist = sight_approach(o, px, py);
        if (dist == 0)
            continue;
        tmpl = object_template(o);
        if (tmpl == NULL || tmpl->script == NULL)
            return 1;
        script = tmpl->script;
        if (pn_obs_raw_flag((u16)(TRAINER_FLAGS_START + T1_READ_16(script + 2))))
            continue; /* already fought */
        if (dist > 0 && script[1] == TRAINER_BATTLE_DOUBLE && sight_double_exempt())
            continue;
        return 1;
    }
    return 0;
}

struct PnForcedMovementEntry {
    bool8 (*check)(u8 metatileBehavior);
    bool8 (*apply)(void);
};

/* the player's tile forces a movement (TryDoMetatileBehaviorForcedMovement) */
static int forced_tile(const struct ObjectEvent *p)
{
    const struct PnForcedMovementEntry *e = SYMPTR(const struct PnForcedMovementEntry *, sForcedMovementFuncs);
    int i;
    if (gPlayerAvatar.flags & PLAYER_AVATAR_FLAG_CONTROLLABLE)
        return 0;
    if (e == NULL)
        return 1;
    for (i = 0; i < 64 && e[i].check != NULL; i++)
        if (e[i].check(p->currentMetatileBehavior))
            return 1;
    return 0;
}

/* returns 0 (no input), 1 (idle: D-pad, A, START, SELECT) or 2 (unfinished stationary
 * animation, T_NOT_MOVING: *keys gets A, START, SELECT and the D-pad keys that interrupt
 * a WALK_IN_PLACE_SLOW bump, TryInterruptObjectEventSpecialAnim) */
static int overworld_free(uint16_t *keys)
{
    const struct ObjectEvent *p;
    int anim = 0;
    if (SYM(CB1_Overworld) == 0 || CB1 != SYM(CB1_Overworld) || !CB2_IS(CB2_Overworld))
        return 0;
    if (ArePlayerFieldControlsLocked() || ScriptContext_IsEnabled())
        return 0;
    if (IsQuestLogInputDpad())
        return 0;
    if (gPlayerAvatar.preventStep || (gPlayerAvatar.flags & PLAYER_AVATAR_FLAG_FORCED))
        return 0;
    if (gPlayerAvatar.objectEventId >= OBJECT_EVENTS_COUNT)
        return 0;
    p = &gObjectEvents[gPlayerAvatar.objectEventId];
    if (!p->active || !p->isPlayer)
        return 0;
    if (p->singleMovementActive)
        return 0;
    if (p->heldMovementActive) {
        if (!anim_stationary(p->movementActionId))
            return 0; /* T_TILE_TRANSITION, or T_TILE_CENTER after a step */
        if (p->heldMovementFinished && gPlayerAvatar.runningState == TURN_DIRECTION)
            return 0; /* T_TILE_CENTER */
        if (!p->heldMovementFinished)
            anim = 1; /* T_NOT_MOVING, animation still running */
    }
    if (on_frame_script_pending())
        return 0;
    if (trainer_may_spot())
        return 0;
    if (forced_tile(p))
        return 0;
    if (!anim) {
        *keys = PN_KEY_DPAD | PN_KEY_A | PN_KEY_START | PN_KEY_SELECT;
        return 1;
    }
    *keys = PN_KEY_A | PN_KEY_START | PN_KEY_SELECT;
    if (p->movementActionId >= MOVEMENT_ACTION_WALK_IN_PLACE_SLOW_DOWN
        && p->movementActionId <= MOVEMENT_ACTION_WALK_IN_PLACE_SLOW_RIGHT) {
        static const uint16_t dir_key[5] = {0, PN_KEY_DOWN, PN_KEY_UP, PN_KEY_LEFT, PN_KEY_RIGHT};
        uint16_t same = p->movementDirection <= 4 ? dir_key[p->movementDirection] : 0;
        *keys |= PN_KEY_DPAD & ~same;
    }
    return 2;
}

/* ---- field: dialog, script menus, start menu, save, shop, PCs (sections 2.4-2.6, 2.10) */

static u8 script_status(void)
{
    const u8 *s = SYMPTR(const u8 *, sGlobalScriptContextStatus);
    return s != NULL ? *s : 0;
}

static const struct ScriptContext *script_ctx(void)
{
    return SYMPTR(const struct ScriptContext *, sGlobalScriptContext);
}

/* no script bytecode runs before RunTasks in the next frame */
static int no_preempt(void)
{
    const struct ScriptContext *c = script_ctx();
    return script_status() != 0 || (c != NULL && c->mode == PN_SCRIPT_MODE_NATIVE);
}

static int msg_cont(PnWait *w, int ctx)
{
    const u8 *win = SYMPTR(const u8 *, sMessageWindowId);
    const uint32_t *next = SYMPTR(const uint32_t *, sMessageNextTask);
    if (!TASK_ON(Task_ContinueTaskAfterMessagePrints) || win == NULL)
        return 0;
    if (pr_wait(*win)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, ctx, K_AB);
        return 1;
    }
    /* the follow-up function is CALLED every frame once the printer is idle; these
     * follow-ups only wait for JOY_NEW(A_BUTTON) (item_menu.c, tm_case.c, berry_pouch.c) */
    if (!printer_active(*win) && next != NULL && *next != 0
        && (*next == SYM(Task_WaitAButtonAndCloseContextMenu) || *next == SYM(Task_WaitButtonAfterErrorPrint_tm)
            || *next == SYM(Task_WaitButtonBeforeDialogueWindowDestruction_brp))) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, ctx, PN_KEY_A);
        return 1;
    }
    return 0;
}

/* the window of the whiteout message while Task_RushInjuredPokemonToCenter runs its printer
 * (states 1 and 4, print state 1: field_screen_effect.c:362-385, :412-425), else -1; also used
 * for in_dialog/dialog_text (observe.c fill_dialog, through PnWait.whiteout_win) */
static int whiteout_window(void)
{
    int t, win;
    /* FieldCB_RushInjuredPokemonToCenter locks the field controls before it creates the task
     * (:458) and only the heal script started after it unlocks them: a cheap pre-test that
     * skips the task scan during free play */
    if (!CB2_IS(CB2_Overworld) || !ArePlayerFieldControlsLocked())
        return -1;
    t = TASK(Task_RushInjuredPokemonToCenter);
    if (t < 0 || (TD(t, 0) != 1 && TD(t, 0) != 4) || TD(t, 2) != 1)
        return -1;
    win = TD(t, 1);
    return (win >= 0 && win < NUM_TEXT_PRINTERS) ? win : -1;
}

static void field_wait(PnWait *w)
{
    int t;
    uint8_t cur = 0, cnt = 0;

    if (!CB2_IS(CB2_Overworld))
        return;

    /* (1) save prompt (start menu SAVE or script Field_AskSaveTheGame) */
    t = TASK(Task_StartMenuHandleInput);
    if ((t >= 0 && TD(t, 0) == 1 && start_menu_callback() == SYM(StartCB_Save2) && SYM(StartCB_Save2) != 0)
        || TASK_ON(task50_save_game)) {
        const uint32_t *cb = SYMPTR(const uint32_t *, sSaveDialogCB);
        w->menu_open = PN_MENU_SAVE_PROMPT;
        if (!printer_active(0) && cb != NULL && *cb != 0
            && (*cb == SYM(SaveDialogCB_AskSaveHandleInput) || *cb == SYM(SaveDialogCB_AskOverwriteOrReplacePreviousFileHandleInput)))
            yesno_menu(w, PN_WAIT_SAVE_YESNO, PN_MENU_SAVE_PROMPT);
        else if (pr_wait(0))
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_SAVE_TEXT, K_AB);
        return;
    }

    /* (2) start menu */
    if (t >= 0) {
        w->menu_open = PN_MENU_START;
        if (TD(t, 0) == 1 && start_menu_callback() == SYM(StartCB_HandleInput) && SYM(StartCB_HandleInput) != 0) {
            const u8 *pos = SYMPTR(const u8 *, sStartMenuCursorPos);
            const u8 *num = SYMPTR(const u8 *, sNumStartMenuItems);
            set_wait(w, PN_INPUT_MENU, PN_WAIT_START_MENU, PN_KEY_UP | PN_KEY_DOWN | PN_KEY_A | PN_KEY_B | PN_KEY_START);
            set_menu(w, PN_MENU_START, pos ? *pos : 0, num ? *num : 0, 255);
        }
        return;
    }

    /* (3) mart BUY / SELL / QUIT */
    if (TASK_ON(Task_ShopMenu)) {
        uint16_t keys = menu_keys(0, &cur, &cnt) | PN_KEY_B;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_SHOP_MENU, keys);
        set_menu(w, PN_MENU_MART, cur, cnt, 255);
        return;
    }

    /* (4) PCs */
    t = TASK(Task_PCMainMenu);
    if (t >= 0) {
        w->menu_open = PN_MENU_PC;
        if (TD(t, 0) == 2) { /* STATE_HANDLE_INPUT: Menu_ProcessInput */
            static const uint8_t acts[5] = {PN_ACTION_WITHDRAW, PN_ACTION_DEPOSIT, PN_ACTION_MOVE_MONS,
                                            PN_ACTION_MOVE_ITEMS, PN_ACTION_CANCEL};
            uint16_t keys = menu_keys(1, &cur, &cnt) | PN_KEY_B;
            set_wait(w, PN_INPUT_MENU, PN_WAIT_PC_STORAGE_TOP, keys);
            set_menu(w, PN_MENU_PC, cur, cnt, 255);
            set_actions(w, acts, 5);
        } else if (TD(t, 0) == 3) { /* STATE_ERROR_MSG */
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_PC_STORAGE_TOP, K_AB | PN_KEY_UP | PN_KEY_DOWN);
        }
        return;
    }
    if (TASK_ON(Task_TopMenuHandleInput_pc) || TASK_ON(Task_TopMenu_ItemStorageSubmenu_HandleInput)
        || TASK_ON(Task_MailSubmenuHandleInput)) {
        /* player_pc.c: sMenuActions_TopMenu (ITEM STORAGE, MAILBOX, TURN OFF; sItemOrder is the identity
         * for both the bedroom and the other player PCs), sMenuActions_ItemPc (WITHDRAW ITEM, DEPOSIT
         * ITEM, CANCEL) */
        static const uint8_t top[3] = {PN_ACTION_ITEM_STORAGE, PN_ACTION_MAILBOX, PN_ACTION_CANCEL};
        static const uint8_t sub[3] = {PN_ACTION_WITHDRAW, PN_ACTION_DEPOSIT, PN_ACTION_CANCEL};
        uint16_t keys = menu_keys(1, &cur, &cnt) | PN_KEY_B;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_PC_PLAYER, keys);
        set_menu(w, PN_MENU_PC, cur, cnt, 255);
        if (TASK_ON(Task_TopMenuHandleInput_pc))
            set_actions(w, top, 3);
        else if (TASK_ON(Task_TopMenu_ItemStorageSubmenu_HandleInput))
            set_actions(w, sub, 3);
        return;
    }
    if (TASK_ON(Task_MoveToBagYesNoMenuHandleInput)) {
        yesno_menu(w, PN_WAIT_PC_PLAYER, PN_MENU_PC);
        return;
    }
    if (player_pc_menu()) {
        w->menu_open = PN_MENU_PC;
        return;
    }

    /* (5) script menus (the script is parked by ScriptContext_Stop) */
    if (no_preempt()) {
        t = TASK(Task_YesNoMenu_HandleInput);
        if (t >= 0) {
            w->menu_open = PN_MENU_SCRIPT_YESNO;
            if (TD(t, 2) >= 5)
                yesno_menu(w, PN_WAIT_SCRIPT_YESNO, PN_MENU_SCRIPT_YESNO);
            return;
        }
        t = TASK(Task_MultichoiceMenu_HandleInput);
        if (t >= 0) {
            const u8 *delay = SYMPTR(const u8 *, sDelay_scriptmenu);
            int open = someones_pc_menu() ? PN_MENU_PC : PN_MENU_MULTICHOICE;
            w->menu_open = (uint8_t)open;
            if (!FADE && delay != NULL && *delay == 0) {
                uint16_t keys = menu_keys(TD(t, 5), &cur, &cnt);
                if (!TD(t, 4))
                    keys |= PN_KEY_B;
                set_wait(w, PN_INPUT_MENU, PN_WAIT_MULTICHOICE, keys);
                set_menu(w, open, cur, cnt, (uint8_t)TD(t, 7));
                if (open == PN_MENU_PC)
                    someones_pc_actions(w);
            }
            return;
        }
        t = TASK(Hask_MultichoiceGridMenu_HandleInput);
        if (t >= 0) {
            const struct PnMenuMirror *m = smenu();
            uint16_t keys = PN_KEY_A | PN_KEY_DPAD;
            if (!TD(t, 4))
                keys |= PN_KEY_B;
            set_wait(w, PN_INPUT_MENU, PN_WAIT_MULTICHOICE_GRID, keys);
            set_menu(w, PN_MENU_MULTICHOICE_GRID, m ? (uint8_t)m->cursorPos : 0,
                     m ? (uint8_t)(m->maxCursorPos - m->minCursorPos + 1) : 0, 255);
            return;
        }
        t = TASK(Task_ListMenuHandleInput);
        if (t >= 0) {
            u16 scroll = 0, row = 0;
            int lt = TD(t, 14);
            uint8_t sel = 0;
            if (lt >= 0 && lt < NUM_TASKS && gTasks[lt].isActive) {
                ListMenuGetScrollAndRow((u8)lt, &scroll, &row);
                sel = (uint8_t)(scroll + row);
            }
            set_wait(w, PN_INPUT_MENU, PN_WAIT_SCRIPT_LIST, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
            set_menu(w, PN_MENU_SCRIPT_LIST, sel, (uint8_t)TD(t, 1), 255);
            return;
        }
    }

    /* (6) field message box / waitbuttonpress */
    if (no_preempt()) {
        const struct ScriptContext *c = script_ctx();
        t = TASK(Task_DrawFieldMessageBox);
        if (t >= 0 && TD(t, 0) == 2 && pr_wait(0)) {
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_DLG_TEXT, K_AB);
            return;
        }
        if (c != NULL && script_status() == 0 && c->mode == PN_SCRIPT_MODE_NATIVE && SYM(WaitForAorBPress) != 0
            && (uint32_t)(uintptr_t)c->nativePtr == SYM(WaitForAorBPress)) {
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_DLG_BUTTON, K_AB);
            return;
        }
    }
    if (msg_cont(w, PN_WAIT_FIELD_MSG_CONT))
        return;

    /* (6b) v3.1: the message after a whiteout ("<player> scurried to a POKEMON CENTER" /
     * "back home", field_screen_effect.c:387 Task_RushInjuredPokemonToCenter, created by
     * FieldCB_RushInjuredPokemonToCenter on the map load of CB2_WhiteOut, overworld.c:1558,
     * which also stops the script context). States 1 and 4 call PrintWhiteOutRecoveryMessage
     * (:362), which in its print state 1 runs RunTextPrinters; the text ends in a prompt
     * (strings.c:961-962), so the printer of the task's window (tWindowId, data[1]) blocks on
     * JOY_NEW(A_BUTTON | B_BUTTON) (TextPrinterWaitWithDownArrow text.c:550, no auto-scroll);
     * once it is done the task goes on to the fade-in and EventScript_AfterWhiteOutHeal. The
     * field controls stay locked (LockPlayerFieldControls :458). */
    if (no_preempt()) {
        int win = w->whiteout_win;
        if (win >= 0 && pr_wait(win)) {
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_WHITEOUT_MSG, K_AB);
            return;
        }
    }

    /* (7) free movement */
    {
        uint16_t keys = 0;
        int ow = overworld_free(&keys);
        if (ow)
            set_wait(w, PN_INPUT_OVERWORLD_FREE, ow == 1 ? PN_WAIT_OW_FREE : PN_WAIT_OW_ANIM, keys);
    }
}

/* ---- title, main menu, Oak speech, naming (section 2.13, 2.10.2) ------------------- */

static void title_wait(PnWait *w)
{
    int t = TASK(Task_TitleScreenMain);
    if (CB2_IS(CB2_TitleScreenRun) && t >= 0 && TD(t, 0) == 3 /* TITLESCREENSCENE_RUN */ && TD(t, 1) >= 1) {
        set_wait(w, PN_INPUT_TITLE_START, PN_WAIT_TITLE, PN_KEY_A | PN_KEY_START);
        w->flags |= PN_INPUT_FLAG_DANGEROUS_CHORD;
    }
}

static void main_menu_wait(PnWait *w)
{
    int t = TASK(Task_HandleMenuInput_main);
    if (!CB2_IS(CB2_MainMenu))
        return;
    if (t >= 0 && !FADE) {
        /* HandleMenuInput (main_menu.c): A selects, B back to the title, UP/DOWN */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_MAIN_MENU, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
        set_menu(w, PN_MENU_FULLSCREEN, (uint8_t)TD(t, 1), 0, (uint8_t)TD(t, 0));
        return;
    }
    /* save error / notice (MAIN_MENU_WINDOW_ERROR = 4): the task runs the printer when no
     * fade is active; a prompt inside the text takes A/B, the finished text A only */
    if (TASK_ON(Task_SaveErrorStatus_RunPrinterThenWaitButton) && !FADE) {
        if (pr_wait(4))
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_MAIN_MENU_MSG, K_AB);
        else if (!printer_active(4))
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_MAIN_MENU_MSG, PN_KEY_A);
        return;
    }
    /* Mystery Gift error: tMGErrorMsgState (data[9]) 3 waits for JOY_NEW(A | B) */
    t = TASK(Task_MysteryGiftError);
    if (t >= 0 && TD(t, 9) == 3)
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_MAIN_MENU_MSG, K_AB);
}

/* struct OakSpeechResources (oak_speech.c, file-local): currentPage @18 (u16) */
static u16 oak_current_page(void)
{
    const u8 *const *pp = SYMPTR(const u8 *const *, sOakSpeechResources);
    if (pp == NULL || *pp == NULL)
        return 0xFFFF;
    return *(const u16 *)(*pp + 18);
}

static void oak_wait(PnWait *w)
{
    int t;
    uint8_t cur = 0, cnt = 0;
    if (!CB2_IS(CB2_NewGameScene))
        return;
    /* page text: CB2_NewGameScene runs RunTextPrinters every frame */
    if (pr_wait(0)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_OAK_TEXT, K_AB);
        return;
    }
    if (TASK_ON(Task_ControlsGuide_HandleInput)) {
        u16 page = oak_current_page();
        if (!FADE && page != 0xFFFF)
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_OAK_CONTROLS, page == 0 ? PN_KEY_A : K_AB);
        return;
    }
    if (TASK_ON(Task_PikachuIntro_HandleInput)) {
        u16 page = oak_current_page();
        if (gMain.state == 1 /* PIKACHU_INTRO_HANDLE_INPUT */ && page != 0xFFFF)
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_OAK_PIKACHU, page == 0 ? PN_KEY_A : K_AB);
        return;
    }
    if (TASK_ON(Task_OakSpeech_HandleGenderInput)) {
        uint16_t keys = menu_keys(0, &cur, &cnt);
        set_wait(w, PN_INPUT_MENU, PN_WAIT_OAK_GENDER, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        return;
    }
    if (TASK_ON(Task_OakSpeech_HandleRivalNameInput)) {
        uint16_t keys = menu_keys(1, &cur, &cnt);
        set_wait(w, PN_INPUT_MENU, PN_WAIT_OAK_NAME_CHOICE, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        return;
    }
    t = TASK(Task_OakSpeech_HandleConfirmNameInput);
    if (t >= 0)
        yesno_menu(w, PN_WAIT_OAK_CONFIRM, PN_MENU_FULLSCREEN);
}

static void naming_wait(PnWait *w)
{
    const u8 *const *pp = SYMPTR(const u8 *const *, sNamingScreen);
    const u8 *n;
    int h;
    if (!CB2_IS(CB2_NamingScreen) || pp == NULL || *pp == NULL)
        return;
    n = *pp;
    if (n[0x1E10] == 2 /* STATE_HANDLE_INPUT */) {
        h = TASK(Task_HandleInput_naming);
        if (h >= 0 && TD(h, 0) == 1 /* INPUT_STATE_ENABLED */)
            set_wait(w, PN_INPUT_NAMING, PN_WAIT_NAMING, PN_KEY_DPAD | PN_KEY_A | PN_KEY_B | PN_KEY_SELECT | PN_KEY_START);
    } else if (n[0x1E10] == 7 /* STATE_WAIT_SENT_TO_PC_MESSAGE */) {
        if (pr_wait(0))
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_NAMING_PCMSG, K_AB);
        else if (!printer_active(0))
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_NAMING_PCMSG, PN_KEY_A);
    }
}

/* ---- bag (section 2.7) -------------------------------------------------------------- */

/* struct BagMenuAlloc (item_menu.c, file-local) */
struct PnBagMenuAllocMirror {
    MainCallback exitCB;
    u8 itemOriginalLocation;
    u8 pocketSwitchMode:4;
    u8 itemMenuIcon:2;
    u8 inhibitItemDescriptionPrint:2;
    u16 contextMenuSelectedItem;
    u8 pocketScrollArrowsTask;
    u8 pocketSwitchArrowsTask;
    u8 nItems[3];
    u8 maxShowed[3];
    u8 data[4];
};

static int list_task_ok(int lt)
{
    return lt >= 0 && lt < NUM_TASKS && gTasks[lt].isActive;
}

static void bag_wait(PnWait *w)
{
    int t;
    uint8_t cur = 0, cnt = 0;
    if (!CB2_IS(CB2_BagMenuRun))
        return;
    t = TASK(Task_BagMenu_HandleInput);
    if (t >= 0) {
        const struct PnBagMenuAllocMirror *const *pp = SYMPTR(const struct PnBagMenuAllocMirror *const *, sBagMenuDisplay);
        const struct PnBagMenuAllocMirror *d = pp ? *pp : NULL;
        u16 scroll = 0, row = 0;
        uint16_t keys;
        int pocket = gBagMenuState.pocket;
        if (FADE || TASK_ON(Task_AnimateWin0v) || d == NULL || pocket < 0 || pocket > 2)
            return;
        if (!list_task_ok(TD(t, 0)))
            return;
        ListMenuGetScrollAndRow((u8)TD(t, 0), &scroll, &row);
        keys = PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN;
        if (d->pocketSwitchMode == 0) {
            if (pocket != 0)
                keys |= PN_KEY_LEFT;
            if (pocket < 2)
                keys |= PN_KEY_RIGHT;
        }
        if (gBagMenuState.location == ITEMMENULOCATION_FIELD && scroll + row != d->nItems[pocket])
            keys |= PN_KEY_SELECT;
        set_wait(w, gMain.inBattle ? PN_INPUT_BATTLE_BAG : PN_INPUT_MENU, PN_WAIT_BAG_LIST, keys);
        set_menu(w, PN_MENU_FULLSCREEN, (uint8_t)(scroll + row), (uint8_t)(d->nItems[pocket] + 1), (uint8_t)pocket);
        /* v4: the item under the cursor (the list shows the pocket in stored order; the last row is CANCEL) */
        if (scroll + row < d->nItems[pocket])
            w->cursor_item = BagGetItemIdByPocketPosition((u8)(pocket + 1), (u16)(scroll + row));
        return;
    }
    if (TASK_ON(Task_MoveItemInPocket_HandleInput)) {
        set_wait(w, PN_INPUT_MENU, PN_WAIT_BAG_MOVE, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN | PN_KEY_SELECT);
        return;
    }
    if (TASK_ON(Task_FieldItemContextMenuHandleInput)) {
        /* v4: the entries (item_menu.c sContextMenuItemsPtr[sContextMenuNumItems], ITEMMENUACTION_*) */
        const u8 *const *ip = SYMPTR(const u8 *const *, sContextMenuItemsPtr);
        const u8 *np = SYMPTR(const u8 *, sContextMenuNumItems);
        uint16_t keys = menu_keys(0, &cur, &cnt) | PN_KEY_B;
        set_wait(w, gMain.inBattle ? PN_INPUT_BATTLE_BAG : PN_INPUT_MENU, PN_WAIT_BAG_CONTEXT, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        if (ip != NULL && *ip != NULL && np != NULL) {
            uint8_t acts[PN_OBS_MENU_ACTIONS];
            int i, n = *np > PN_OBS_MENU_ACTIONS ? PN_OBS_MENU_ACTIONS : *np;
            for (i = 0; i < n; i++)
                acts[i] = bag_action((*ip)[i]);
            set_actions(w, acts, n);
        }
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    t = TASK(Task_SelectQuantityToToss);
    if (t < 0)
        t = TASK(Task_SelectQuantityToSell);
    if (t < 0)
        t = TASK(Task_SelectQuantityToDeposit);
    if (t >= 0) {
        set_wait(w, PN_INPUT_QUANTITY, PN_WAIT_BAG_QUANTITY, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        w->quantity = (uint16_t)TD(t, 8);     /* AdjustQuantityAccordingToDPadInput(&data[8], data[2]) */
        w->quantity_max = (uint16_t)TD(t, 2);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    if (TASK_ON(Task_CallYesOrNoCallback)) {
        yesno_menu(w, PN_WAIT_BAG_YESNO, PN_MENU_FULLSCREEN);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    if (TASK_ON(Task_WaitAB_RedrawAndReturnToBag) || TASK_ON(Task_WaitPressAB_AfterSell)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_BAG_ACK, K_AB);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    msg_cont(w, PN_WAIT_BAG_MSG_CONT);
    if (w->waiting)
        w->context_item = gSpecialVar_ItemId;
}

/* ---- party menu and summary (section 2.8) -------------------------------------------- */

static void party_wait(PnWait *w)
{
    int i;
    static const uint16_t yn_ids[] = {
        PNSYM_Task_HandleCancelChooseMonYesNoInput, PNSYM_Task_HandleCancelParticipationYesNoInput,
        PNSYM_Task_HandleSwitchItemsYesNoInput, PNSYM_Task_HandleSendMailToPCYesNoInput,
        PNSYM_Task_HandleLoseMailMessageYesNoInput, PNSYM_Task_HandleFieldMoveExitAreaYesNoInput,
        PNSYM_Task_HandleReplaceMoveYesNoInput, PNSYM_Task_HandleStopLearningMoveYesNoInput,
        PNSYM_Task_HandleSwitchItemsFromBagYesNoInput,
    };
    uint8_t cur = 0, cnt = 0;
    if (!CB2_IS(CB2_UpdatePartyMenu))
        return;
    /* v10: the mon a submenu, YES/NO or message wait acts on (the choose wait below overrides it) */
    if (gPartyMenu.slotId >= 0 && gPartyMenu.slotId <= 7)
        w->party_cursor = (uint8_t)(gPartyMenu.slotId + 1);
    if (TASK_ON(Task_HandleChooseMonInput)) {
        int slot;
        if (FADE)
            return;
        slot = (gPartyMenu.action == PARTY_ACTION_SWITCH || gPartyMenu.action == PARTY_ACTION_SOFTBOILED)
                   ? gPartyMenu.slotId2 : gPartyMenu.slotId;
        set_wait(w, gMain.inBattle ? PN_INPUT_BATTLE_PARTY : PN_INPUT_MENU, PN_WAIT_PARTY_CHOOSE,
                 PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        set_menu(w, PN_MENU_FULLSCREEN, (uint8_t)slot, 0, 255);
        if (slot >= 0 && slot <= 7)
            w->party_cursor = (uint8_t)(slot + 1); /* v10 */
        /* v4: the item being used on / given to the chosen mon (gSpecialVar_ItemId: the bag's or the item
         * PC's selection, CB2_ChooseMonToGiveItem) */
        if (gPartyMenu.action == PARTY_ACTION_USE_ITEM || gPartyMenu.action == PARTY_ACTION_GIVE_ITEM
            || gPartyMenu.action == PARTY_ACTION_GIVE_PC_ITEM)
            w->context_item = gSpecialVar_ItemId;
        return;
    }
    if (TASK_ON(Task_HandleSelectionMenuInput)) {
        /* v4: struct PartyMenuInternal (party_menu.c, file-local): actions[8] @15, numActions @23 */
        const u8 *const *pp = SYMPTR(const u8 *const *, sPartyMenuInternal);
        uint16_t keys;
        if (FADE)
            return;
        keys = menu_keys(1, &cur, &cnt) | PN_KEY_B;
        set_wait(w, gMain.inBattle ? PN_INPUT_BATTLE_PARTY : PN_INPUT_MENU, PN_WAIT_PARTY_SUBMENU, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        if (pp != NULL && *pp != NULL) {
            const u8 *in = *pp;
            uint8_t acts[PN_OBS_MENU_ACTIONS];
            int i, n = in[23] > PN_OBS_MENU_ACTIONS ? PN_OBS_MENU_ACTIONS : in[23];
            for (i = 0; i < n; i++)
                acts[i] = party_action(in[15 + i]);
            set_actions(w, acts, n);
        }
        return;
    }
    for (i = 0; i < (int)(sizeof(yn_ids) / sizeof(yn_ids[0])); i++) {
        if (pn_obs_task_find(gPnObsSym[yn_ids[i]]) >= 0) {
            yesno_menu(w, PN_WAIT_PARTY_YESNO, PN_MENU_FULLSCREEN);
            /* v7: "Delete a move to make room for X?" / "Stop learning X?" (level-up by Rare Candy, TM / HM,
             * tutor): party_menu.c gPartyMenu.learnMoveId (#define of data[0]) for gPartyMenu.slotId */
            if (yn_ids[i] == PNSYM_Task_HandleReplaceMoveYesNoInput || yn_ids[i] == PNSYM_Task_HandleStopLearningMoveYesNoInput)
                set_offer(w, (u16)gPartyMenu.data[0], gPartyMenu.slotId);
            return;
        }
    }
    if (TASK_ON(Task_HandleRestoreWhichMoveInput)) {
        uint16_t keys = menu_keys(1, &cur, &cnt) | PN_KEY_B;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_PARTY_MOVE, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        return;
    }
    if (TASK_ON(Task_CancelAfterAorBPress) || TASK_ON(Task_DisplayLevelUpStatsPg2)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_PARTY_ACK, K_AB);
        return;
    }
    if (TASK_ON(Task_PrintAndWaitForText) && pr_wait(6)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_PARTY_MSG, K_AB);
        return;
    }
    /* Oak's voice-over in the first battle's party menu: the task runs
     * RunTextPrinters_CheckActive(data[0]) every frame */
    {
        int t = TASK(Task_FirstBattleEnterParty_RunPrinterMsg1);
        if (t < 0)
            t = TASK(Task_FirstBattleEnterParty_RunPrinterMsg2);
        if (t >= 0 && TD(t, 0) >= 0 && TD(t, 0) < NUM_TEXT_PRINTERS && pr_wait(TD(t, 0)))
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_PARTY_MSG, K_AB);
    }
}

static void summary_wait(PnWait *w)
{
    const u8 *const *pp = SYMPTR(const u8 *const *, sMonSummaryScreen);
    const u8 *s;
    int ht;
    uint32_t f;
    if (!CB2_IS(CB2_RunPokemonSummaryScreen) || pp == NULL || *pp == NULL)
        return;
    s = *pp;
    ht = s[12312]; /* inputHandlerTaskId */
    if (ht >= NUM_TASKS || !gTasks[ht].isActive)
        return;
    f = (uint32_t)(uintptr_t)gTasks[ht].func;
    if (f == SYM(Task_InputHandler_Info) && f != 0) {
        /* v3.2: and no page flip buffered during the last flip animation (lastPageFlipDirection
         * @0x3300 != 0xff): IsPageFlipInput (pokemon_summary_screen.c:1069) then returns TRUE
         * without reading a key and the handler flips or returns, whatever is pressed */
        if (s[12912] == PSS_STATE3270_HANDLEINPUT && s[0x3300] == 0xff && !TASK_ON(Task_PokeSum_FlipPages)
            && !TASK_ON(Task_PokeSum_SwitchDisplayedPokemon)) {
            set_wait(w, PN_INPUT_MENU, PN_WAIT_SUMMARY, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
            set_menu(w, PN_MENU_FULLSCREEN, s[12820] /* curPageIndex */, 0, 255);
        }
    } else if (f == SYM(Task_HandleInput_SelectMove) && f != 0) {
        if (s[12936] == 0) {
            set_wait(w, PN_INPUT_MENU, PN_WAIT_SUMMARY_MOVES, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
            set_move_cursor(w); /* v8 */
        }
    } else if (f == SYM(Task_InputHandler_SelectOrForgetMove) && f != 0) {
        if (s[12936] == 2) {
            const u8 *last = SYMPTR(const u8 *, sLastViewedMonIndex_pss);
            set_wait(w, PN_INPUT_MENU, PN_WAIT_SUMMARY_FORGET, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
            set_move_cursor(w); /* v8 */
            /* v7: ShowSelectMovePokemonSummaryScreen stores the move to learn in moveIds[4] (@12888 + 8; 0 for the
             * Move Deleter) of mode PSS_MODE_SELECT_MOVE (@12808); the mon is gPlayerParty[sLastViewedMonIndex]
             * (every caller passes gPlayerParty; the select-move mode does not switch mons) */
            if (s[12808] == PSS_MODE_SELECT_MOVE && last != NULL) {
                u16 mv;
                memcpy(&mv, s + 12888 + 2 * MAX_MON_MOVES, sizeof(mv));
                set_offer(w, mv, *last);
            }
        }
    }
}

/* ---- item PC (section 2.10.1, PC-ITEM) ------------------------------------------------- */

/* struct ItemPcResources (item_pc.c, file-local): savedCallback @0, moveModeOrigPos @4,
 * itemMenuIconSlot @5, maxShowed @6, nItems @7; struct ItemPcStaticResources: savedCallback @0,
 * scroll (u16) @4, row (u16) @6 (synced from the live list every frame of Task_ItemPcMain and
 * on exit; the submenu / quantity / messages act on pcItems[scroll + row]) */
static uint16_t item_pc_selected(void)
{
    const u8 *st = SYMPTR(const u8 *, sListMenuState_itempc);
    u16 scroll, row;
    if (st == NULL || gSaveBlock1Ptr == NULL)
        return 0;
    memcpy(&scroll, st + 4, 2);
    memcpy(&row, st + 6, 2);
    return scroll + row < PC_ITEMS_COUNT ? gSaveBlock1Ptr->pcItems[scroll + row].itemId : 0;
}

static void item_pc_wait(PnWait *w)
{
    static const uint8_t sub[3] = {PN_ACTION_WITHDRAW, PN_ACTION_GIVE, PN_ACTION_CANCEL};
    uint8_t cur = 0, cnt = 0;
    int t;
    if (!CB2_IS(ItemPc_MainCB))
        return;
    t = TASK(Task_ItemPcMain);
    if (t >= 0) {
        /* guard: !gPaletteFade.active && !IsPCScreenEffectRunning_TurnOn() */
        if (!FADE && !TASK_ON(Task_PCScreenEffect_TurnOn)) {
            const u8 *const *rp = SYMPTR(const u8 *const *, sStateDataPtr_itempc);
            set_wait(w, PN_INPUT_MENU, PN_WAIT_ITEM_PC, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN | PN_KEY_SELECT);
            /* v4: list cursor (row = PC slot, the last row is CANCEL) and the item under it */
            if (rp != NULL && *rp != NULL && list_task_ok(TD(t, 0)) && gSaveBlock1Ptr != NULL) {
                u16 scroll = 0, row = 0;
                u8 n = (*rp)[7];
                ListMenuGetScrollAndRow((u8)TD(t, 0), &scroll, &row);
                set_menu(w, PN_MENU_FULLSCREEN, (uint8_t)(scroll + row), (uint8_t)(n + 1), 255);
                if (scroll + row < n && scroll + row < PC_ITEMS_COUNT)
                    w->cursor_item = gSaveBlock1Ptr->pcItems[scroll + row].itemId;
            }
        }
        return;
    }
    if (TASK_ON(Task_ItemPcSubmenuRun)) {
        uint16_t keys = menu_keys(0, &cur, &cnt) | PN_KEY_B;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_ITEM_PC, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        set_actions(w, sub, 3);
        w->context_item = item_pc_selected();
        return;
    }
    if (TASK_ON(Task_ItemPcMoveItemModeRun)) {
        set_wait(w, PN_INPUT_MENU, PN_WAIT_ITEM_PC, PN_KEY_A | PN_KEY_B | PN_KEY_SELECT | PN_KEY_UP | PN_KEY_DOWN);
        return;
    }
    t = TASK(Task_ItemPcHandleWithdrawMultiple);
    if (t >= 0) {
        set_wait(w, PN_INPUT_QUANTITY, PN_WAIT_ITEM_PC, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        w->quantity = (uint16_t)TD(t, 8);
        w->quantity_max = (uint16_t)TD(t, 2);
        w->context_item = item_pc_selected();
        return;
    }
    if (TASK_ON(Task_ItemPcWaitButtonAndFinishWithdrawMultiple) || TASK_ON(Task_ItemPcWaitButtonWithdrawMultipleFailed)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_ITEM_PC, K_AB);
        w->context_item = item_pc_selected();
        return;
    }
    msg_cont(w, PN_WAIT_ITEM_PC);
    if (w->waiting)
        w->context_item = item_pc_selected();
}

/* ---- v2 screens: Pokedex, PC box, region/Fly map, TM case, berry pouch, relearner ---- */

/* File-local structs are read at the offsets of reference/layout_probes.tsv (native
 * column, identical to the ARM build); every pointer is re-read at each call and only
 * dereferenced under the callback2 + task proof. */
#define RD8(p, off) (((const u8 *)(p))[off])
#define RD16(p, off) (*(const u16 *)((const u8 *)(p) + (off)))
#define RD32(p, off) (*(const u32 *)((const u8 *)(p) + (off)))

static const u8 *deref_ptr(uint32_t sym)
{
    const u8 *const *pp = (const u8 *const *)(uintptr_t)sym;
    return pp != NULL ? *pp : NULL;
}

static uint32_t task_func(int t)
{
    return (t >= 0 && t < NUM_TASKS && gTasks[t].isActive) ? (uint32_t)(uintptr_t)gTasks[t].func : 0;
}

static int fn_is(uint32_t f, uint32_t sym)
{
    return sym != 0 && f == sym;
}

/* scroll + row of a list menu task (0 if the list task is gone) */
static uint8_t list_cursor(int lt)
{
    u16 scroll = 0, row = 0;
    if (!list_task_ok(lt))
        return 0;
    ListMenuGetScrollAndRow((u8)lt, &scroll, &row);
    return (uint8_t)(scroll + row);
}

/* struct PokedexScreenData (pokedex_screen.c): taskId @0, state @1, modeSelectListMenuId @23,
 * numMonsOnPage @44, categoryCursorPosInPage @45, orderedListMenuTaskId @65 */
static void dex_wait(PnWait *w)
{
    const u8 *d = deref_ptr(SYM(sPokedexScreenData)); /* freed without NULL on exit: */
    uint32_t f;
    u8 st;
    if (d == NULL || !(CB2_IS(BattleMainCB2) || CB2_IS(CB2_PokedexScreen))) /* read only under these */
        return;
    f = task_func(RD8(d, 0));
    st = RD8(d, 1);
    if (CB2_IS(BattleMainCB2)) {
        /* post-catch entry page (Cmd_displaydexinfo): state 11 waits for JOY_NEW(A | B) */
        if (gMain.inBattle && fn_is(f, SYM(Task_DexScreen_RegisterMonToPokedex)) && st == 11)
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_DEX_REGISTER, K_AB);
        return;
    }
    /* CB2_PokedexScreen runs the tasks only if !gPaletteFade.active (or a BG copy is
     * pending); require no fade */
    if (!CB2_IS(CB2_PokedexScreen) || FADE)
        return;
    if (fn_is(f, SYM(Task_PokedexScreen)) && st == 6) {
        /* ListMenu_ProcessInput (no multiple scroll): A, B, UP/DOWN */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_DEX_MENU, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
        set_menu(w, PN_MENU_FULLSCREEN, list_cursor(RD8(d, 23)), 0, 255);
    } else if ((fn_is(f, SYM(Task_DexScreen_NumericalOrder)) || fn_is(f, SYM(Task_DexScreen_CharacteristicOrder))) && st == 6) {
        /* ordered list (scrollMultiple = LIST_MULTIPLE_SCROLL_DPAD: LEFT/RIGHT page) */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_DEX_LIST, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        set_menu(w, PN_MENU_FULLSCREEN, list_cursor(RD8(d, 65)), 0, 255);
    } else if (fn_is(f, SYM(Task_DexScreen_CategorySubmenu)) && st == 5) {
        /* habitat page: A (seen mons), B, LEFT/RIGHT (cursor, page flip or close) */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_DEX_CATEGORY, PN_KEY_A | PN_KEY_B | PN_KEY_LEFT | PN_KEY_RIGHT);
        set_menu(w, PN_MENU_FULLSCREEN, RD8(d, 45), RD8(d, 44), 255);
    } else if (fn_is(f, SYM(Task_DexScreen_CategorySubmenu)) && (st == 17 || st == 23)) {
        /* entry / area page: A next page, B back, START plays the cry */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_DEX_PAGE, PN_KEY_A | PN_KEY_B | PN_KEY_START);
    } else if (fn_is(f, SYM(Task_DexScreen_ShowMonPage)) && (st == 5 || st == 9)) {
        /* state 5 also scrolls to the previous/next mon with UP/DOWN */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_DEX_PAGE,
                 PN_KEY_A | PN_KEY_B | PN_KEY_START | (st == 5 ? (PN_KEY_UP | PN_KEY_DOWN) : 0));
    }
}

/* Pokemon storage (pokemon_storage_system_tasks.c): one task (gStorage->taskId) whose
 * func is swapped by SetPokeStorageTask (state reset to 0); the sub states below read
 * input unconditionally (messages are printed instantly, TEXT_SKIP_DRAW). */
enum { PSS_M = 1, PSS_Y, PSS_B, PSS_X, PSS_MARK };
#define PSS_KEYS_X (PN_KEY_A | PN_KEY_B | PN_KEY_DPAD)

static void pc_box_wait(PnWait *w)
{
    static const struct { uint16_t sym; uint8_t state, kind; } sub[] = {
        {PNSYM_Task_PokeStorageMain, 3, PSS_X},    {PNSYM_Task_PokeStorageMain, 6, PSS_X},
        {PNSYM_Task_OnSelectedMon, 2, PSS_M},      {PNSYM_Task_OnSelectedMon, 6, PSS_X},
        {PNSYM_Task_WithdrawMon, 1, PSS_X},        {PNSYM_Task_DepositMenu, 1, PSS_B},
        {PNSYM_Task_DepositMenu, 4, PSS_X},        {PNSYM_Task_ReleaseMon, 1, PSS_Y},
        {PNSYM_Task_ReleaseMon, 4, PSS_X},         {PNSYM_Task_ReleaseMon, 5, PSS_X},
        {PNSYM_Task_ReleaseMon, 9, PSS_X},         {PNSYM_Task_ReleaseMon, 10, PSS_X},
        {PNSYM_Task_ReleaseMon, 12, PSS_X},        {PNSYM_Task_ReleaseMon, 13, PSS_X},
        {PNSYM_Task_ShowMarkMenu, 1, PSS_MARK},    {PNSYM_Task_GiveMovingItemToMon, 3, PSS_X},
        {PNSYM_Task_ItemToBag, 2, PSS_X},          {PNSYM_Task_ItemToBag, 3, PSS_X},
        {PNSYM_Task_SwitchSelectedItem, 3, PSS_X}, {PNSYM_Task_ShowItemInfo, 4, PSS_X},
        {PNSYM_Task_CloseBoxWhileHoldingItem, 1, PSS_Y}, {PNSYM_Task_CloseBoxWhileHoldingItem, 2, PSS_X},
        {PNSYM_Task_PrintCantStoreMail, 2, PSS_X}, {PNSYM_Task_HandleBoxOptions, 2, PSS_M},
        {PNSYM_Task_HandleBoxOptions, 1, PSS_M}, /* state 1 falls through into 2 (IsMenuLoading is a stub) */
        {PNSYM_Task_HandleWallpapers, 2, PSS_M},   {PNSYM_Task_HandleWallpapers, 4, PSS_M},
        {PNSYM_Task_JumpBox, 1, PSS_B},            {PNSYM_Task_OnCloseBoxPressed, 1, PSS_X},
        {PNSYM_Task_OnCloseBoxPressed, 2, PSS_Y},  {PNSYM_Task_OnBPressed, 1, PSS_X},
        {PNSYM_Task_OnBPressed, 2, PSS_Y},
    };
    const struct PokemonStorageSystemData *s = gStorage;
    const s8 *area = SYMPTR(const s8 *, sCursorArea_pss);
    const s8 *pos = SYMPTR(const s8 *, sCursorPosition_pss);
    uint8_t cur = 0, cnt = 0;
    uint32_t f;
    int i;
    if (!CB2_IS(CB2_PokeStorage) || s == NULL || area == NULL)
        return;
    f = task_func(s->taskId);
    if (fn_is(f, SYM(Task_PokeStorageMain)) && s->state == 0) {
        /* HandleInput (pokemon_storage_system_data.c) by cursor area */
        uint16_t keys;
        switch (*area) {
        case CURSOR_AREA_IN_BOX:
            if (s->inBoxMovingMode == 1 /* MOVE_MODE_MULTIPLE_SELECTING: held A, no new key */)
                return;
            keys = s->inBoxMovingMode == 2 /* MOVE_MODE_MULTIPLE_MOVING */ ? (PN_KEY_DPAD | PN_KEY_A | PN_KEY_B)
                                                                         : (PN_KEY_DPAD | PN_KEY_START | PN_KEY_A | PN_KEY_B | PN_KEY_SELECT);
            break;
        case CURSOR_AREA_IN_PARTY:
        case CURSOR_AREA_BOX_TITLE:
            keys = PN_KEY_DPAD | PN_KEY_A | PN_KEY_B | PN_KEY_SELECT;
            break;
        case CURSOR_AREA_BUTTONS:
            keys = PN_KEY_DPAD | PN_KEY_START | PN_KEY_A | PN_KEY_B | PN_KEY_SELECT;
            break;
        default:
            return;
        }
        set_wait(w, PN_INPUT_MAP_CURSOR, PN_WAIT_PC_BOX, keys);
        set_menu(w, PN_MENU_FULLSCREEN, pos != NULL ? (uint8_t)*pos : 0, 0, (uint8_t)*area);
        return;
    }
    for (i = 0; i < (int)(sizeof(sub) / sizeof(sub[0])); i++) {
        if (s->state != sub[i].state || !fn_is(f, gPnObsSym[sub[i].sym]))
            continue;
        switch (sub[i].kind) {
        case PSS_M: /* HandleMenuInput: A, B, UP, DOWN (UP/DOWN always play SE_SELECT) */
            menu_keys(1, &cur, &cnt);
            set_wait(w, PN_INPUT_MENU, PN_WAIT_PC_BOX_MENU, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
            set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
            {
                /* v4: the menu's entries (gStorage->menuItems[menuItemsCount].textId, SetMenuText) */
                uint8_t acts[PN_OBS_MENU_ACTIONS];
                int k, n = s->menuItemsCount > PN_OBS_MENU_ACTIONS ? PN_OBS_MENU_ACTIONS : s->menuItemsCount;
                for (k = 0; k < n; k++)
                    acts[k] = storage_action(s->menuItems[k].textId);
                set_actions(w, acts, n);
            }
            break;
        case PSS_MARK: /* HandleMonMarkingsMenuInput: A, B, UP, DOWN */
            set_wait(w, PN_INPUT_MENU, PN_WAIT_PC_BOX_MENU, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
            break;
        case PSS_Y: /* Menu_ProcessInputNoWrapClearOnChoose */
            yesno_menu(w, PN_WAIT_PC_BOX_YESNO, PN_MENU_FULLSCREEN);
            if (sub[i].sym == PNSYM_Task_ReleaseMon) { /* v4: "Release this POKeMON?" YES = RELEASE */
                static const uint8_t rel[2] = {PN_ACTION_RELEASE, PN_ACTION_CANCEL};
                set_actions(w, rel, 2);
            }
            break;
        case PSS_B: /* HandleBoxChooseSelectionInput: A, B, LEFT, RIGHT */
            set_wait(w, PN_INPUT_BOX_CHOOSE, PN_WAIT_PC_BOX_CHOOSE, PN_KEY_A | PN_KEY_B | PN_KEY_LEFT | PN_KEY_RIGHT);
            w->box_choose_cursor = s->chooseBoxMenu.curBox; /* v4 */
            break;
        default: /* JOY_NEW(A_BUTTON | B_BUTTON | DPAD_ANY) */
            set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_PC_BOX_MSG, PSS_KEYS_X);
            break;
        }
        return;
    }
}

/* region map (Town Map, wall map) and Fly map (region_map.c): struct RegionMap mainState
 * @18336, type @18326, savedCallback @18364; struct MapCursor x @0, y @2, inputHandler
 * @16; struct FlyMap state @0. Input is read by HandleRegionMapInput only. */
#define PN_REGION_MAP_WIDTH 22
#define PN_REGION_MAP_HEIGHT 15

static int region_map_type(void)
{
    const u8 *rm = deref_ptr(SYM(sRegionMap));
    if (!(CB2_IS(CB2_RegionMap) || CB2_IS(CB2_OpenRegionMap)) || rm == NULL)
        return -1;
    return RD8(rm, 18326);
}

static void region_map_wait(PnWait *w)
{
    const u8 *rm = deref_ptr(SYM(sRegionMap));
    const u8 *mc = deref_ptr(SYM(sMapCursor));
    const u8 *fly = deref_ptr(SYM(sFlyMap));
    int ctx;
    s16 x, y;
    uint16_t keys;
    if (!CB2_IS(CB2_RegionMap) || rm == NULL || mc == NULL || !fn_is(RD32(mc, 16), SYM(HandleRegionMapInput)))
        return;
    if (TASK_ON(Task_RegionMap) && RD8(rm, 18336) == 3)
        ctx = PN_WAIT_REGION_MAP;
    else if (TASK_ON(Task_FlyMap) && fly != NULL && RD8(fly, 0) == 4)
        ctx = PN_WAIT_FLY_MAP;
    else
        return;
    x = (s16)RD16(mc, 0);
    y = (s16)RD16(mc, 2);
    keys = PN_KEY_A | PN_KEY_B | PN_KEY_START; /* A may be inert off the buttons/icons; B cancels */
    if (y > 0)
        keys |= PN_KEY_UP;
    if (y < PN_REGION_MAP_HEIGHT - 1)
        keys |= PN_KEY_DOWN;
    if (x > 0)
        keys |= PN_KEY_LEFT;
    if (x < PN_REGION_MAP_WIDTH - 1)
        keys |= PN_KEY_RIGHT;
    if (fn_is(RD32(rm, 18364), SYM(CB2_ReturnToField)))
        keys |= PN_KEY_SELECT;
    set_wait(w, PN_INPUT_MAP_CURSOR, ctx, keys);
    set_menu(w, PN_MENU_FULLSCREEN, (uint8_t)x, (uint8_t)y, 255);
}

/* TM case (tm_case.c, run loop CB2_Idle): the list task's data[0] is the list menu task;
 * sTMCaseStaticResources (tm_case.c:85-92): exitCallback @0, menuType @4, allowSelectClose @5,
 * unused @6, selectedRow @8, scrollOffset @10.
 * v4: *sTMCaseDynamicResources (tm_case.c:95-107): nextScreenCallback @0, discSpriteId @4,
 * maxTMsShown @5, numTMs @6, contextMenuWindowId @7, scrollArrowsTaskId @8, currItem (u16) @10,
 * menuActionIndices (const u8 *) @12, numMenuActions @16, seqId @18, unused[8] @20 (28 bytes).
 * currItem is not the selected TM (CreateQuantityScrollArrows uses it as the quantity arrows'
 * counter); the TM acted on is gSpecialVar_ItemId (Task_HandleListInput, tm_case.c:926).
 * The list task's data: tSelection [1], tQuantityOwned [2], tQuantitySelected [8] (tm_case.c:462-465). */
static uint8_t tm_case_action(u8 a)
{
    switch (a) {
    case 0: return PN_ACTION_USE;    /* ACTION_USE */
    case 1: return PN_ACTION_GIVE;   /* ACTION_GIVE */
    case 2: return PN_ACTION_CANCEL; /* ACTION_EXIT */
    default: return PN_ACTION_OTHER;
    }
}

static void tm_case_wait(PnWait *w)
{
    const u8 *st = SYMPTR(const u8 *, sTMCaseStaticResources);
    const u8 *dyn = deref_ptr(SYM(sTMCaseDynamicResources));
    uint8_t cur = 0, cnt = 0;
    int t;
    if (!CB2_IS(CB2_Idle_tm) || st == NULL)
        return;
    t = TASK(Task_HandleListInput_tm);
    if (t >= 0) {
        uint16_t keys = PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN;
        if (FADE || !list_task_ok(TD(t, 0)))
            return;
        if (RD8(st, 5) == 1)
            keys |= PN_KEY_SELECT;
        set_wait(w, gMain.inBattle ? PN_INPUT_BATTLE_BAG : PN_INPUT_MENU, PN_WAIT_TM_CASE, keys);
        set_menu(w, PN_MENU_FULLSCREEN, list_cursor(TD(t, 0)), dyn != NULL ? (uint8_t)(RD8(dyn, 6) + 1) : 0, 255);
        /* v4: the list shows the TM/HM pocket in stored order (sorted, HMs first; tm_case.c:644-659),
         * CLOSE last; Task_HandleListInput copies the list's scroll/row into the static resources */
        if (dyn != NULL) {
            u16 pos = (u16)(RD16(st, 10) + RD16(st, 8));
            if (pos < RD8(dyn, 6))
                w->cursor_item = BagGetItemIdByPocketPosition(POCKET_TM_CASE, pos);
        }
        return;
    }
    if (TASK_ON(Task_ContextMenu_HandleInput_tm)) {
        uint16_t keys = menu_keys(0, &cur, &cnt) | PN_KEY_B;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_TM_CASE_MENU, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        /* v4: menuActionIndices[numMenuActions] (USE GIVE EXIT, or GIVE EXIT in the Union Room) */
        if (dyn != NULL && RD32(dyn, 12) != 0) {
            const u8 *ix = (const u8 *)(uintptr_t)RD32(dyn, 12);
            uint8_t acts[PN_OBS_MENU_ACTIONS];
            int i, n = RD8(dyn, 16) > PN_OBS_MENU_ACTIONS ? PN_OBS_MENU_ACTIONS : RD8(dyn, 16);
            for (i = 0; i < n; i++)
                acts[i] = tm_case_action(ix[i]);
            set_actions(w, acts, n);
        }
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    t = TASK(Task_QuantitySelect_HandleInput_tm);
    if (t >= 0) {
        set_wait(w, PN_INPUT_QUANTITY, PN_WAIT_TM_CASE_QTY, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        w->quantity = (uint16_t)TD(t, 8);     /* AdjustQuantityAccordingToDPadInput(&tQuantitySelected, tQuantityOwned) */
        w->quantity_max = (uint16_t)TD(t, 2);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    if (TASK_ON(Task_CallYesOrNoCallback)) {
        yesno_menu(w, PN_WAIT_TM_CASE_YESNO, PN_MENU_FULLSCREEN);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    if (TASK_ON(Task_AfterSale_ReturnToList_tm)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_TM_CASE_MSG, K_AB);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    msg_cont(w, PN_WAIT_TM_CASE_MSG);
    if (w->waiting)
        w->context_item = gSpecialVar_ItemId;
}

/* berry pouch (berry_pouch.c, run loop CB2_BerryPouchIdle): sStaticCnt (struct BerryPouchStruct_203F370,
 * berry_pouch.c:46-54): savedCallback @0, type @4, allowSelect @5, unused_06 @6, listMenuSelectedRow @8,
 * listMenuScrollOffset @10; BERRYPOUCH_FROMBERRYCRUSH (5) ignores B (link: never reached here).
 * v4: *sResources (struct BerryPouchStruct_203F36C, :34-44): exitCallback @0, indicatorOffset @4,
 * indicatorTaskId @6, listMenuNumItems @7, listMenuMaxShowed @8, itemMenuIconId @9; the context menu
 * is sContextMenuOptions[sContextMenuNumOptions] (CreateNormalContextMenu, :997-1028, BP_ACTION_*);
 * the berry acted on is gSpecialVar_ItemId (Task_BerryPouchMain, :981); the task's data[8] / data[2]
 * are the quantity selected / owned (Task_Toss_SelectMultiple :1127, Task_Sell_SelectMultiple :1341) */
static uint8_t berry_pouch_action(u8 a)
{
    switch (a) {
    case 0: return PN_ACTION_USE;    /* BP_ACTION_USE */
    case 1: return PN_ACTION_TOSS;   /* BP_ACTION_TOSS */
    case 2: return PN_ACTION_GIVE;   /* BP_ACTION_GIVE */
    case 3: return PN_ACTION_CANCEL; /* BP_ACTION_EXIT */
    case 4: return PN_ACTION_NONE;   /* BP_ACTION_DUMMY */
    default: return PN_ACTION_OTHER;
    }
}

static void berry_pouch_wait(PnWait *w)
{
    const u8 *st = SYMPTR(const u8 *, sStaticCnt_brp);
    const u8 *res = deref_ptr(SYM(sResources_brp));
    uint8_t cur = 0, cnt = 0;
    int t;
    if (!CB2_IS(CB2_BerryPouchIdle) || st == NULL)
        return;
    t = TASK(Task_BerryPouchMain);
    if (t >= 0) {
        uint16_t keys = PN_KEY_A | PN_KEY_UP | PN_KEY_DOWN;
        if (FADE || !list_task_ok(TD(t, 0)) || RD8(st, 4) == 5)
            return;
        keys |= PN_KEY_B;
        if (RD8(st, 5) == 1)
            keys |= PN_KEY_SELECT;
        set_wait(w, gMain.inBattle ? PN_INPUT_BATTLE_BAG : PN_INPUT_MENU, PN_WAIT_BERRY_POUCH, keys);
        set_menu(w, PN_MENU_FULLSCREEN, list_cursor(TD(t, 0)), res != NULL ? (uint8_t)(RD8(res, 7) + 1) : 0, 255);
        /* v4: the list shows the berry pocket in stored order (SortAndCountBerries), CLOSE last;
         * Task_BerryPouchMain copies the list's scroll/row into sStaticCnt */
        if (res != NULL) {
            u16 pos = (u16)(RD16(st, 10) + RD16(st, 8));
            if (pos < RD8(res, 7))
                w->cursor_item = BagGetItemIdByPocketPosition(POCKET_BERRY_POUCH, pos);
        }
        return;
    }
    if (TASK_ON(Task_NormalContextMenu_HandleInput_brp)) {
        const u8 *const *op = SYMPTR(const u8 *const *, sContextMenuOptions_brp);
        const u8 *np = SYMPTR(const u8 *, sContextMenuNumOptions_brp);
        uint16_t keys = menu_keys(0, &cur, &cnt) | PN_KEY_B;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_BERRY_POUCH_MENU, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, cnt, 255);
        if (op != NULL && *op != NULL && np != NULL) {
            uint8_t acts[PN_OBS_MENU_ACTIONS];
            int i, n = *np > PN_OBS_MENU_ACTIONS ? PN_OBS_MENU_ACTIONS : *np;
            for (i = 0; i < n; i++)
                acts[i] = berry_pouch_action((*op)[i]);
            set_actions(w, acts, n);
        }
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    t = TASK(Task_Toss_SelectMultiple_brp);
    if (t < 0)
        t = TASK(Task_Sell_SelectMultiple_brp);
    if (t >= 0) {
        set_wait(w, PN_INPUT_QUANTITY, PN_WAIT_BERRY_POUCH_QTY, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        w->quantity = (uint16_t)TD(t, 8);     /* AdjustQuantityAccordingToDPadInput(&data[8], data[2]) */
        w->quantity_max = (uint16_t)TD(t, 2);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    if (TASK_ON(Task_CallYesOrNoCallback)) {
        yesno_menu(w, PN_WAIT_BERRY_POUCH_YESNO, PN_MENU_FULLSCREEN);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    if (TASK_ON(Task_WaitButtonThenTossBerries_brp) || TASK_ON(Task_SellBerries_WaitButton_brp)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_BERRY_POUCH_MSG, K_AB);
        w->context_item = gSpecialVar_ItemId;
        return;
    }
    msg_cont(w, PN_WAIT_BERRY_POUCH_MSG);
    if (w->waiting)
        w->context_item = gSpecialVar_ItemId;
}

/* move relearner (learn_move.c): CB2_MoveRelearner runs the state machine only while
 * printer 7 is idle; struct LearnMoveGfxResources state @0, listMenuTaskId @611 */
static void relearner_wait(PnWait *w)
{
    const u8 *r = deref_ptr(SYM(sMoveRelearner));
    u8 st;
    if (!CB2_IS(CB2_MoveRelearner) || r == NULL)
        return;
    if (pr_wait(7)) { /* RunTextPrinters: a prompt inside the text */
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_RELEARNER_MSG, K_AB);
        return;
    }
    if (printer_active(7))
        return;
    st = RD8(r, 0);
    if (st == 4) { /* MENU_STATE_IDLE_BATTLE_MODE: ListMenu_ProcessInput, A, B */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_RELEARNER, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
        set_menu(w, PN_MENU_FULLSCREEN, list_cursor(RD8(r, 611)), RD8(r, 26), 255);
    } else if (st == 9 || st == 13 || st == 18 || st == 26) { /* YesNoMenuProcessInput */
        yesno_menu(w, PN_WAIT_RELEARNER_YESNO, PN_MENU_FULLSCREEN);
        /* v7: "Teach X?" (9), "X is trying to learn Y ... delete?" (18), "Stop trying to teach Y?" (26):
         * learnableMoves[selectedIndex] (u16 @232, u8 @2663; 0xFE = CANCEL) for selectedPartyMember (@608).
         * 13 is "Give up trying to teach a new move?" (B / CANCEL on the list): no move */
        if (st != 13 && RD8(r, 2663) < 25)
            set_offer(w, RD16(r, 232 + 2 * RD8(r, 2663)), RD8(r, 608));
    } else if (st == 33) { /* MENU_STATE_WAIT_FOR_A_BUTTON */
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_RELEARNER_MSG, PN_KEY_A);
    }
}

/* Hall of Fame (hall_of_fame.c, CB2_HofIdle runs the tasks every frame): the end of the
 * ceremony and the PC viewer's "no data" message wait for A; the PC viewer takes A (next
 * team / exit), B (exit), UP/DOWN (previous/next mon while one is left) */
static void hof_wait(PnWait *w)
{
    int t;
    if (!CB2_IS(CB2_HofIdle))
        return;
    if (TASK_ON(Task_Hof_ExitOnKeyPressed) || TASK_ON(Task_HofPC_ExitOnButtonPress)) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_HALL_OF_FAME, PN_KEY_A);
        return;
    }
    t = TASK(Task_HofPC_HandleInput);
    if (t >= 0) {
        uint16_t keys = K_AB;
        if (TD(t, 2) != 0)
            keys |= PN_KEY_UP;
        if (TD(t, 2) < TD(t, 4) - 1)
            keys |= PN_KEY_DOWN;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_HALL_OF_FAME, keys);
    }
}

/* ---- options, trainer card, shop buy menu, evolution (section 2.12, 2.10.3) --------- */

static void options_wait(PnWait *w)
{
    const u8 *const *pp = SYMPTR(const u8 *const *, sOptionMenuPtr);
    const u8 *o;
    if (!CB2_IS(CB2_InitOptionMenu) || !TASK_ON(Task_OptionMenu) || pp == NULL || *pp == NULL)
        return;
    o = *pp;
    if (o[16] == 2) { /* loadState */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_OPTIONS, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        set_menu(w, PN_MENU_FULLSCREEN, (uint8_t)*(const u16 *)(o + 14), 7, 255);
    }
}

static void trainer_card_wait(PnWait *w)
{
    const u8 *const *pp = SYMPTR(const u8 *const *, sTrainerCardDataPtr);
    if (!CB2_IS(CB2_TrainerCard) || !TASK_ON(Task_TrainerCard) || pp == NULL || *pp == NULL)
        return;
    if ((*pp)[0] == 10 || (*pp)[0] == 11) /* STATE_HANDLE_INPUT_FRONT / _BACK */
        set_wait(w, PN_INPUT_MENU, PN_WAIT_TRAINER_CARD, K_AB);
}

/* struct ShopData (shop.c, file-local): callback @0, itemList (const u16 *) @4, itemPrice @8,
 * selectedRow @0xC, scrollOffset @0xE, itemCount @0x10, itemsShowed @0x12, maxQuantity @0x14.
 * The buy task: tItemCount data[1], tItemId data[5], tListTaskId data[7]. */
static void shop_data(const u16 **items, u16 *count, u16 *maxq)
{
    const u8 *d = SYMPTR(const u8 *, sShopData);
    uint32_t p = 0;
    *items = NULL;
    *count = *maxq = 0;
    if (d == NULL)
        return;
    memcpy(&p, d + 4, 4);
    *items = (const u16 *)(uintptr_t)p;
    memcpy(count, d + 0x10, 2);
    memcpy(maxq, d + 0x14, 2);
}

static void shop_buy_wait(PnWait *w)
{
    const u16 *items;
    u16 count, maxq;
    int t;
    if (!CB2_IS(CB2_BuyMenu))
        return;
    t = TASK(Task_BuyMenu);
    if (t >= 0) {
        if (!FADE) {
            set_wait(w, PN_INPUT_MENU, PN_WAIT_SHOP_BUY, PN_KEY_A | PN_KEY_B | PN_KEY_UP | PN_KEY_DOWN);
            /* v4: list cursor and the item under it (the last row is CANCEL) */
            shop_data(&items, &count, &maxq);
            if (list_task_ok(TD(t, 7))) {
                u16 scroll = 0, row = 0;
                ListMenuGetScrollAndRow((u8)TD(t, 7), &scroll, &row);
                set_menu(w, PN_MENU_MART, (uint8_t)(scroll + row), (uint8_t)(count + 1), 255);
                if (items != NULL && scroll + row < count)
                    w->cursor_item = items[scroll + row];
            }
        }
        return;
    }
    t = TASK(Task_BuyHowManyDialogueHandleInput);
    if (t >= 0) {
        set_wait(w, PN_INPUT_QUANTITY, PN_WAIT_SHOP_QTY, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
        shop_data(&items, &count, &maxq);
        w->quantity = (uint16_t)TD(t, 1);
        w->quantity_max = maxq;
        w->context_item = (uint16_t)TD(t, 5);
        return;
    }
    t = TASK(Task_CallYesOrNoCallback);
    if (t >= 0) {
        yesno_menu(w, PN_WAIT_SHOP_YESNO, PN_MENU_MART);
        w->context_item = (uint16_t)TD(t, 5); /* the buy task itself runs the YES/NO (CreateYesNoMenuWithCallbacks) */
        return;
    }
    t = TASK(Task_ReturnToItemListAfterItemPurchase);
    if (t >= 0) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_SHOP_ACK, K_AB);
        w->context_item = (uint16_t)TD(t, 5);
        return;
    }
    msg_cont(w, PN_WAIT_SHOP_MSG_CONT);
}

static void evolution_wait(PnWait *w)
{
    int i;
    if (CB2_IS(CB2_EvolutionSceneUpdate)) {
        int t = TASK(Task_EvolutionScene);
        if (t >= 0 && TD(t, 0) == 22 /* EVOSTATE_REPLACE_MOVE */ && TD(t, 6) == 4 /* MVSTATE_HANDLE_YES_NO */) {
            uint16_t keys = PN_KEY_A | PN_KEY_B;
            keys |= gBattleCommunication[1] != 0 ? PN_KEY_UP : PN_KEY_DOWN;
            set_wait(w, PN_INPUT_YESNO, PN_WAIT_EVO_YESNO, keys);
            w->yesno_cursor = gBattleCommunication[1];
            /* v7: the delete-a-move and stop-learning prompts share this YES/NO: gMoveToLearn for tPartyId
             * (data[10]) */
            set_offer(w, gMoveToLearn, TD(t, 10));
            return;
        }
    }
    if (CB2_IS(CB2_EvolutionSceneUpdate) || CB2_IS(CB2_TradeEvolutionSceneUpdate) || CB2_IS(CB2_EggHatch_1)
        || CB2_IS(CB2_InGameTrade) || CB2_IS(CB2_PSA)) {
        for (i = 0; i < NUM_TEXT_PRINTERS; i++) {
            if (pr_wait(i)) {
                set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_SCENE_MSG, K_AB);
                return;
            }
        }
    }
}

static int battler_func_is(int b, uint32_t fn);

/* ---- v3.2: item-use animation (PSA), battle records, Teachy TV, Pokedude demos ---------- */

/* the item-use animation on the field (pokemon_special_anim.c, run loop CB2_PSA = RunTextPrinters +
 * RunTasks: medicine, Rare Candy, PP items, evolution stones, TMs). One task, its struct
 * PokemonSpecialAnim pointer in data[0..1] (GetWordTaskArg, task.c), state @0x92
 * (pokemon_special_anim_internal.h). Derived: the last state of Task_UseItem_Normal (12, :249) and
 * of Task_EvoStone_CantEvolve (8, :440), JOY_NEW(A | B) closes the scene. Not derived (stay 0,
 * OBS_SOURCES 2.12 PSA-WAIT): the cancel windows before them (A/B held while !cancelDisabled,
 * :165, :385; a new B in Task_UseTM_NoForget :462 and Task_MachineSet :505 aborts a TM), which an
 * agent made to press a key would always cancel. The message prompts stay SCENE_MSG. */
static void item_anim_wait(PnWait *w)
{
    const struct PokemonSpecialAnim *p;
    int t, normal = 1;
    if (!CB2_IS(CB2_PSA))
        return;
    t = TASK(Task_UseItem_Normal_psa);
    if (t < 0) {
        t = TASK(Task_EvoStone_CantEvolve_psa);
        normal = 0;
    }
    if (t < 0)
        return;
    p = (const struct PokemonSpecialAnim *)(uintptr_t)((u16)TD(t, 0) | ((u32)(u16)TD(t, 1) << 16));
    if (p != NULL && p->state == (normal ? 12 : 8))
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_ITEM_ANIM, K_AB);
}

/* BATTLE RECORDS (battle_records.c; Cable Club PC 2F and Trainer Tower): run loop MainCB2 =
 * RunTasks; Task_WaitButton (:168) reads JOY_NEW(A) || JOY_NEW(B) unguarded (entered after the
 * fade-in by Task_WaitFadeIn) */
static void battle_records_wait(PnWait *w)
{
    if (CB2_IS(MainCB2_records) && TASK_ON(Task_WaitButton_records))
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_BATTLE_RECORDS, K_AB);
}

/* struct TeachyTvCtrlBlk (teachy_tv.c:31) */
struct PnTeachyTvCtrlMirror {
    MainCallback callback;
    u8 mode, whichScript;
    u16 scrollOffset, selectedRow;
};
_Static_assert(sizeof(struct PnTeachyTvCtrlMirror) == 12, "TeachyTvCtrlBlk mirror");

/* Teachy TV (teachy_tv.c, run loop TeachyTvCallback = RunTasks + ...; one task) */
static void teachy_tv_wait(PnWait *w)
{
    const struct PnTeachyTvCtrlMirror *st = SYMPTR(const struct PnTeachyTvCtrlMirror *, sStaticResources_ttv);
    int t;
    if (!CB2_IS(TeachyTvCallback) || st == NULL)
        return;
    t = TASK(TeachyTvOptionListController);
    if (t >= 0) {
        /* :713: without a fade ListMenu_ProcessInput(data[0]): A picks a programme or CANCEL, B
         * (LIST_CANCEL) quits, UP/DOWN move one row, LEFT/RIGHT one page (scrollMultiple
         * LIST_MULTIPLE_SCROLL_DPAD), inert at the ends (ListMenuChangeSelection); then a new SELECT
         * quits unless the Teachy TV was opened from the start menu's bag */
        const struct ListMenu *lm;
        uint16_t keys = K_AB;
        uint8_t cur;
        int lt = TD(t, 0);
        if (FADE || !list_task_ok(lt))
            return;
        lm = (const struct ListMenu *)gTasks[lt].data;
        cur = list_cursor(lt);
        if (cur > 0)
            keys |= PN_KEY_UP | PN_KEY_LEFT;
        if (cur + 1 < lm->template.totalItems)
            keys |= PN_KEY_DOWN | PN_KEY_RIGHT;
        if (SYM(CB2_BagMenuFromStartMenu) != 0 && (uint32_t)(uintptr_t)st->callback != SYM(CB2_BagMenuFromStartMenu))
            keys |= PN_KEY_SELECT;
        set_wait(w, PN_INPUT_MENU, PN_WAIT_TEACHY_TV_MENU, keys);
        set_menu(w, PN_MENU_FULLSCREEN, cur, (uint8_t)lm->template.totalItems, 255);
        return;
    }
    t = TASK(TeachyTvRenderMsgAndSwitchClusterFuncs);
    if (t >= 0) {
        /* :808: JOY_NEW(B) ends the programme in every state (no fade guard); otherwise it runs
         * the programme's command data[3] of the script whichScript: EraseTextWindowIfKeyPressed
         * waits for JOY_NEW(A | B) (:939), the IdleIfTextPrinterIsActive commands run printer 0
         * (RunTextPrinters_CheckActive, :804/:932) whose page prompts take A or B */
        static const uint16_t scripts[] = {PNSYM_sBattleScript_ttv, PNSYM_sStatusScript_ttv, PNSYM_sMatchupsScript_ttv,
                                           PNSYM_sCatchingScript_ttv, PNSYM_sTMsScript_ttv,
                                           PNSYM_sRegisterKeyItemScript_ttv};
        static const uint8_t lengths[] = {19, 19, 19, 19, 16, 16}; /* teachy_tv.c:272-402 */
        uint16_t keys = PN_KEY_B;
        int s = st->whichScript, c = TD(t, 3);
        if (s >= 0 && s < 6 && gPnObsSym[scripts[s]] != 0 && c >= 0 && c < lengths[s]) {
            uint32_t cmd = ((const uint32_t *)(uintptr_t)gPnObsSym[scripts[s]])[c];
            if (fn_is(cmd, SYM(TTVcmd_EraseTextWindowIfKeyPressed))
                || ((fn_is(cmd, SYM(TTVcmd_IdleIfTextPrinterIsActive)) || fn_is(cmd, SYM(TTVcmd_IdleIfTextPrinterIsActive2)))
                    && pr_wait(0)))
                keys |= PN_KEY_A;
        }
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_TEACHY_TV, keys);
    }
}

/* the Pokedude's demo battle (Teachy TV): BattleMainCB2 (battle_main.c:1455) quits it on
 * JOY_HELD(B) whenever gBattleTypeFlags has BATTLE_TYPE_POKEDUDE; A acknowledges the voice-over
 * box (battle_controller_pokedude.c:2563 timer 3, :2624 timer 4: !IsTextPrinterActive(24) &&
 * JOY_NEW(A)) and the prompts of its printer 24 (no auto-scroll on B_WIN_OAK_OLD_MAN,
 * battle_message.c:2773) */
static int pokedude_battle_wait(PnWait *w)
{
    uint16_t keys = PN_KEY_B;
    uint32_t mf = (uint32_t)(uintptr_t)gBattleMainFunc;
    const struct TextPrinter *p24 = printer(B_WIN_OAK_OLD_MAN);
    int b, n;
    if (!CB2_IS(BattleMainCB2) || !(gBattleTypeFlags & BATTLE_TYPE_POKEDUDE))
        return 0;
    /* BattleMainCB1 runs first in the frame (main.c CallCallbacks): if it replaces callback2, the
     * B test of BattleMainCB2 does not run in that frame (R-ORDER). It can only through
     * gBattleMainFunc ReturnFromBattleToOverworld (battle_main.c:3938) or TryEvolvePokemon
     * (EvolutionScene) and the Pokedude controller functions that leave the battle screen
     * (battle_controller_pokedude.c:256 Pokedude_SetBattleEndCallbacks, :330 OpenPartyMenuToChooseMon,
     * :353 OpenBagAndChooseItem; their fade test may turn true in the VBlank before CB1): not waiting
     * while any of them is installed. */
    if (!gMain.inBattle || SYM(BattleMainCB1) == 0 || CB1 != SYM(BattleMainCB1))
        return 1; /* not the battle loop: nothing claimed */
    if (fn_is(mf, SYM(ReturnFromBattleToOverworld)) || fn_is(mf, SYM(TryEvolvePokemon)))
        return 1;
    n = gBattlersCount;
    if (n > MAX_BATTLERS_COUNT)
        n = MAX_BATTLERS_COUNT;
    for (b = 0; b < n; b++)
        if (battler_func_is(b, SYM(Pokedude_SetBattleEndCallbacks)) || battler_func_is(b, SYM(OpenPartyMenuToChooseMon_pokedude))
            || battler_func_is(b, SYM(OpenBagAndChooseItem_pokedude)))
            return 1;
    {
        for (b = 0; b < n && p24 != NULL; b++) {
            const struct PokedudeBattlerState *ps = gPokedudeBattlerStates[b];
            if (ps == NULL || p24->active)
                continue;
            if ((battler_func_is(b, SYM(PokedudeAction_PrintVoiceoverMessage)) && ps->timer == 3)
                || (battler_func_is(b, SYM(PokedudeAction_PrintMessageWithHealthboxPals)) && ps->timer == 4))
                keys |= PN_KEY_A;
        }
        if (pr_wait(B_WIN_OAK_OLD_MAN))
            keys |= PN_KEY_A;
    }
    set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_POKEDUDE, keys);
    return 1;
}

/* the Pokedude's bag, party menu and TM case (Teachy TV programmes, in and out of his battle):
 * the scripted tasks read only B (and A on the TM case's messages) */
static int pokedude_menu_wait(PnWait *w, int kind)
{
    if (kind == PN_SCREEN_BAG && CB2_IS(CB2_BagMenuRun)) {
        /* item_menu.c:2192 Task_BButtonInterruptTeachyTv (JOY_NEW(B)), called by the four tasks
         * only when !gPaletteFade.active (:2211, :2265, :2319, :2362) */
        if (TASK_ON(Task_Bag_TeachyTvRegister) || TASK_ON(Task_Bag_TeachyTvCatching) || TASK_ON(Task_Bag_TeachyTvStatus)
            || TASK_ON(Task_Bag_TeachyTvTMs)) {
            if (!FADE)
                set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_POKEDUDE, PN_KEY_B);
            return 1;
        }
    } else if (kind == PN_SCREEN_PARTY_MENU && CB2_IS(CB2_UpdatePartyMenu)) {
        /* party_menu.c:2032/:2084: !gPaletteFade.active && PartyMenuPokedudeIsCancelled (JOY_NEW(B), :2052) */
        if (TASK_ON(Task_PartyMenu_PokedudeStep) || TASK_ON(Task_PartyMenuFromBag_PokedudeStep)) {
            if (!FADE)
                set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_POKEDUDE, PN_KEY_B);
            return 1;
        }
        /* after his item use (:2093 gItemUseCB(taskId, Task_ClosePartyMenuAfterText)) the result
         * message ("... was cured of poisoning") waits like any party-menu message: printer 6 run by
         * Task_PrintAndWaitForText (:1529) every frame, its prompt takes A or B (PTY-MSG) */
        if (gMain.inBattle && (gBattleTypeFlags & BATTLE_TYPE_POKEDUDE) && TASK_ON(Task_PrintAndWaitForText)) {
            if (pr_wait(6))
                set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_PARTY_MSG, K_AB);
            return 1;
        }
    } else if (kind == PN_SCREEN_TM_CASE && CB2_IS(CB2_Idle_tm)) {
        /* tm_case.c:1353 Task_Pokedude_Run: JOY_NEW(B) jumps to state 21 (exit) from any state
         * < 21 (tPokedudeState = data[8]); states 10 and 20 wait for JOY_NEW(A | B); states 9 and 19
         * run the message printer (RunTextPrinters) whose page prompts take A or B */
        int t = TASK(Task_Pokedude_Run_tm);
        if (t >= 0) {
            int s = TD(t, 8);
            if (s < 21) {
                uint16_t keys = PN_KEY_B;
                if (s == 10 || s == 20 || ((s == 9 || s == 19) && pr_wait(6 /* WIN_MESSAGE */)))
                    keys |= PN_KEY_A;
                set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_POKEDUDE, keys);
            }
            return 1;
        }
    }
    return 0;
}

/* ---- battle (section 2.11) -------------------------------------------------------------- */

static int battle_guard(void)
{
    return gMain.inBattle && SYM(BattleMainCB1) != 0 && CB1 == SYM(BattleMainCB1) && CB2_IS(BattleMainCB2)
        && gBattleSpritesDataPtr != NULL && gBattleStruct != NULL && gBattleResources != NULL
        && !(gBattleTypeFlags & BATTLE_TYPE_LINK);
}

/* the battle script runner executes gBattlescriptCurrInstr in the next frame */
static int battle_script_runs(void)
{
    uint32_t f = (uint32_t)(uintptr_t)gBattleMainFunc;
    if (gBattleControllerExecFlags != 0 || gBattlescriptCurrInstr == NULL)
        return 0;
    if (f == SYM(RunBattleScriptCommands) && f != 0)
        return 1;
    if (((f == SYM(RunBattleScriptCommands_PopCallbacksStack) && f != 0) || (f == SYM(HandleEndTurn_FinishBattle) && f != 0))
        && gCurrentActionFuncId != B_ACTION_TRY_FINISH && gCurrentActionFuncId != B_ACTION_FINISHED)
        return 1;
    if (f == SYM(RunTurnActionsFunctions) && f != 0 && gBattleOutcome == 0 && gCurrentActionFuncId == B_ACTION_EXEC_SCRIPT)
        return 1;
    return 0;
}

static uint32_t battle_cmd(void)
{
    u8 op = *gBattlescriptCurrInstr;
    if (op >= PN_NUM_BATTLE_SCRIPT_CMDS)
        return 0;
    return (uint32_t)(uintptr_t)gBattleScriptingCommandsTable[op];
}

static int battler_func_is(int b, uint32_t fn)
{
    return fn != 0 && (uint32_t)(uintptr_t)gBattlerControllerFuncs[b] == fn;
}

static void battle_wait(PnWait *w)
{
    int b, n;
    if (!battle_guard())
        return;
    n = gBattlersCount;
    if (n > MAX_BATTLERS_COUNT)
        n = MAX_BATTLERS_COUNT;
    for (b = 0; b < n; b++) {
        if (GetBattlerSide(b) != B_SIDE_PLAYER)
            continue;
        if (battler_func_is(b, SYM(HandleInputChooseAction_player)) || battler_func_is(b, SYM(HandleInputChooseAction_oak))
            || battler_func_is(b, SYM(HandleInputChooseAction_safari))) {
            u8 c = gActionSelectionCursor[b];
            uint16_t keys = PN_KEY_A;
            keys |= (c & 1) ? PN_KEY_LEFT : PN_KEY_RIGHT;
            keys |= (c & 2) ? PN_KEY_UP : PN_KEY_DOWN;
            if ((gBattleTypeFlags & BATTLE_TYPE_DOUBLE) && !battler_func_is(b, SYM(HandleInputChooseAction_safari)))
                keys |= PN_KEY_START | PN_KEY_B;
            set_wait(w, PN_INPUT_BATTLE_ACTION, PN_WAIT_BATTLE_ACTION, keys);
            w->battle_input_battler = (uint8_t)b;
            return;
        }
        if (battler_func_is(b, SYM(HandleInputChooseMove)) || battler_func_is(b, SYM(OakOldManHandleInputChooseMove))) {
            /* HandleInputChooseMove (battle_controller_player.c): the cursor moves inside the
             * 2x2 grid of gNumberOfMovesToChoose moves; SELECT starts reordering */
            u8 c = gMoveSelectionCursor[b], n = gNumberOfMovesToChoose;
            uint16_t keys = PN_KEY_A | PN_KEY_B;
            if (c & 1)
                keys |= PN_KEY_LEFT;
            else if ((c ^ 1) < n)
                keys |= PN_KEY_RIGHT;
            if (c & 2)
                keys |= PN_KEY_UP;
            else if ((c ^ 2) < n)
                keys |= PN_KEY_DOWN;
            if (n > 1)
                keys |= PN_KEY_SELECT;
            set_wait(w, PN_INPUT_BATTLE_MOVE, PN_WAIT_BATTLE_MOVE, keys);
            w->battle_input_battler = (uint8_t)b;
            return;
        }
        if (battler_func_is(b, SYM(HandleInputChooseTarget))) {
            set_wait(w, PN_INPUT_BATTLE_TARGET, PN_WAIT_BATTLE_TARGET, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
            w->battle_input_battler = (uint8_t)b;
            return;
        }
        if (battler_func_is(b, SYM(HandleMoveSwitching))) {
            set_wait(w, PN_INPUT_BATTLE_MOVE_SWAP, PN_WAIT_BATTLE_SWAP, PN_KEY_A | PN_KEY_B | PN_KEY_DPAD);
            w->battle_input_battler = (uint8_t)b;
            return;
        }
    }
    if (!(gBattleTypeFlags & BATTLE_TYPE_POKEDUDE) && (pr_wait(B_WIN_MSG) || pr_wait(B_WIN_OAK_OLD_MAN))) {
        set_wait(w, PN_INPUT_DIALOG_ADVANCE, PN_WAIT_BATTLE_MSG, K_AB);
        return;
    }
    if (battle_script_runs()) {
        uint32_t cmd = battle_cmd();
        if (cmd == SYM(Cmd_yesnobox) && cmd != 0 && gBattleCommunication[0] == 1) {
            set_wait(w, PN_INPUT_YESNO, PN_WAIT_BATTLE_YESNO,
                     PN_KEY_A | PN_KEY_B | (gBattleCommunication[1] ? PN_KEY_UP : PN_KEY_DOWN));
            w->yesno_cursor = gBattleCommunication[1];
            return;
        }
        if (((cmd == SYM(Cmd_yesnoboxlearnmove) && cmd != 0) || (cmd == SYM(Cmd_yesnoboxstoplearningmove) && cmd != 0))
            && gBattleScripting.learnMoveState == 1) {
            set_wait(w, PN_INPUT_YESNO, PN_WAIT_BATTLE_LEARN,
                     PN_KEY_A | PN_KEY_B | (gBattleCommunication[1] ? PN_KEY_UP : PN_KEY_DOWN));
            w->yesno_cursor = gBattleCommunication[1];
            /* v7: gMoveToLearn (MonTryLearningNewMove) for gBattleStruct->expGetterMonId */
            if (gBattleStruct != NULL)
                set_offer(w, gMoveToLearn, gBattleStruct->expGetterMonId);
            return;
        }
        if (cmd == SYM(Cmd_trygivecaughtmonnick) && cmd != 0 && gBattleCommunication[0] == 1) {
            set_wait(w, PN_INPUT_YESNO, PN_WAIT_BATTLE_NICK,
                     PN_KEY_A | PN_KEY_B | (gBattleCommunication[1] ? PN_KEY_UP : PN_KEY_DOWN));
            w->yesno_cursor = gBattleCommunication[1];
            return;
        }
        if (cmd == SYM(Cmd_drawlvlupbox) && cmd != 0
            && (gBattleScripting.drawlvlupboxState == 6 || gBattleScripting.drawlvlupboxState == 8)) {
            set_wait(w, PN_INPUT_ANY_KEY, PN_WAIT_BATTLE_LVLUP, PN_KEY_A | PN_KEY_B);
            return;
        }
    }
}

/* the party menu / bag opened by a battle controller (BTL-PARTY, BTL-BAG) */
static int battle_owns(uint32_t fn_a, uint32_t fn_b)
{
    int b, n = gBattlersCount;
    if (!gMain.inBattle || SYM(BattleMainCB1) == 0 || CB1 != SYM(BattleMainCB1))
        return 0;
    if (n > MAX_BATTLERS_COUNT)
        n = MAX_BATTLERS_COUNT;
    for (b = 0; b < n; b++)
        if (battler_func_is(b, fn_a) || battler_func_is(b, fn_b))
            return 1;
    return 0;
}

/* ---- help overlay, Quest Log (sections 2.9, 2.3.4) ------------------------------------ */

static void help_wait(PnWait *w)
{
    const u8 *vs = SYMPTR(const u8 *, sVideoState);
    const u8 *delay = SYMPTR(const u8 *, sDelayTimer_help);
    u8 st;
    if (vs == NULL || vs[0x15] != 5)
        return;
    st = gHelpSystemListMenu.state;
    if (st == 9)
        set_wait(w, PN_INPUT_HELP, PN_WAIT_HELP, PN_KEY_A);
    else if ((st == 0 || st == 3) && delay != NULL && *delay == 0)
        set_wait(w, PN_INPUT_HELP, PN_WAIT_HELP, PN_KEY_A | PN_KEY_B | PN_KEY_L | PN_KEY_R | PN_KEY_DPAD);
    else if (st == 6)
        set_wait(w, PN_INPUT_HELP, PN_WAIT_HELP, PN_KEY_A | PN_KEY_B | PN_KEY_L | PN_KEY_R);
    if (w->waiting)
        w->menu_open = PN_MENU_HELP;
}

/* struct PlaybackControl (quest_log.c): byte 0 = state:4, playingEvent:2, endMode:2 */
static u8 ql_end_mode(void)
{
    const u8 *pc = SYMPTR(const u8 *, sPlaybackControl);
    return pc != NULL ? (u8)(pc[0] >> 6) : 1;
}

static void quest_log_wait(PnWait *w)
{
    if (!CB2_IS(CB2_Overworld) || SYM(CB1_Overworld) == 0 || CB1 != SYM(CB1_Overworld))
        return;
    if (gQuestLogState == QL_STATE_PLAYBACK_LAST) {
        /* "saved the game at ...": Task_WaitAtEndOfQuestLog ends on JOY_NEW(A|B) unless its
         * timer (data[0]) reached 127 or the playback was skipped (END_MODE_FINISH) */
        int t = TASK(Task_WaitAtEndOfQuestLog);
        if (t >= 0 && TD(t, 0) < 127 && ql_end_mode() != 1 /* END_MODE_FINISH */)
            set_wait(w, PN_INPUT_QUEST_LOG, PN_WAIT_QUEST_LOG, K_AB);
        return;
    }
    if (QuestLogScenePlaybackIsEnding())
        return;
    if (QL_GetPlaybackState() == QL_PLAYBACK_STATE_RUNNING || gQuestLogState == QL_STATE_PLAYBACK)
        set_wait(w, PN_INPUT_QUEST_LOG, PN_WAIT_QUEST_LOG, K_AB);
}

/* ---- decision procedure (section 2.1) --------------------------------------------------- */

void pn_obs_wait(PnWait *w, int soft_reset)
{
    int sf = save_failed(), help = help_open(), link = link_active(), ql = pn_obs_quest_log_playback();
    int kind;

    memset(w, 0, sizeof(*w));
    w->menu_id = 255;
    w->yesno_cursor = 255;
    w->battle_input_battler = 255;
    w->box_choose_cursor = 255;
    w->offered_slot = 255; /* v7 */
    w->help_open = (uint8_t)help;
    w->save_failed = (uint8_t)sf;
    w->link = (uint8_t)link;
    w->ql_playback = (uint8_t)ql;
    kind = screen_kind(sf, help);
    w->screen = (uint8_t)kind;
    w->field_state = (uint8_t)field_state();
    w->whiteout_win = (int8_t)(kind == PN_SCREEN_OVERWORLD ? whiteout_window() : -1);
    w->gate_ok = !help && !sf && !link && !ql && !soft_reset;

    /* L/R open the help overlay and swallow the frame (HELP button mode); read only in
     * RunHelpSystemCallback state 0 */
    if (!sf && help_state() == 0 && !link && gHelpSystemEnabled && gSaveBlock2Ptr != NULL
        && gSaveBlock2Ptr->optionsButtonMode == OPTIONS_BUTTON_MODE_HELP)
        w->flags |= PN_INPUT_FLAG_LR_SWALLOWED;

    if (soft_reset || sf || link)
        return; /* D0, D1 (save-failed: unreachable natively), D2 */
    if (help) {
        help_wait(w);
        return;
    }
    if (ql) {
        quest_log_wait(w);
        return;
    }

    switch (kind) {
    case PN_SCREEN_OVERWORLD:
    case PN_SCREEN_START_MENU:
    case PN_SCREEN_SAVE_MENU:
    case PN_SCREEN_MART:
    case PN_SCREEN_PC_MENU:
        field_wait(w);
        break;
    case PN_SCREEN_TITLE:
        title_wait(w);
        break;
    case PN_SCREEN_MAIN_MENU:
        main_menu_wait(w);
        break;
    case PN_SCREEN_NEW_GAME_SPEECH:
        oak_wait(w);
        break;
    case PN_SCREEN_NAMING:
        naming_wait(w);
        break;
    case PN_SCREEN_BAG:
        if (pokedude_menu_wait(w, kind))
            break; /* v3.2: the Pokedude's bag (B) */
        if (gMain.inBattle && !battle_owns(SYM(CompleteWhenChoseItem_player), SYM(CompleteWhenChoseItem_oak)))
            break; /* scripted demo bags (Old Man, Pokedude) */
        bag_wait(w);
        break;
    case PN_SCREEN_PARTY_MENU:
        if (pokedude_menu_wait(w, kind))
            break; /* v3.2: the Pokedude's party menu (B) */
        /* owned by the controller that opened it: POKEMON (WaitForMonSelection) or, v3.2, an item
         * used from the battle bag (the bag's controller CompleteWhenChoseItem stays installed while
         * the bag opens the party menu, item_menu.c ItemMenu_UseInBattle -> party_menu.c
         * ChooseMonForInBattleItem); never a scripted demo (their controllers are other functions) */
        if (gMain.inBattle && !battle_owns(SYM(WaitForMonSelection_player), SYM(WaitForMonSelection_oak))
            && !battle_owns(SYM(CompleteWhenChoseItem_player), SYM(CompleteWhenChoseItem_oak)))
            break;
        party_wait(w);
        break;
    case PN_SCREEN_SUMMARY:
        summary_wait(w);
        break;
    case PN_SCREEN_OPTIONS:
        options_wait(w);
        break;
    case PN_SCREEN_TRAINER_CARD:
        trainer_card_wait(w);
        break;
    case PN_SCREEN_ITEM_PC:
        item_pc_wait(w);
        break;
    case PN_SCREEN_POKEDEX:
        dex_wait(w);
        break;
    case PN_SCREEN_PC_BOX:
        pc_box_wait(w);
        break;
    case PN_SCREEN_REGION_MAP:
    case PN_SCREEN_FLY_MAP:
        region_map_wait(w);
        break;
    case PN_SCREEN_TM_CASE:
        if (pokedude_menu_wait(w, kind))
            break; /* v3.2: the Pokedude's TM case */
        if (gMain.inBattle && !battle_owns(SYM(CompleteWhenChoseItem_player), SYM(CompleteWhenChoseItem_oak)))
            break;
        tm_case_wait(w);
        break;
    case PN_SCREEN_BERRY_POUCH:
        if (gMain.inBattle && !battle_owns(SYM(CompleteWhenChoseItem_player), SYM(CompleteWhenChoseItem_oak)))
            break;
        berry_pouch_wait(w);
        break;
    case PN_SCREEN_MOVE_RELEARNER:
        relearner_wait(w);
        break;
    case PN_SCREEN_HALL_OF_FAME:
        hof_wait(w);
        break;
    case PN_SCREEN_EVOLUTION:
    case PN_SCREEN_EGG_HATCH:
    case PN_SCREEN_TRADE:
    case PN_SCREEN_PARTY_ITEM_ANIM:
        evolution_wait(w);
        if (!w->waiting && kind == PN_SCREEN_PARTY_ITEM_ANIM)
            item_anim_wait(w); /* v3.2 */
        break;
    case PN_SCREEN_BATTLE:
        if (pokedude_battle_wait(w))
            break; /* v3.2 */
        battle_wait(w);
        break;
    case PN_SCREEN_BATTLE_RECORDS:
        battle_records_wait(w); /* v3.2 */
        break;
    case PN_SCREEN_TEACHY_TV:
        teachy_tv_wait(w); /* v3.2 */
        break;
    default:
        break;
    }
    if (kind == PN_SCREEN_MART && !CB2_IS(CB2_Overworld))
        shop_buy_wait(w);
}
