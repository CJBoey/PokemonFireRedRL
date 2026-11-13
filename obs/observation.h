/* fireredsim: structured state observation (docs/OBSERVATION.md, contract layout v7).
 *
 * C ABI shared by the C observer (obs/src, compiled with the game's headers and
 * flags), the C++ worker and Python (python/fireredsim/obs_dtype.py is GENERATED
 * from this file by tools/gen_obs_dtype.py; the build checks that C and NumPy agree).
 *
 * Layout rules: packed, explicit widths, no pointers, little-endian, every struct
 * size static_asserted. Field names ending in "_priv" are PRIVILEGED: they are zero
 * in PN_OBS_VISIBLE mode (rule 1). Everything else is visible (what a player can see
 * on screen now or reach through the game's own menus).
 *
 * Keep the syntax of the struct definitions simple (one field declaration per
 * line, fixed-width types, constant array sizes from the #defines below, nested
 * structs by typedef name): tools/gen_obs_dtype.py parses them.
 */
#ifndef FIREREDSIM_OBSERVATION_H
#define FIREREDSIM_OBSERVATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define PN_OBS_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define PN_OBS_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif

#define PN_OBS_MAGIC 0x424F4E50u /* bytes 'P','N','O','B' */
#define PN_OBS_VERSION 10

/* pn_observe() modes */
#define PN_OBS_VISIBLE 0
#define PN_OBS_PRIVILEGED 1

/* array sizes (the game's constants, checked against the game headers in obs/src) */
#define PN_OBS_PARTY_SIZE 6
#define PN_OBS_NUM_OBJECTS 16
#define PN_OBS_DIALOG_LEN 256
#define PN_OBS_NAME_LEN 8          /* PLAYER_NAME_LENGTH + 1 */
#define PN_OBS_NICKNAME_LEN 11     /* POKEMON_NAME_LENGTH + 1 */
#define PN_OBS_TRAINER_NAME_LEN 13 /* struct Trainer.trainerName[12] + EOS */
#define PN_OBS_BAG_ITEMS 42
#define PN_OBS_BAG_KEY_ITEMS 30
#define PN_OBS_BAG_BALLS 13
#define PN_OBS_BAG_TMHM 58
#define PN_OBS_BAG_BERRIES 43
#define PN_OBS_PC_ITEMS 30
#define PN_OBS_NUM_BATTLERS 4
#define PN_OBS_FLAG_BYTES 288      /* NUM_FLAG_BYTES 0x120 */
#define PN_OBS_NUM_VARS 256        /* VARS_COUNT 0x100 */
#define PN_OBS_NUM_GAME_STATS 52   /* NUM_USED_GAME_STATS */
#define PN_OBS_TILE_ROWS 11        /* tile window (v3): map rows py-5..py+5 */
#define PN_OBS_TILE_COLS 15        /* map columns px-7..px+7 */
#define PN_OBS_TILES 165           /* PN_OBS_TILE_ROWS * PN_OBS_TILE_COLS, row-major */
#define PN_OBS_MENU_ACTIONS 8      /* v4: entries of a context menu (party_menu.c actions[8]) */
#define PN_OBS_BOXES 14            /* TOTAL_BOXES_COUNT */
#define PN_OBS_BOX_SLOTS 30        /* IN_BOX_COUNT */

/* ---- enums (values are part of the contract; python names come from here) ---- */

/* PnObs.screen (docs/OBS_SOURCES.md section 3.1) */
enum PnScreen {
    PN_SCREEN_UNKNOWN = 0,
    PN_SCREEN_BOOT,
    PN_SCREEN_INTRO,
    PN_SCREEN_TITLE,
    PN_SCREEN_MAIN_MENU,
    PN_SCREEN_NEW_GAME_SPEECH,
    PN_SCREEN_NAMING,
    PN_SCREEN_OVERWORLD,
    PN_SCREEN_MAP_TRANSITION,
    PN_SCREEN_BATTLE_TRANSITION,
    PN_SCREEN_QUEST_LOG,
    PN_SCREEN_START_MENU,
    PN_SCREEN_SAVE_MENU,
    PN_SCREEN_BAG,
    PN_SCREEN_TM_CASE,
    PN_SCREEN_BERRY_POUCH,
    PN_SCREEN_PARTY_MENU,
    PN_SCREEN_PARTY_ITEM_ANIM,
    PN_SCREEN_SUMMARY,
    PN_SCREEN_POKEDEX,
    PN_SCREEN_TRAINER_CARD,
    PN_SCREEN_OPTIONS,
    PN_SCREEN_PC_MENU,
    PN_SCREEN_PC_BOX,
    PN_SCREEN_ITEM_PC,
    PN_SCREEN_MART,
    PN_SCREEN_REGION_MAP,
    PN_SCREEN_FLY_MAP,
    PN_SCREEN_MOVE_RELEARNER,
    PN_SCREEN_EGG_HATCH,
    PN_SCREEN_TRADE,
    PN_SCREEN_MAIL,
    PN_SCREEN_EASY_CHAT,
    PN_SCREEN_BATTLE,
    PN_SCREEN_EVOLUTION,
    PN_SCREEN_HALL_OF_FAME,
    PN_SCREEN_CREDITS,
    PN_SCREEN_SAVE_FAILED,
    PN_SCREEN_HELP_SYSTEM,
    PN_SCREEN_BATTLE_RECORDS,
    PN_SCREEN_SLOT_MACHINE,
    PN_SCREEN_DIPLOMA,
    PN_SCREEN_TEACHY_TV,
    PN_SCREEN_FAME_CHECKER,
    PN_SCREEN_CUTSCENE,
    PN_SCREEN_MYSTERY_GIFT,
    PN_SCREEN_BERRY_FIX,
    PN_SCREEN_CLEAR_SAVE,
    PN_SCREEN_LINK,
    PN_SCREEN_COUNT
};

/* PnObs.input_kind (docs/OBS_SOURCES.md section 2.1) */
enum PnInputKind {
    PN_INPUT_NONE = 0,
    PN_INPUT_DIALOG_ADVANCE,   /* A/B (or A only: see input_keys) acknowledges a message */
    PN_INPUT_YESNO,            /* UP/DOWN, A (choose), B (= NO) */
    PN_INPUT_MENU,             /* list/menu cursor, A, B and screen-specific extras */
    PN_INPUT_QUANTITY,         /* UP/DOWN +-1, LEFT/RIGHT +-10, A, B */
    PN_INPUT_OVERWORLD_FREE,   /* D-pad, A, START, SELECT (B held = run) */
    PN_INPUT_NAMING,           /* naming keyboard */
    PN_INPUT_BATTLE_ACTION,    /* FIGHT/BAG/POKEMON/RUN */
    PN_INPUT_BATTLE_MOVE,      /* move menu */
    PN_INPUT_BATTLE_TARGET,    /* doubles target selection */
    PN_INPUT_BATTLE_MOVE_SWAP, /* move reordering (SELECT in the move menu) */
    PN_INPUT_BATTLE_PARTY,     /* party menu owned by a battle controller */
    PN_INPUT_BATTLE_BAG,       /* bag owned by a battle controller */
    PN_INPUT_QUEST_LOG,        /* Quest Log playback: A next scene, B skip */
    PN_INPUT_MAP_CURSOR,       /* region map / storage box cursor */
    PN_INPUT_BOX_CHOOSE,       /* storage box chooser */
    PN_INPUT_TITLE_START,      /* title screen: A or START */
    PN_INPUT_HELP,             /* help overlay */
    PN_INPUT_HOLD_A,           /* A must be HELD (save error) */
    PN_INPUT_ANY_KEY,          /* any newly pressed key */
    PN_INPUT_COUNT
};

/* PnObs.input_flags */
#define PN_INPUT_FLAG_LR_SWALLOWED 0x01         /* HELP button mode + help enabled: a new L/R opens the
                                                   help overlay and swallows the whole frame */
#define PN_INPUT_FLAG_INPUT_SENSITIVE_ANIM 0x02 /* an animation that a held key cancels (evolution B) */
#define PN_INPUT_FLAG_HOLD_A 0x04
#define PN_INPUT_FLAG_DANGEROUS_CHORD 0x08      /* title RUN: B+SELECT(+UP) held clears the save / Berry Fix */

/* PnObs.valid: which gates of docs/OBS_SOURCES.md section 1 hold */
#define PN_VALID_INGAME 0x01 /* progress/party/bag belong to the running game (after CB2_NewGame/continue) */
#define PN_VALID_FIELD 0x02  /* callback2 == CB2_Overworld outside Quest Log playback */
#define PN_VALID_BATTLE 0x04 /* battle data valid (BATTLE gate) */
#define PN_VALID_BATTLE_MONS 0x08 /* gBattleMons copied from the controllers */

/* PnObs.field_state */
enum PnFieldState {
    PN_FIELD_NONE = 0,        /* not on the field */
    PN_FIELD_FREE,            /* field, no warp/battle pending */
    PN_FIELD_WARP,            /* warp window (map loads, door/stair/teleport tasks) */
    PN_FIELD_QUEST_LOG,       /* Quest Log playback */
    PN_FIELD_BATTLE_PENDING,  /* Task_BattleStart / battle transition */
};

/* PnObs.player_motion */
enum PnPlayerMotion {
    PN_MOTION_IDLE = 0,
    PN_MOTION_STEP,           /* walking/surfing/biking step in progress */
    PN_MOTION_RUN,
    PN_MOTION_LEDGE,
    PN_MOTION_BUMP,
    PN_MOTION_TURN,
    PN_MOTION_OTHER,          /* other held/scripted movement */
};

/* PnObs.menu_open */
enum PnMenuOpen {
    PN_MENU_NONE = 0,
    PN_MENU_START,
    PN_MENU_SAVE_PROMPT,
    PN_MENU_SCRIPT_YESNO,
    PN_MENU_MULTICHOICE,
    PN_MENU_MULTICHOICE_GRID,
    PN_MENU_SCRIPT_LIST,
    PN_MENU_MART,
    PN_MENU_PC,
    PN_MENU_HELP,
    PN_MENU_FULLSCREEN,       /* see PnObs.screen */
};

/* PnObs.dialog_state */
enum PnDialogState {
    PN_DIALOG_NONE = 0,
    PN_DIALOG_PRINTING,       /* text being typed (A/B only speed it up) */
    PN_DIALOG_PROMPT,         /* printer waits for A/B inside the text (\p, \l, PAUSE_UNTIL_PRESS) */
    PN_DIALOG_WAIT_BUTTON,    /* text finished, script waitbuttonpress */
    PN_DIALOG_AUTO,           /* auto-scroll (Quest Log, link/Pokedude battles) */
    PN_DIALOG_SHOWN,          /* finished text still on screen */
};

/* PnMon.status / PnBattler.status (visible ailment, GetMonAilment order) */
enum PnAilment {
    PN_AILMENT_NONE = 0,
    PN_AILMENT_PSN,
    PN_AILMENT_PRZ,
    PN_AILMENT_SLP,
    PN_AILMENT_FRZ,
    PN_AILMENT_BRN,
    PN_AILMENT_PKRS,
    PN_AILMENT_FNT,
};

/* PnObs.battle_phase */
enum PnBattlePhase {
    PN_BATTLE_NONE = 0,
    PN_BATTLE_PENDING,        /* transition on the field (battle data already set for the next battle) */
    PN_BATTLE_INTRO,          /* battle callbacks installed, intro running */
    PN_BATTLE_ACTIVE,         /* BATTLE gate holds */
    PN_BATTLE_ENDING,         /* outcome decided / exit started */
    PN_BATTLE_EVOLUTION,      /* post-battle evolution scene */
};

/* PnObs.wait_ctx: the docs/OBS_SOURCES.md section 2 predicate that produced waiting_for_input */
enum PnWaitCtx {
    PN_WAIT_NONE = 0,
    PN_WAIT_TITLE,            /* TTL */
    PN_WAIT_MAIN_MENU,        /* MM */
    PN_WAIT_OAK_TEXT,         /* OAK page text (printer 0 prompt) */
    PN_WAIT_OAK_CONTROLS,     /* OAK controls guide */
    PN_WAIT_OAK_PIKACHU,      /* OAK Pikachu intro */
    PN_WAIT_OAK_GENDER,       /* OAK gender menu */
    PN_WAIT_OAK_NAME_CHOICE,  /* OAK rival/player name list */
    PN_WAIT_OAK_CONFIRM,      /* OAK name confirmation YES/NO */
    PN_WAIT_NAMING,           /* NAM-KB */
    PN_WAIT_NAMING_PCMSG,     /* NAM-PCMSG */
    PN_WAIT_OW_FREE,          /* OW free movement */
    PN_WAIT_DLG_TEXT,         /* DLG-TXT */
    PN_WAIT_DLG_BUTTON,       /* DLG-BTN */
    PN_WAIT_SCRIPT_YESNO,     /* SYN */
    PN_WAIT_MULTICHOICE,      /* SMC (incl. PC-SOMEONE) */
    PN_WAIT_MULTICHOICE_GRID, /* SMG */
    PN_WAIT_SCRIPT_LIST,      /* SLM */
    PN_WAIT_START_MENU,       /* start menu */
    PN_WAIT_SAVE_YESNO,       /* save prompt YES/NO */
    PN_WAIT_SAVE_TEXT,        /* save dialog message prompt */
    PN_WAIT_FIELD_MSG_CONT,   /* MSG-CONT on the field (item use message) */
    PN_WAIT_SHOP_MENU,        /* MRT-FIELD */
    PN_WAIT_SHOP_BUY,         /* MRT-BUY */
    PN_WAIT_SHOP_QTY,         /* MRT-QTY */
    PN_WAIT_SHOP_YESNO,       /* MRT-YN */
    PN_WAIT_SHOP_ACK,         /* MRT-ACK */
    PN_WAIT_SHOP_MSG_CONT,    /* MSG-CONT in the buy menu */
    PN_WAIT_PC_STORAGE_TOP,   /* PC-STORAGE-TOP */
    PN_WAIT_PC_PLAYER,        /* PC-PLAYER */
    PN_WAIT_BAG_LIST,         /* BAG-LIST */
    PN_WAIT_BAG_MOVE,         /* BAG-MOVE */
    PN_WAIT_BAG_CONTEXT,      /* BAG-CTX */
    PN_WAIT_BAG_QUANTITY,     /* BAG-QTY */
    PN_WAIT_BAG_YESNO,        /* BAG-YN */
    PN_WAIT_BAG_MSG_CONT,     /* MSG-CONT in the bag */
    PN_WAIT_BAG_ACK,          /* BAG-ACK */
    PN_WAIT_PARTY_CHOOSE,     /* PTY-CHOOSE */
    PN_WAIT_PARTY_SUBMENU,    /* PTY-SUB */
    PN_WAIT_PARTY_YESNO,      /* PTY-YN */
    PN_WAIT_PARTY_MOVE,       /* PTY-MOVE */
    PN_WAIT_PARTY_ACK,        /* PTY-ACK */
    PN_WAIT_PARTY_MSG,        /* PTY-MSG */
    PN_WAIT_SUMMARY,          /* SUM-INFO */
    PN_WAIT_SUMMARY_MOVES,    /* SUM-MOVES */
    PN_WAIT_SUMMARY_FORGET,   /* SUM-FORGET */
    PN_WAIT_OPTIONS,          /* OPT */
    PN_WAIT_TRAINER_CARD,     /* TCD */
    PN_WAIT_BATTLE_ACTION,    /* BTL-ACT */
    PN_WAIT_BATTLE_MOVE,      /* BTL-MOVE */
    PN_WAIT_BATTLE_TARGET,    /* BTL-TARGET */
    PN_WAIT_BATTLE_SWAP,      /* BTL-SWAP */
    PN_WAIT_BATTLE_MSG,       /* BTL-MSG */
    PN_WAIT_BATTLE_YESNO,     /* BTL-YN */
    PN_WAIT_BATTLE_LEARN,     /* BTL-LEARN */
    PN_WAIT_BATTLE_NICK,      /* BTL-NICK */
    PN_WAIT_BATTLE_LVLUP,     /* BTL-LVL */
    PN_WAIT_EVO_YESNO,        /* EVO-YN */
    PN_WAIT_SCENE_MSG,        /* EVO-MSG / HATCH-MSG / TRD-MSG / PSA-MSG */
    PN_WAIT_HELP,             /* HLP */
    PN_WAIT_QUEST_LOG,        /* QL scene */
    PN_WAIT_OW_ANIM,          /* field: turn / bump animation (T_NOT_MOVING): A, START, SELECT (+D-pad that interrupts a bump) */
    PN_WAIT_ITEM_PC,          /* PC-ITEM: item storage list, submenu, move mode, quantity, acknowledgements */
    /* ---- v2 (values above are unchanged from v1) ---- */
    PN_WAIT_DEX_MENU,         /* DEX: Pokedex top menu (mode / habitat list) */
    PN_WAIT_DEX_LIST,         /* DEX: numerical / A-Z / type / weight / height list */
    PN_WAIT_DEX_CATEGORY,     /* DEX: habitat page (mons of a category) */
    PN_WAIT_DEX_PAGE,         /* DEX: a mon's entry page or area page */
    PN_WAIT_DEX_REGISTER,     /* DEX: the new entry shown after a catch (in battle) */
    PN_WAIT_PC_BOX,           /* PC-BOX: storage box main cursor (box, party, title, buttons) */
    PN_WAIT_PC_BOX_MENU,      /* PC-BOX-SUB M: selected-mon menu, box options, wallpapers, markings */
    PN_WAIT_PC_BOX_YESNO,     /* PC-BOX-SUB Y: release / exit / close-while-holding YES/NO */
    PN_WAIT_PC_BOX_CHOOSE,    /* PC-BOX-SUB B: deposit / jump box chooser */
    PN_WAIT_PC_BOX_MSG,       /* PC-BOX-SUB X: message acknowledged by A, B or the D-pad */
    PN_WAIT_REGION_MAP,       /* MAP: Town Map / wall map cursor */
    PN_WAIT_FLY_MAP,          /* MAP: Fly map cursor */
    PN_WAIT_TM_CASE,          /* TMC: TM list */
    PN_WAIT_TM_CASE_MENU,     /* TMC: context menu (USE / GIVE / EXIT) */
    PN_WAIT_TM_CASE_QTY,      /* TMC: sell quantity */
    PN_WAIT_TM_CASE_YESNO,    /* TMC: sell YES/NO */
    PN_WAIT_TM_CASE_MSG,      /* TMC: message prompt / acknowledgement */
    PN_WAIT_BERRY_POUCH,      /* BRP: berry list */
    PN_WAIT_BERRY_POUCH_MENU, /* BRP: context menu */
    PN_WAIT_BERRY_POUCH_QTY,  /* BRP: toss / sell quantity */
    PN_WAIT_BERRY_POUCH_YESNO, /* BRP: toss / sell YES/NO */
    PN_WAIT_BERRY_POUCH_MSG,  /* BRP: message prompt / acknowledgement */
    PN_WAIT_RELEARNER,        /* MRL: move relearner list */
    PN_WAIT_RELEARNER_YESNO,  /* MRL: teach / give up / forget / stop YES/NO */
    PN_WAIT_RELEARNER_MSG,    /* MRL: text prompt or the A wait after learning */
    PN_WAIT_MAIN_MENU_MSG,    /* MM: save error / Mystery Gift error message */
    PN_WAIT_HALL_OF_FAME,     /* HOF: end of the ceremony (A) / Hall of Fame PC viewer */
    /* ---- v3.1 (values above are unchanged) ---- */
    PN_WAIT_WHITEOUT_MSG,     /* field: "scurried to a POKEMON CENTER / back home" after a whiteout (A, B) */
    /* ---- v3.2 (values above are unchanged) ---- */
    PN_WAIT_BATTLE_RECORDS,   /* Cable Club / Trainer Tower BATTLE RECORDS screen: A or B closes it */
    PN_WAIT_TEACHY_TV_MENU,   /* Teachy TV programme list (A, B, cursor keys, SELECT if not from the start menu) */
    PN_WAIT_TEACHY_TV,        /* Teachy TV programme: B ends it (+A on its messages) */
    PN_WAIT_POKEDUDE,         /* Pokedude demo battle / bag / party menu / TM case: B quits (+A on its messages) */
    PN_WAIT_ITEM_ANIM,        /* field item-use animation (PSA), its last state: A or B closes it */
    PN_WAIT_COUNT
};

/* PnObs.tiles[] (v3): one class per tile of the 15x11 window around the player
 * (docs/OBSERVATION.md section "Tile map"). An object's class replaces the terrain's. */
enum PnTileClass {
    PN_TILE_OUTSIDE = 0,      /* beyond the loaded grid / border block; every cell off the field */
    PN_TILE_WALKABLE,         /* no collision, elevation compatible */
    PN_TILE_IMPASSABLE,       /* collision bit or elevation mismatch (walls, counters, signs, PCs) */
    PN_TILE_WILD,             /* walkable, land encounter tile (tall grass, cave floors) */
    PN_TILE_WATER,            /* surfable water */
    PN_TILE_WATERFALL,
    PN_TILE_LEDGE_S,          /* jump ledges: passable only moving in that direction */
    PN_TILE_LEDGE_W,
    PN_TILE_LEDGE_E,
    PN_TILE_LEDGE_N,
    PN_TILE_WARP,             /* a warp event of the current map on a tile the game warps from (door, stairs,
                                 ladder, arrow mat, cave mouth, hole, escalator, warp pad) */
    PN_TILE_FORCED_E,         /* forced movement: spin / walk / slide tiles, currents */
    PN_TILE_FORCED_W,
    PN_TILE_FORCED_N,
    PN_TILE_FORCED_S,
    PN_TILE_ICE,
    PN_TILE_NPC,              /* person sprite */
    PN_TILE_ITEM_BALL,
    PN_TILE_CUT_TREE,
    PN_TILE_BOULDER,
    PN_TILE_ROCK,             /* Rock Smash rock */
    PN_TILE_POKEMON,          /* Pokemon sprite on the field */
    PN_TILE_OBJECT_OTHER,     /* other inanimate objects (signs, fossils, dolls, ships) */
    PN_TILE_COUNT
};

/* PnObs.menu_action, menu_actions[] (v4): what the entry under the cursor of the open menu does.
 * One vocabulary for the bag context menu, the party menu's action submenu, the item PC
 * submenu, the player's PC menus, the Pokemon Storage System's menus (Bill's PC main menu,
 * selected-mon / item / box-options menus, the RELEASE YES/NO) and the "Someone's PC"
 * multichoice (docs/OBSERVATION.md "Menu actions"). 0 = no menu of that kind is open. */
enum PnMenuAction {
    PN_ACTION_NONE = 0,
    PN_ACTION_CANCEL,          /* CANCEL, EXIT, SEE YA, LOG OFF, TURN OFF, the NO of a listed YES/NO */
    PN_ACTION_USE,             /* bag USE (field / battle), CHECK, OPEN, WALK */
    PN_ACTION_GIVE,            /* give an item to a mon (bag, item PC, party ITEM/GIVE, storage GIVE) */
    PN_ACTION_TOSS,
    PN_ACTION_REGISTER,        /* bag REGISTER / DESELECT (SELECT button), party REGISTER */
    PN_ACTION_WITHDRAW,        /* item PC WITHDRAW, player PC WITHDRAW ITEM, Bill's PC WITHDRAW, storage WITHDRAW */
    PN_ACTION_DEPOSIT,         /* player PC DEPOSIT ITEM, Bill's PC DEPOSIT, storage STORE, party STORE */
    PN_ACTION_MOVE_MONS,       /* Bill's PC MOVE POKEMON */
    PN_ACTION_MOVE_ITEMS,      /* Bill's PC MOVE ITEMS */
    PN_ACTION_MOVE,            /* storage MOVE (pick the mon up) */
    PN_ACTION_SHIFT,           /* storage SHIFT (swap with the mon in hand) */
    PN_ACTION_PLACE,           /* storage PLACE (put the mon in hand down) */
    PN_ACTION_SUMMARY,
    PN_ACTION_SWITCH,          /* party SWITCH, storage item SWITCH */
    PN_ACTION_ITEM,            /* party ITEM (opens GIVE / TAKE) */
    PN_ACTION_TAKE,            /* party TAKE ITEM / TAKE MAIL, storage TAKE */
    PN_ACTION_MAIL,            /* party MAIL (opens READ / TAKE) */
    PN_ACTION_READ,
    PN_ACTION_SEND_OUT,        /* battle party SEND OUT / SHIFT */
    PN_ACTION_ENTER,           /* multi-choose ENTER */
    PN_ACTION_NO_ENTRY,
    PN_ACTION_TRADE,
    PN_ACTION_RELEASE,         /* storage RELEASE (and the YES of its YES/NO) */
    PN_ACTION_MARK,
    PN_ACTION_JUMP,            /* box options JUMP */
    PN_ACTION_WALLPAPER,       /* box options WALLPAPER */
    PN_ACTION_NAME,            /* box options NAME */
    PN_ACTION_BAG,             /* storage item: to the BAG */
    PN_ACTION_INFO,            /* storage item: INFO */
    PN_ACTION_WALLPAPER_SET,   /* a wallpaper theme / wallpaper entry */
    PN_ACTION_ITEM_STORAGE,    /* player PC: ITEM STORAGE */
    PN_ACTION_MAILBOX,         /* player PC: MAILBOX */
    PN_ACTION_PC_STORAGE,      /* PC menu: SOMEONE'S / BILL'S PC (the Pokemon Storage System) */
    PN_ACTION_PC_PLAYER,       /* PC menu: <player>'s PC */
    PN_ACTION_PC_OAK,          /* PC menu: PROF. OAK'S PC */
    PN_ACTION_PC_HALL_OF_FAME, /* PC menu: HALL OF FAME */
    PN_ACTION_FLASH,           /* party field moves (party_menu.c sFieldMoves order) */
    PN_ACTION_CUT,
    PN_ACTION_FLY,
    PN_ACTION_STRENGTH,
    PN_ACTION_SURF,
    PN_ACTION_ROCK_SMASH,
    PN_ACTION_WATERFALL,
    PN_ACTION_TELEPORT,
    PN_ACTION_DIG,
    PN_ACTION_MILK_DRINK,
    PN_ACTION_SOFT_BOILED,
    PN_ACTION_SWEET_SCENT,
    PN_ACTION_OTHER,           /* a menu entry outside this vocabulary */
    PN_ACTION_COUNT
};

/* PnObs.box_option (v4): the Pokemon Storage System's mode while its box screen is open */
enum PnBoxOption {
    PN_BOXOPT_NONE = 0,       /* box screen closed */
    PN_BOXOPT_WITHDRAW,
    PN_BOXOPT_DEPOSIT,
    PN_BOXOPT_MOVE_MONS,
    PN_BOXOPT_MOVE_ITEMS,
};

/* PnObs.skill_events_priv (v5, u32): bit PN_SKILL_x = the skill-retention event x of reward v0.11
 * (rl/pokerl/reward.py SKILL_EVENTS, same order; the "attempt" definitions), detected by the worker against
 * the env's previous record (obs/skill_events.h; docs/OBSERVATION.md "Changes in v5"). Bits
 * PN_SKILL_SUCCESS_SHIFT + x (16..28) are reserved for "actually succeeded" variants of the same events
 * and are 0 in v5; bits 29-31 are reserved, 0. pn_observe always writes 0. Bit 12 (PN_SKILL_MART,
 * reward v0.12, 2026-10-20), bit 13 (PN_SKILL_USE_CONSUMABLE, 2026-10-28), bit 14 (PN_SKILL_USE_BATTLE_ITEM,
 * 2026-10-29) and bit 15 (PN_SKILL_MART_BUY, 2026-10-29) were added without a layout or version change: a worker
 * reports how many event bits it sets in its --info ("skill_events"; an observer v5 worker without the key: 12).
 * Bits 13 and 14 are the only events not
 * derived from records: the game reports the item it removed from the bag because the player used it, and
 * whether gMain.inBattle was set (gba/portable_abi.h PORT_NOTE_ITEM_USE, taken by the worker after the STEP's
 * frames), so a refused use (nothing removed) is not an event. */
#define PN_SKILL_SUCCESS_SHIFT 16
enum PnSkillEvent {
    PN_SKILL_PARTY_MENU = 0,   /* the screen becomes PARTY_MENU, outside battle */
    PN_SKILL_BAG_MENU,         /* the screen becomes BAG, outside battle */
    PN_SKILL_PARTY_ORDER,      /* the same party mons in a different order (outside battle / the box screen) */
    PN_SKILL_USE_ITEM,         /* USE on an Items-pocket item (bag) or a berry (berry pouch) taken */
    PN_SKILL_USE_KEY_ITEM,     /* USE on a Key Items-pocket item taken */
    PN_SKILL_USE_TM,           /* a party mon gains the move of the TM/HM last selected for it */
    PN_SKILL_PC_STORAGE,       /* Bill's PC main menu waits, once per PC session */
    PN_SKILL_PC_PLAYER,        /* the player's PC menu waits, once per PC session */
    PN_SKILL_PC_PARTY,         /* the set of party mons changes on the PC box screen, no mon in hand */
    PN_SKILL_TREE,             /* A on a Cut tree (free on the field facing it, then a dialog) */
    PN_SKILL_BOULDER,          /* A on a Strength boulder */
    PN_SKILL_ROCK,             /* A on a Rock Smash rock */
    PN_SKILL_MART,             /* the Poke Mart BUY / SELL / QUIT menu waits, once per shop session (v0.12) */
    PN_SKILL_USE_CONSUMABLE,   /* a consumable the player used on the field was removed from the bag during the
                                  step (Repel, Escape Rope, medicine, berries, PP restorers, Rare Candy, Sacred
                                  Ash, evolution stones; not TMs, key items, flutes, refused uses, battle uses):
                                  the game's PORT_NOTE_ITEM_USE latch, a use outside battle (2026-10-28) */
    PN_SKILL_USE_BATTLE_ITEM,  /* a consumable the player used in battle was removed from the bag during the step
                                  (medicine, status heals, revives, berries, Ether / Elixir, X items, Guard Spec,
                                  Dire Hit, Poke Doll / Fluffy Tail; not Poke Balls): the same latch, a use while
                                  gMain.inBattle (2026-10-29) */
    PN_SKILL_MART_BUY,         /* a purchase completed: the mart's "Here you go, thank you" acknowledgement
                                  (SHOP_ACK, Task_ReturnToItemListAfterItemPurchase, reached only after the item
                                  went into the bag and the money was paid) starts waiting (2026-10-29) */
    PN_SKILL_COUNT
};

/* PnObs.interact_kind_priv (v6): the overworld interaction the game started during the frames of the STEP that
 * produced the record (reward v0.14; docs/OBSERVATION.md "Changes in v6"). Reported by a PORTABLE hook in
 * game/src/field_control_avatar.c (PORT_NOTE_INTERACTION: plain stores into the host-side gPortInteractionLatch) at the moment the interaction's script is chosen,
 * so it is exact: no inference from the screen. Only object events and background (sign) events: metatile
 * scripts (bookshelves, PCs on tiles, signs drawn into the tileset, ...) and water (Surf / Waterfall prompts)
 * are not reported. Nothing is reported during quest-log playback. */
enum PnInteractKind {
    PN_INTERACT_NONE = 0,
    PN_INTERACT_OBJECT,        /* A on an object event (NPC, trainer, item ball, Cut tree, boulder, ...; also
                                  across a counter): key = (map, local id) of the object itself */
    PN_INTERACT_SIGN,          /* A on a background event of sign / script type: key = (map, x, y) */
    PN_INTERACT_HIDDEN_ITEM,   /* A on a hidden item not yet found: key = (map, x, y) */
    PN_INTERACT_SIGN_WALK,     /* a scripted sign event read by walking into it (TrySetUpWalkIntoSignpostScript,
                                  SIGNPOST_SCRIPTED only): the same key as PN_INTERACT_SIGN */
    PN_INTERACT_COUNT
};

/* GBA key bits (KEYINPUT layout, active high) used by input_keys / held_keys_prev */
#define PN_KEY_A 0x001
#define PN_KEY_B 0x002
#define PN_KEY_SELECT 0x004
#define PN_KEY_START 0x008
#define PN_KEY_RIGHT 0x010
#define PN_KEY_LEFT 0x020
#define PN_KEY_UP 0x040
#define PN_KEY_DOWN 0x080
#define PN_KEY_R 0x100
#define PN_KEY_L 0x200
#define PN_KEY_DPAD 0x0F0

/* ---- structs ---------------------------------------------------------------- */
#pragma pack(push, 1)

/* party / enemy party slot (GetMonData on a local copy) */
typedef struct PnMon {
    uint16_t species;            /* visible species: SPECIES_EGG (412) for eggs, 0 = empty slot */
    uint16_t held_item;
    uint16_t hp;
    uint16_t max_hp;
    uint16_t moves[4];
    uint16_t attack;
    uint16_t defense;
    uint16_t speed;
    uint16_t sp_attack;
    uint16_t sp_defense;
    uint16_t ot_id_lo;           /* trainer ID shown on the summary (low 16 bits) */
    uint32_t exp;
    uint8_t pp[4];
    uint8_t max_pp[4];
    uint8_t level;
    uint8_t status;              /* enum PnAilment */
    uint8_t nature;
    uint8_t ability;
    uint8_t gender;              /* 0 male, 0xFE female, 0xFF genderless */
    uint8_t is_egg;
    uint8_t type1;
    uint8_t type2;
    uint8_t is_shiny;
    uint8_t pokerus_state;       /* 0 none, 1 infected, 2 cured */
    uint8_t nickname[11];        /* game encoding, 0xFF terminated and padded */
    uint8_t ot_name[8];
    uint8_t _pad0[3];
    uint16_t species_priv;       /* real species (eggs) */
    uint8_t friendship_priv;     /* eggs: remaining egg cycles */
    uint8_t pokerus_priv;
    uint8_t iv_priv[6];          /* HP, Atk, Def, Spe, SpA, SpD */
    uint8_t ev_priv[6];
    uint32_t pid_priv;
    uint32_t ot_id_priv;
    uint32_t status_raw_priv;    /* STATUS1 as stored (sleep turns, toxic counter) */
    uint8_t pp_bonuses_priv;
    uint8_t _pad1[3];
} PnMon;
PN_OBS_STATIC_ASSERT(sizeof(PnMon) == 104, "PnMon layout");

/* one battler (gBattleMons[b]); own side fully visible, opponent as displayed */
typedef struct PnBattler {
    uint16_t species;            /* 0 while not shown; masked in ghost battles */
    uint16_t hp;                 /* own side only (opponent: 0, see hp_bar_*) */
    uint16_t max_hp;             /* own side only */
    uint16_t item;               /* own side only (opponent: privileged, item_priv) */
    uint16_t moves[4];           /* own side: the moves in slot order; opponent: the moves REVEALED in this
                                    battle (PP below the maximum <=> used, see docs/OBSERVATION.md),
                                    ascending move ids, 0 padded; a transformed opponent: the copied moves */
    uint16_t attack;             /* own side only (in-battle stats) */
    uint16_t defense;
    uint16_t speed;
    uint16_t sp_attack;
    uint16_t sp_defense;
    uint8_t pp[4];               /* own side only */
    uint8_t level;
    uint8_t gender;              /* 0 male, 0xFE female, 0xFF genderless/not drawn */
    uint8_t status;              /* enum PnAilment (icon category) */
    uint8_t is_player_side;
    uint8_t is_active;           /* healthbox shown */
    uint8_t ability;             /* own side: gBattleMons; opponent: the species' only possible ability
                                    (static data), 0 when the species has two (privileged: ability_priv) */
    uint8_t type1;
    uint8_t type2;
    uint8_t hp_bar_pixels;       /* 0..48 as drawn */
    uint8_t hp_bar_color;        /* 0 not shown, 1 green, 2 yellow, 3 red */
    uint8_t hp_bar_fraction;     /* hp_bar_pixels * 255 / 48 */
    uint8_t party_slot;          /* v10: own side: the slot of party[] (as reported in THIS record) holding this battler's
                                    mon, matched by personality and OT id (the battle party menu reorders the array,
                                    so gBattlerPartyIndexes does not index it there); 255 for an opponent, an absent
                                    battler (a fainted mon awaiting its replacement) and the Safari Zone's own side
                                    (no mon out). Visible: the player sees which mon is out */
    uint16_t last_move;          /* the move this battler last announced ("X used Y!", gLastPrintedMoves:
                                    set when that move ends, 0 after a switch-in) */
    int8_t stat_stages[8];       /* stage - 6: HP(unused), ATK, DEF, SPD, SPATK, SPDEF, ACC, EVA */
    uint32_t status2;            /* volatile status presence bits (turn counters collapsed) */
    uint16_t species_priv;
    uint16_t hp_priv;
    uint16_t max_hp_priv;
    uint16_t item_priv;
    uint16_t moves_priv[4];
    uint16_t attack_priv;
    uint16_t defense_priv;
    uint16_t speed_priv;
    uint16_t sp_attack_priv;
    uint16_t sp_defense_priv;
    uint8_t pp_priv[4];
    uint8_t ability_priv;
    uint8_t party_index_priv;    /* gBattlerPartyIndexes */
    uint32_t status1_priv;
    uint32_t status2_priv;
    uint32_t status3_priv;
    uint32_t personality_priv;
} PnBattler;
PN_OBS_STATIC_ASSERT(sizeof(PnBattler) == 104, "PnBattler layout");

/* one NPC / object event on the map (player and camera object excluded) */
typedef struct PnObjectEvent {
    int16_t x;                   /* map coordinates (currentCoords - 7) */
    int16_t y;
    uint8_t graphics_id;
    uint8_t facing;              /* 1 S, 2 N, 3 W, 4 E */
    uint8_t visible;
    uint8_t moving;              /* held/single movement in progress */
    uint8_t local_id_priv;
    uint8_t map_num_priv;
    uint8_t map_group_priv;
    uint8_t movement_type_priv;
    uint8_t trainer_type_priv;
    uint8_t trainer_range_priv;
    uint8_t _pad0[2];
} PnObjectEvent;
PN_OBS_STATIC_ASSERT(sizeof(PnObjectEvent) == 16, "PnObjectEvent layout");

typedef struct PnItem {
    uint16_t id;
    uint16_t qty;
} PnItem;
PN_OBS_STATIC_ASSERT(sizeof(PnItem) == 4, "PnItem layout");

/* v4: one slot of the current PC box (GetBoxMonData on a copy; empty slot: all 0) */
typedef struct PnBoxMon {
    uint16_t species;            /* visible species: SPECIES_EGG (412) for eggs, 0 = empty slot */
    uint16_t held_item;
    uint8_t level;               /* GetLevelFromBoxMonExp; 0 for eggs */
    uint8_t is_egg;
    uint8_t _pad0[2];            /* struct sizes stay multiples of 4 (the game build aligns every struct to 4) */
} PnBoxMon;
PN_OBS_STATIC_ASSERT(sizeof(PnBoxMon) == 8, "PnBoxMon layout");

typedef struct PnObs {
    /* ---- header ---- */
    uint32_t magic;              /* PN_OBS_MAGIC */
    uint16_t version;            /* PN_OBS_VERSION */
    uint16_t mode;               /* PN_OBS_VISIBLE / PN_OBS_PRIVILEGED */
    uint32_t size;               /* sizeof(PnObs) */
    uint32_t _pad0;
    uint64_t frame;              /* logical frames since power-on */
    uint8_t priv_valid;          /* 1 iff mode == PRIVILEGED (privileged fields filled) */
    uint8_t screen;              /* enum PnScreen */
    uint8_t waiting_for_input;   /* 1 = the next frame acts on a new press of a key in input_keys */
    uint8_t input_kind;          /* enum PnInputKind */
    uint16_t input_keys;         /* GBA key mask the wait context acts on (0 when not waiting) */
    uint16_t held_keys_prev;     /* gMain.heldKeysRaw: held at the boundary, cannot be a new press next frame */
    uint8_t input_flags;         /* PN_INPUT_FLAG_* */
    uint8_t wait_ctx;            /* enum PnWaitCtx */
    uint8_t soft_reset_happened; /* the last frame ended in a soft reset (env powered on again) */
    uint8_t valid;               /* PN_VALID_* */
    uint8_t in_battle;           /* BATTLE gate holds */
    uint8_t battle_phase;        /* enum PnBattlePhase */
    uint8_t field_state;         /* enum PnFieldState */
    uint8_t _pad1[1];

    /* ---- overworld (FIELD) ---- */
    uint8_t map_group;
    uint8_t map_num;
    uint8_t facing;              /* 1 S, 2 N, 3 W, 4 E */
    uint8_t player_flags;        /* gPlayerAvatar.flags without CONTROLLABLE */
    int16_t x;                   /* player tile (map coordinates) */
    int16_t y;
    int16_t camera_x;            /* gSaveBlock1Ptr->pos */
    int16_t camera_y;
    uint16_t map_layout_id;
    uint8_t weather;
    uint8_t flash_level;
    uint8_t mapsec;              /* region map section (decode names in Python) */
    int8_t floor;
    uint8_t player_motion;       /* enum PnPlayerMotion */
    uint8_t controls_locked;     /* ArePlayerFieldControlsLocked() */
    uint8_t in_dialog;
    uint8_t dialog_state;        /* enum PnDialogState */
    uint8_t dialog_font;         /* 2 normal, 4 male (blue), 5 female (red) */
    uint8_t menu_open;           /* enum PnMenuOpen */
    uint8_t menu_cursor;         /* cursor of the menu that owns the input (0 = first/YES) */
    uint8_t menu_count;          /* number of entries of that menu (0 unknown) */
    uint8_t menu_id;             /* multichoice id (255 none) */
    uint8_t start_menu_cursor;   /* option id under the cursor: 0 POKEDEX 1 POKEMON 2 BAG 3 PLAYER 4 SAVE 5 OPTION 6 EXIT 7 RETIRE 8 PLAYER2 (255 closed) */
    uint8_t safari_balls;
    uint8_t object_count;
    uint16_t safari_steps;
    PnObjectEvent objects[16];
    uint8_t dialog_text[256];    /* current page of the shown message, game encoding, 0xFF padded */

    /* ---- progress (INGAME) ---- */
    uint32_t money;
    uint16_t coins;
    uint8_t badges;              /* bit i = badge i+1 */
    uint8_t pokedex_obtained;
    uint16_t dex_seen;
    uint16_t dex_caught;
    uint16_t play_hours;
    uint8_t play_minutes;
    uint8_t player_gender;       /* 0 male, 1 female */
    uint8_t player_name[8];
    uint8_t rival_name[8];
    uint16_t trainer_id_lo;
    uint16_t registered_item;

    /* ---- party and bag (INGAME) ---- */
    uint8_t party_count;         /* slots before the first empty one */
    uint8_t _pad2[3];
    PnMon party[6];
    PnItem items[42];
    PnItem key_items[30];
    PnItem balls[13];
    PnItem tmhm[58];
    PnItem berries[43];
    PnItem pc_items[30];

    /* ---- battle (BATTLE) ---- */
    uint8_t battle_outcome;      /* gBattleOutcome (non-zero before the result messages) */
    uint8_t battle_turn;         /* completed turns */
    uint8_t battler_count;
    uint8_t action_cursor;       /* 0 FIGHT 1 BAG 2 POKEMON 3 RUN (Safari: BALL BAIT ROCK RUN) */
    uint8_t move_cursor;         /* 0 TL 1 TR 2 BL 3 BR */
    uint8_t target_cursor;       /* targeted battler (doubles), 255 none */
    uint8_t yesno_cursor;        /* battle / evolution YES/NO cursor (0 = YES), 255 none */
    uint8_t trainer_class;       /* trainer battles */
    uint8_t opponent_party_count;
    uint8_t battle_input_battler; /* battler whose controller waits for input, 255 none */
    uint16_t battle_weather;     /* gBattleWeather (B_WEATHER_*) */
    uint32_t battle_type_flags;  /* visible subset of gBattleTypeFlags */
    uint8_t trainer_name[13];
    uint8_t _pad3[3];
    PnBattler battlers[4];

    /* ---- privileged ---- */
    uint32_t rng_priv;           /* gRngValue at the boundary */
    uint32_t rng_wild_priv;      /* sWildEncounterData.rngState */
    uint32_t callback1_priv;     /* raw native addresses */
    uint32_t callback2_priv;
    uint32_t callback_saved_priv;
    uint32_t battle_main_func_priv;
    uint32_t controller_funcs_priv[4];
    uint32_t script_ptr_priv;
    uint32_t script_native_priv;
    uint32_t battle_type_flags_priv; /* raw gBattleTypeFlags */
    uint16_t trainer_id_priv;    /* gTrainerBattleOpponent_A */
    uint16_t trainer_id_hi_priv; /* secret ID */
    uint8_t script_mode_priv;
    uint8_t script_status_priv;
    uint8_t play_seconds_priv;
    uint8_t repel_steps_priv;
    uint8_t enemy_party_count_priv;
    uint8_t main_state_priv;     /* gMain.state */
    uint8_t text_flags_priv;     /* gTextFlags */
    uint8_t _pad4[1];
    PnMon enemy_party_priv[6];
    uint32_t game_stats_priv[52];
    uint8_t flags_priv[288];
    uint16_t vars_priv[256];

    /* ---- v3: tile map (FIELD; every v2 offset above is unchanged) ---- */
    uint8_t tiles[165];          /* enum PnTileClass, 11 rows x 15 columns, row-major: tiles[r * 15 + c] is map
                                    (x - 7 + c, y - 5 + r); tiles[5 * 15 + 7] is the player's own tile (terrain).
                                    All PN_TILE_OUTSIDE (0) when the FIELD gate does not hold */
    uint8_t _pad5[3];

    /* ---- v4: menu semantics, item PC, PC boxes (every v3 offset above is unchanged) ---- */
    uint16_t cursor_item;        /* item id under the list cursor: bag list, item PC list, mart BUY list (0: CANCEL / none) */
    uint16_t context_item;       /* the item a context menu, quantity prompt, YES/NO or message acts on (0 none):
                                    gSpecialVar_ItemId in the bag, the item PC's selected slot, the mart's chosen item */
    uint16_t quantity;           /* the number being chosen in a QUANTITY wait (bag toss/sell/deposit, item PC
                                    withdraw, mart buy); 0 otherwise */
    uint16_t quantity_max;       /* its maximum */
    uint8_t menu_action;         /* enum PnMenuAction of the entry under the cursor (menu_actions[menu_cursor]) */
    uint8_t menu_action_count;   /* entries of menu_actions (0: the open menu has no action list) */
    uint8_t menu_actions[8];     /* the menu's entries top to bottom, PN_ACTION_NONE padded */
    uint8_t box_option;          /* enum PnBoxOption (box screen open) */
    uint8_t box_current;         /* gPokemonStoragePtr->currentBox 0..13 (INGAME) */
    uint8_t box_holding;         /* a mon is in hand in the box screen (sIsMonBeingMoved) */
    uint8_t box_held_level;      /* the mon in hand */
    uint16_t box_held_species;
    uint8_t box_choose_cursor;   /* box under the cursor of the box chooser (deposit / jump), 255 none */
    uint8_t box_counts[14];      /* occupied slots per box (INGAME) */
    uint8_t _pad6[1];
    PnBoxMon box_mons[30];       /* the current box (INGAME; the box shown in the box screen) */

    /* ---- v5 (every v4 offset above is unchanged) ---- */
    uint8_t temp_flags;          /* bit k-1 = FLAG_TEMP_k, k = 1..6 (raw; INGAME): the map-local script flags the
                                    game clears on every map change (visible: events the player was shown) */
    uint8_t learned_replaced_priv; /* v8: bit i = learned_move_priv[i]'s mon knew four moves before the step (the
                                    gained move replaced one: the forget-a-move flow), 0 elsewhere / older workers */
    uint8_t _pad7[2];
    uint32_t skill_events_priv;  /* bits 0-15: bit PN_SKILL_x (enum PnSkillEvent) = the skill event x of reward
                                    v0.11 (bits 0-11) / v0.12 (bit 12, mart) ("attempt" definition) at this record;
                                    bit 13 (2026-10-28): a consumable used on the field left the bag during the
                                    step (the game's latch, not a record comparison); bit 14 (2026-10-29): one
                                    used in battle (same latch); bit 15 (2026-10-29): a mart purchase completed;
                                    bits 16-28 RESERVED for the "succeeded" variants of the same events
                                    (bit 16 + x), 0 in v5; bits 29-31 reserved, 0. Written by the worker (privileged mode); 0 from pn_observe */

    /* ---- v6 (every v5 offset above is unchanged): reward v0.14 interactions. Written by the worker in
       privileged mode from the record's own STEP (all 0 for the records of RESET / RESTORE / SNAPSHOT /
       OBSERVE ops, in visible mode and from pn_observe) ---- */
    uint8_t interact_kind_priv;      /* enum PnInteractKind of the FIRST interaction started during the step's
                                        frames (0 none) */
    uint8_t interact_count_priv;     /* interactions started during the step's frames (saturates at 255; >1 is
                                        not expected: an interaction's script locks the player) */
    uint8_t interact_map_group_priv; /* the key's map: OBJECT = the object's own map (gObjectEvents[].mapGroup /
                                        mapNum: a clone reports its original's map), others = the current map */
    uint8_t interact_map_num_priv;
    uint8_t interact_local_id_priv;  /* OBJECT: gObjectEvents[].localId (= gSpecialVar_LastTalked); 0 otherwise */
    uint8_t _pad8[1];
    int16_t interact_x_priv;         /* OBJECT: the object's map coordinates (currentCoords - 7); SIGN /
                                        HIDDEN_ITEM / SIGN_WALK: the background event's map coordinates */
    int16_t interact_y_priv;
    uint8_t _pad9[2];

    /* ---- v7 (every v6 offset above is unchanged): learn-move prompts and learnt moves (behaviour novelty,
       docs/NOVELTY_BEHAVIOUR_PLAN.md; docs/OBSERVATION.md "Changes in v7"). The plan's TM pocket is the v1 field
       tmhm[].id ---- */
    uint16_t offered_move;           /* the move the game offers to teach while it waits on a learn-move YES/NO or
                                        on the forget-move selection (battle / party-menu / evolution level-up
                                        prompts, TM / HM / tutor teaching, the move relearner's YES/NO, the
                                        Summary forget screen reached from any of them); 0 elsewhere. Visible */
    uint8_t offered_slot;            /* party slot (0..5) of the mon concerned; 255 when offered_move is 0 */
    uint8_t summary_cursor;          /* v8: 1 + the Summary moves page cursor (sMoveSelectionCursorPos) while the
                                        page waits for input (SUMMARY_MOVES, SUMMARY_FORGET): 1..4 = move slot
                                        0..3, 5 = row 4 (the move to learn on the forget screen); 0 elsewhere /
                                        older workers. Visible */
    uint16_t learned_move_priv[6];   /* the move party slot i's mon GAINED during the step that produced the
                                        record (0 none): party before / after the step matched by personality
                                        and OT id, so a reorder, deposit, withdrawal or catch reports nothing.
                                        Written by the worker in privileged mode from STEP records (0 for
                                        RESET / RESTORE / SNAPSHOT / OBSERVE records, in visible mode and from
                                        pn_observe) */
    /* ---- v9 (2026-10-28, user; every v8 offset above is unchanged; docs/PC_OBS.md, docs/OBSERVATION.md "Changes
       in v9"): the Pokemon under the storage system's cursor, so the policy knows which mon A would pick up.
       All visible, written whenever the box screen is live (CB2_PokeStorage with gStorage, any task state: the
       selected-mon menu, YES/NO and messages keep the cursor), 0 elsewhere ---- */
    uint8_t box_cursor_area;         /* 0 box screen closed; else 1 + sCursorArea: 1 IN_BOX (the 6 x 5 grid of the
                                        current box), 2 IN_PARTY (the party column), 3 BOX_TITLE, 4 BUTTONS */
    uint8_t box_cursor_pos;          /* sCursorPosition: IN_BOX slot 0..29 (column = pos % 6, row = pos / 6),
                                        IN_PARTY party slot 0..5, BUTTONS button index; 0 when closed */
    uint8_t box_cursor_level;        /* the mon under the cursor: IN_BOX box_mons[pos] (its level from the
                                        experience), IN_PARTY party[pos]; 0 for an egg / an empty slot / closed */
    uint8_t box_cursor_is_egg;
    uint16_t box_cursor_species;     /* visible species (SPECIES_EGG for eggs); 0 = no mon under the cursor */
    uint16_t box_cursor_moves[4];    /* its moves in slot order (IN_BOX: GetBoxMonData MOVE1..4 on a copy of the
                                        80-byte struct BoxPokemon; IN_PARTY: the party mon's moves); 0 padded,
                                        all 0 for an egg / no mon */
    /* ---- v10 (2026-11-03, user; every v9 offset unchanged, sizeof unchanged: docs/OBSERVATION.md "Changes in v10",
       docs/ENTITY_NET.md): the party menu's cursor in every party-menu wait, for the entity network ---- */
    uint8_t party_cursor;            /* party menu waits: the slot the menu acts on (0..5, 6 CONFIRM, 7 CANCEL): the
                                        cursor while choosing a mon (a SWITCH's second cursor), the selected mon in its
                                        submenu, YES/NO and message waits; 255 elsewhere. Visible */
    uint8_t _pad10[1];
} PnObs;
#pragma pack(pop)

PN_OBS_STATIC_ASSERT(sizeof(PnObs) == 4756, "PnObs layout v9");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, box_cursor_area) == 4740, "PnObs v9: appended after the v8 layout");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, party_cursor) == 4754, "PnObs v10: party_cursor in v9's padding");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, offered_move) == 4724, "PnObs v7: appended after the v6 layout");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, offered_slot) == 4726, "PnObs v7: offered_slot");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, learned_move_priv) == 4728, "PnObs v7: learned_move_priv");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, interact_kind_priv) == 4712, "PnObs v6: appended after the v5 layout");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, tiles) == 4256, "PnObs v3: tiles appended after the v2 layout");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, cursor_item) == 4424, "PnObs v4: appended after the v3 layout");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, box_mons) == 4464, "PnObs v4: box_mons");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, temp_flags) == 4704, "PnObs v5: appended after the v4 layout");
PN_OBS_STATIC_ASSERT(offsetof(PnObs, skill_events_priv) == 4708, "PnObs v5: skill_events_priv");

/* sizeof(PnObs) */
size_t pn_obs_size(void);

/* Fills *out from the ACTIVE environment's game state (pn::Simulator::activate). Call
 * it at a frame boundary (after run_frame/step). Pure: never writes game memory.
 * mode: PN_OBS_VISIBLE (privileged fields zeroed) or PN_OBS_PRIVILEGED. */
void pn_observe(PnObs *out, int mode);

/* The waiting_for_input bit pn_observe would report for the ACTIVE environment now, without
 * building the record (the same predicate, docs/OBS_SOURCES.md section 2; ~1 us). Pure. Used by
 * the worker's decision-point stepping (PN_F_AUTO_ADVANCE, docs/IPC_PROTOCOL.md). */
int pn_obs_waiting(void);

/* Testing: enable (default) or disable the observer's decode memo (party/enemy-party slots
 * and Pokedex counts keyed by their exact bytes; host memory, outputs are identical). */
void pn_obs_set_memo(int enabled);

/* "OVERWORLD", ... ; "?" for out-of-range values */
const char *pn_screen_name(int kind);
const char *pn_input_kind_name(int kind);
const char *pn_wait_ctx_name(int ctx);
const char *pn_tile_class_name(int cls);   /* "WALKABLE", ... (v3) */
const char *pn_menu_action_name(int act);  /* "CANCEL", ... (v4) */

/* Number of observer symbols that could not be resolved in the running executable
 * (0 expected; each unresolved predicate simply never reports input). Writes the
 * names to stderr when verbose != 0. */
int pn_obs_selfcheck(int verbose);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FIREREDSIM_OBSERVATION_H */
