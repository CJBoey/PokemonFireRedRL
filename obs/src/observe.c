/* fireredsim observer: fills PnObs from the active environment's game state.
 *
 * Sources and visibility of every field: docs/OBS_SOURCES.md section 1 and
 * docs/OBSERVATION.md. Rules applied throughout (section 0.2):
 *   - read memory only; never call a function that reads keys, runs printers/tasks
 *     or writes globals (FlagGet/VarGet have a Quest Log side effect: raw reads);
 *   - GetMonData/GetMonGender/GetMonAilment/IsMonShiny only on a LOCAL COPY of
 *     struct Pokemon (GetBoxMonData decrypts in place);
 *   - never call GetMonAbility/GetAbilityBySpecies (they write gLastUsedAbility);
 *   - re-read save-block pointers at every call (they move at every map load and
 *     battle start, as does the encryption key).
 * The observer keeps no state of its own except the resolved symbol addresses and a decode
 * memo keyed by the exact bytes it decodes (host memory; outputs identical without it).
 */
#include "global.h"
#include "main.h"
#include "task.h"
#include "text.h"
#include "string_util.h"
#include "script.h"
#include "bg.h"
#include "palette.h"
#include "random.h"
#include "event_data.h"
#include "event_object_movement.h"
#include "field_player_avatar.h"
#include "field_weather.h"
#include "fieldmap.h"
#include "metatile_behavior.h"
#include "overworld.h"
#include "quest_log.h"
#include "pokemon.h"
#include "pokedex.h"
#include "item.h"
#include "money.h"
#include "coins.h"
#include "party_menu.h"
#include "window.h"
#include "pokemon_storage_system_internal.h"
#include "safari_zone.h"
#include "battle.h"
#include "battle_main.h"
#include "battle_anim.h"
#include "battle_message.h"
#include "constants/battle.h"
#include "constants/flags.h"
#include "constants/vars.h"
#include "constants/species.h"
#include "constants/moves.h"
#include "constants/trainers.h"
#include "constants/opponents.h"
#include "constants/event_objects.h"
#include "constants/event_object_movement.h"
#include "constants/game_stat.h"
#include "constants/party_menu.h"
#include "constants/metatile_behaviors.h"

#include "../observation.h"
#include "obs_internal.h"

/* item.c: encryptionKey ^ *quantity (global, not declared in a header) */
u16 GetBagItemQuantity(u16 *ptr);
/* metatile_behavior.c:253 (global, not declared in metatile_behavior.h) */
bool8 MetatileBehavior_IsArrowWarp(u8 metatileBehavior);

_Static_assert(PN_OBS_PARTY_SIZE == PARTY_SIZE, "party size");
_Static_assert(PN_OBS_NUM_OBJECTS == OBJECT_EVENTS_COUNT, "object events");
_Static_assert(PN_OBS_NAME_LEN == PLAYER_NAME_LENGTH + 1, "player name length");
_Static_assert(PN_OBS_NICKNAME_LEN == POKEMON_NAME_LENGTH + 1, "nickname length");
_Static_assert(PN_OBS_BAG_ITEMS == BAG_ITEMS_COUNT, "bag items");
_Static_assert(PN_OBS_BAG_KEY_ITEMS == BAG_KEYITEMS_COUNT, "bag key items");
_Static_assert(PN_OBS_BAG_BALLS == BAG_POKEBALLS_COUNT, "bag balls");
_Static_assert(PN_OBS_BAG_TMHM == BAG_TMHM_COUNT, "bag tm/hm");
_Static_assert(PN_OBS_BAG_BERRIES == BAG_BERRIES_COUNT, "bag berries");
_Static_assert(PN_OBS_PC_ITEMS == PC_ITEMS_COUNT, "pc items");
_Static_assert(PN_OBS_NUM_BATTLERS == MAX_BATTLERS_COUNT, "battlers");
_Static_assert(PN_OBS_FLAG_BYTES == NUM_FLAG_BYTES, "flag bytes");
_Static_assert(PN_OBS_NUM_VARS == VARS_COUNT, "vars");
_Static_assert(PN_OBS_NUM_GAME_STATS == NUM_USED_GAME_STATS, "game stats");
_Static_assert(sizeof(((struct Trainer *)0)->trainerName) + 1 == PN_OBS_TRAINER_NAME_LEN, "trainer name");
_Static_assert(PN_AILMENT_FNT == AILMENT_FNT && PN_AILMENT_PKRS == AILMENT_PKRS && PN_AILMENT_PSN == AILMENT_PSN
               && PN_AILMENT_PRZ == AILMENT_PRZ && PN_AILMENT_SLP == AILMENT_SLP && PN_AILMENT_FRZ == AILMENT_FRZ
               && PN_AILMENT_BRN == AILMENT_BRN, "ailment enum");
/* v6: the game's interaction hook (game/include/gba/portable_abi.h, force-included) reports the kinds of
 * enum PnInteractKind; the worker copies them into interact_*_priv (pn_observe leaves those fields 0) */
_Static_assert((int)PORT_INTERACT_OBJECT == (int)PN_INTERACT_OBJECT && (int)PORT_INTERACT_SIGN == (int)PN_INTERACT_SIGN
               && (int)PORT_INTERACT_HIDDEN_ITEM == (int)PN_INTERACT_HIDDEN_ITEM
               && (int)PORT_INTERACT_SIGN_WALK == (int)PN_INTERACT_SIGN_WALK && PN_INTERACT_COUNT == 5,
               "interaction kinds");

/* ---- symbols and the callback2 table ----------------------------------------------- */

uint32_t gPnObsSym[PNSYM_COUNT];

static const char *const sSymNames[PNSYM_COUNT] = {
#define PN_OBS_SYM_NAME(id, name) name,
    PN_OBS_SYMBOLS(PN_OBS_SYM_NAME)
#undef PN_OBS_SYM_NAME
};

static const char *const sScreenRowNames[] = {
#define PN_SCREEN_ROW(name, kind, role) name,
#include "screen_table.inc"
#undef PN_SCREEN_ROW
};

static const uint8_t sScreenRowKinds[] = {
#define PN_SCREEN_ROW(name, kind, role) PN_SCREEN_##kind,
#include "screen_table.inc"
#undef PN_SCREEN_ROW
};

#define PN_SCREEN_ROWS (sizeof(sScreenRowKinds) / sizeof(sScreenRowKinds[0]))

static uint32_t sScreenAddr[PN_SCREEN_ROWS];
static uint8_t sScreenKind[PN_SCREEN_ROWS]; /* sorted by address */
static int sScreenCount;
static int sInitDone;
static int sMissingSyms, sMissingRows;

static void sort_screen_table(void)
{
    uint32_t addr[PN_SCREEN_ROWS];
    int i, j, n = 0;
    sMissingRows = pn_obs_resolve(sScreenRowNames, addr, (int)PN_SCREEN_ROWS, 0);
    for (i = 0; i < (int)PN_SCREEN_ROWS; i++) {
        if (addr[i] == 0)
            continue;
        /* insertion sort, unique addresses */
        for (j = n; j > 0 && sScreenAddr[j - 1] > addr[i]; j--) {
            sScreenAddr[j] = sScreenAddr[j - 1];
            sScreenKind[j] = sScreenKind[j - 1];
        }
        sScreenAddr[j] = addr[i];
        sScreenKind[j] = sScreenRowKinds[i];
        n++;
    }
    sScreenCount = n;
}

static void tile_init(void);

void pn_obs_init(void)
{
    if (sInitDone)
        return;
    sMissingSyms = pn_obs_resolve(sSymNames, gPnObsSym, PNSYM_COUNT, 0);
    sort_screen_table();
    tile_init();
    sInitDone = 1;
}

int pn_obs_screen_table_lookup(uint32_t cb2)
{
    int lo = 0, hi = sScreenCount - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (sScreenAddr[mid] == cb2)
            return sScreenKind[mid];
        if (sScreenAddr[mid] < cb2)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

int pn_obs_selfcheck(int verbose)
{
    uint32_t tmp[PNSYM_COUNT];
    pn_obs_init();
    if (verbose) {
        uint32_t rows[PN_SCREEN_ROWS];
        pn_obs_resolve(sSymNames, tmp, PNSYM_COUNT, 1);
        pn_obs_resolve(sScreenRowNames, rows, (int)PN_SCREEN_ROWS, 1);
    }
    return sMissingSyms + sMissingRows;
}

/* ---- names ---------------------------------------------------------------------------- */

static const char *const sScreenNames[] = {
    "UNKNOWN", "BOOT", "INTRO", "TITLE", "MAIN_MENU", "NEW_GAME_SPEECH", "NAMING", "OVERWORLD",
    "MAP_TRANSITION", "BATTLE_TRANSITION", "QUEST_LOG", "START_MENU", "SAVE_MENU", "BAG", "TM_CASE",
    "BERRY_POUCH", "PARTY_MENU", "PARTY_ITEM_ANIM", "SUMMARY", "POKEDEX", "TRAINER_CARD", "OPTIONS",
    "PC_MENU", "PC_BOX", "ITEM_PC", "MART", "REGION_MAP", "FLY_MAP", "MOVE_RELEARNER", "EGG_HATCH",
    "TRADE", "MAIL", "EASY_CHAT", "BATTLE", "EVOLUTION", "HALL_OF_FAME", "CREDITS", "SAVE_FAILED",
    "HELP_SYSTEM", "BATTLE_RECORDS", "SLOT_MACHINE", "DIPLOMA", "TEACHY_TV", "FAME_CHECKER", "CUTSCENE",
    "MYSTERY_GIFT", "BERRY_FIX", "CLEAR_SAVE", "LINK",
};

static const char *const sInputKindNames[] = {
    "NONE", "DIALOG_ADVANCE", "YESNO", "MENU", "QUANTITY", "OVERWORLD_FREE", "NAMING", "BATTLE_ACTION",
    "BATTLE_MOVE", "BATTLE_TARGET", "BATTLE_MOVE_SWAP", "BATTLE_PARTY", "BATTLE_BAG", "QUEST_LOG",
    "MAP_CURSOR", "BOX_CHOOSE", "TITLE_START", "HELP", "HOLD_A", "ANY_KEY",
};

static const char *const sWaitCtxNames[] = {
    "NONE", "TITLE", "MAIN_MENU", "OAK_TEXT", "OAK_CONTROLS", "OAK_PIKACHU", "OAK_GENDER",
    "OAK_NAME_CHOICE", "OAK_CONFIRM", "NAMING", "NAMING_PCMSG", "OW_FREE", "DLG_TEXT", "DLG_BUTTON",
    "SCRIPT_YESNO", "MULTICHOICE", "MULTICHOICE_GRID", "SCRIPT_LIST", "START_MENU", "SAVE_YESNO",
    "SAVE_TEXT", "FIELD_MSG_CONT", "SHOP_MENU", "SHOP_BUY", "SHOP_QTY", "SHOP_YESNO", "SHOP_ACK",
    "SHOP_MSG_CONT", "PC_STORAGE_TOP", "PC_PLAYER", "BAG_LIST", "BAG_MOVE", "BAG_CONTEXT", "BAG_QUANTITY",
    "BAG_YESNO", "BAG_MSG_CONT", "BAG_ACK", "PARTY_CHOOSE", "PARTY_SUBMENU", "PARTY_YESNO", "PARTY_MOVE",
    "PARTY_ACK", "PARTY_MSG", "SUMMARY", "SUMMARY_MOVES", "SUMMARY_FORGET", "OPTIONS", "TRAINER_CARD",
    "BATTLE_ACTION", "BATTLE_MOVE", "BATTLE_TARGET", "BATTLE_SWAP", "BATTLE_MSG", "BATTLE_YESNO",
    "BATTLE_LEARN", "BATTLE_NICK", "BATTLE_LVLUP", "EVO_YESNO", "SCENE_MSG", "HELP", "QUEST_LOG", "OW_ANIM", "ITEM_PC",
    /* v2 */
    "DEX_MENU", "DEX_LIST", "DEX_CATEGORY", "DEX_PAGE", "DEX_REGISTER", "PC_BOX", "PC_BOX_MENU", "PC_BOX_YESNO",
    "PC_BOX_CHOOSE", "PC_BOX_MSG", "REGION_MAP", "FLY_MAP", "TM_CASE", "TM_CASE_MENU", "TM_CASE_QTY", "TM_CASE_YESNO",
    "TM_CASE_MSG", "BERRY_POUCH", "BERRY_POUCH_MENU", "BERRY_POUCH_QTY", "BERRY_POUCH_YESNO", "BERRY_POUCH_MSG",
    "RELEARNER", "RELEARNER_YESNO", "RELEARNER_MSG", "MAIN_MENU_MSG", "HALL_OF_FAME",
    /* v3.1 */
    "WHITEOUT_MSG",
    /* v3.2 */
    "BATTLE_RECORDS", "TEACHY_TV_MENU", "TEACHY_TV", "POKEDUDE", "ITEM_ANIM",
};

static const char *const sTileClassNames[] = {
    "OUTSIDE", "WALKABLE", "IMPASSABLE", "WILD", "WATER", "WATERFALL", "LEDGE_S", "LEDGE_W", "LEDGE_E", "LEDGE_N",
    "WARP", "FORCED_E", "FORCED_W", "FORCED_N", "FORCED_S", "ICE", "NPC", "ITEM_BALL", "CUT_TREE", "BOULDER", "ROCK",
    "POKEMON", "OBJECT_OTHER",
};

static const char *const sMenuActionNames[] = {
    "NONE", "CANCEL", "USE", "GIVE", "TOSS", "REGISTER", "WITHDRAW", "DEPOSIT", "MOVE_MONS", "MOVE_ITEMS", "MOVE",
    "SHIFT", "PLACE", "SUMMARY", "SWITCH", "ITEM", "TAKE", "MAIL", "READ", "SEND_OUT", "ENTER", "NO_ENTRY", "TRADE",
    "RELEASE", "MARK", "JUMP", "WALLPAPER", "NAME", "BAG", "INFO", "WALLPAPER_SET", "ITEM_STORAGE", "MAILBOX",
    "PC_STORAGE", "PC_PLAYER", "PC_OAK", "PC_HALL_OF_FAME", "FLASH", "CUT", "FLY", "STRENGTH", "SURF", "ROCK_SMASH",
    "WATERFALL", "TELEPORT", "DIG", "MILK_DRINK", "SOFT_BOILED", "SWEET_SCENT", "OTHER",
};

_Static_assert(sizeof(sMenuActionNames) / sizeof(sMenuActionNames[0]) == PN_ACTION_COUNT, "menu action names");
_Static_assert(sizeof(sTileClassNames) / sizeof(sTileClassNames[0]) == PN_TILE_COUNT, "tile class names");
_Static_assert(sizeof(sScreenNames) / sizeof(sScreenNames[0]) == PN_SCREEN_COUNT, "screen names");
_Static_assert(sizeof(sInputKindNames) / sizeof(sInputKindNames[0]) == PN_INPUT_COUNT, "input kind names");
_Static_assert(sizeof(sWaitCtxNames) / sizeof(sWaitCtxNames[0]) == PN_WAIT_COUNT, "wait ctx names");

const char *pn_screen_name(int kind)
{
    return kind >= 0 && kind < PN_SCREEN_COUNT ? sScreenNames[kind] : "?";
}

const char *pn_input_kind_name(int kind)
{
    return kind >= 0 && kind < PN_INPUT_COUNT ? sInputKindNames[kind] : "?";
}

const char *pn_wait_ctx_name(int ctx)
{
    return ctx >= 0 && ctx < PN_WAIT_COUNT ? sWaitCtxNames[ctx] : "?";
}

const char *pn_tile_class_name(int cls)
{
    return cls >= 0 && cls < PN_TILE_COUNT ? sTileClassNames[cls] : "?";
}

const char *pn_menu_action_name(int act)
{
    return act >= 0 && act < PN_ACTION_COUNT ? sMenuActionNames[act] : "?";
}

size_t pn_obs_size(void)
{
    return sizeof(PnObs);
}

/* ---- helpers ------------------------------------------------------------------------------ */

static void copy_name(uint8_t *dst, const u8 *src, int n)
{
    int i, end = 0;
    for (i = 0; i < n; i++) {
        if (!end && src[i] == EOS)
            end = 1;
        dst[i] = end ? EOS : src[i];
    }
}

/* PnMon from a party slot (section 1.5); visible fields always, privileged when priv */
static void read_mon(PnMon *out, const struct Pokemon *src, int priv)
{
    struct Pokemon m;
    u16 species, species_or_egg;
    u8 buf[POKEMON_NAME_LENGTH + 2];
    int k, egg, bad;

    memset(out->nickname, EOS, sizeof(out->nickname));
    memset(out->ot_name, EOS, sizeof(out->ot_name));
    memcpy(&m, src, sizeof(m)); /* GetMonData decrypts in place: work on a copy */
    species = (u16)GetMonData(&m, MON_DATA_SPECIES);
    if (species == SPECIES_NONE)
        return; /* empty slot: zero, names EOS */
    species_or_egg = (u16)GetMonData(&m, MON_DATA_SPECIES_OR_EGG);
    egg = GetMonData(&m, MON_DATA_IS_EGG) != 0 || species_or_egg == SPECIES_EGG;
    bad = GetMonData(&m, MON_DATA_SANITY_IS_BAD_EGG) != 0;

    out->species = species_or_egg;
    out->is_egg = (uint8_t)egg;
    memset(buf, EOS, sizeof(buf));
    GetMonData(&m, MON_DATA_NICKNAME, buf);
    copy_name(out->nickname, buf, PN_OBS_NICKNAME_LEN);
    memset(buf, EOS, sizeof(buf));
    GetMonData(&m, MON_DATA_OT_NAME, buf);
    copy_name(out->ot_name, buf, PN_OBS_NAME_LEN);
    out->ot_id_lo = (uint16_t)(GetMonData(&m, MON_DATA_OT_ID) & 0xFFFF);
    if (!egg) {
        u8 bonuses = (u8)GetMonData(&m, MON_DATA_PP_BONUSES);
        u8 pokerus = (u8)GetMonData(&m, MON_DATA_POKERUS);
        out->held_item = (uint16_t)GetMonData(&m, MON_DATA_HELD_ITEM);
        out->hp = (uint16_t)GetMonData(&m, MON_DATA_HP);
        out->max_hp = (uint16_t)GetMonData(&m, MON_DATA_MAX_HP);
        for (k = 0; k < MAX_MON_MOVES; k++) {
            u16 mv = (u16)GetMonData(&m, MON_DATA_MOVE1 + k);
            out->moves[k] = mv;
            out->pp[k] = (uint8_t)GetMonData(&m, MON_DATA_PP1 + k);
            out->max_pp[k] = (mv != MOVE_NONE && mv < MOVES_COUNT) ? CalculatePPWithBonus(mv, bonuses, (u8)k) : 0;
        }
        out->attack = (uint16_t)GetMonData(&m, MON_DATA_ATK);
        out->defense = (uint16_t)GetMonData(&m, MON_DATA_DEF);
        out->speed = (uint16_t)GetMonData(&m, MON_DATA_SPEED);
        out->sp_attack = (uint16_t)GetMonData(&m, MON_DATA_SPATK);
        out->sp_defense = (uint16_t)GetMonData(&m, MON_DATA_SPDEF);
        out->exp = GetMonData(&m, MON_DATA_EXP);
        out->level = (uint8_t)GetMonData(&m, MON_DATA_LEVEL);
        out->status = GetMonAilment(&m);
        out->nature = GetNature(&m);
        out->is_shiny = IsMonShiny(&m);
        out->pokerus_state = (pokerus & 0xF) ? 1 : (pokerus ? 2 : 0);
        if (species < NUM_SPECIES && !bad) {
            out->ability = gSpeciesInfo[species].abilities[GetMonData(&m, MON_DATA_ABILITY_NUM) & 1];
            out->type1 = gSpeciesInfo[species].types[0];
            out->type2 = gSpeciesInfo[species].types[1];
            out->gender = GetMonGender(&m);
        } else {
            out->gender = MON_GENDERLESS;
        }
    }
    if (priv) {
        out->species_priv = species;
        out->friendship_priv = (uint8_t)GetMonData(&m, MON_DATA_FRIENDSHIP);
        out->pokerus_priv = (uint8_t)GetMonData(&m, MON_DATA_POKERUS);
        for (k = 0; k < NUM_STATS; k++) {
            out->iv_priv[k] = (uint8_t)GetMonData(&m, MON_DATA_HP_IV + k);
            out->ev_priv[k] = (uint8_t)GetMonData(&m, MON_DATA_HP_EV + k);
        }
        out->pid_priv = GetMonData(&m, MON_DATA_PERSONALITY);
        out->ot_id_priv = GetMonData(&m, MON_DATA_OT_ID);
        out->status_raw_priv = GetMonData(&m, MON_DATA_STATUS);
        out->pp_bonuses_priv = (uint8_t)GetMonData(&m, MON_DATA_PP_BONUSES);
    }
}

/* ---- decode memo (host memory, not game state) ------------------------------------------
 * read_mon is a pure function of the slot's struct Pokemon bytes and constant tables
 * (gSpeciesInfo, gBattleMoves), but costs ~25 GetMonData calls per mon, each decrypting the
 * 48 substruct bytes and verifying the checksum: half of a field observation. The memo keeps
 * the last full (privileged) decode per party/enemy slot keyed by the exact 100 bytes; a hit
 * is a memcmp. The Pokedex counts are memoised the same way, keyed by the four flag arrays
 * the game's getter reads and the national-dex rule. Outputs are bit-identical with the memo
 * disabled (pn_obs_set_memo(0); ctest obs_purity* compare both at every frame). It lives in
 * the process (.bss), outside every environment's state: snapshots, clones and the purity of
 * pn_observe (no game memory written) are unaffected. */
struct PnMonMemo {
    uint8_t valid;
    struct Pokemon key;
    PnMon val; /* privileged decode */
};
static struct PnMonMemo sMonMemo[2][PARTY_SIZE];
static int sMemoEnabled = 1;    /* also gates the v4 box memo */

struct PnDexMemo {
    uint8_t valid, national;
    u8 owned[DEX_FLAGS_NO], seen[DEX_FLAGS_NO], seen1[DEX_FLAGS_NO], seen2[DEX_FLAGS_NO];
    uint16_t seen_count, caught_count;
};
static struct PnDexMemo sDexMemo;

void pn_obs_set_memo(int enabled)
{
    sMemoEnabled = enabled != 0; /* entries are keyed by content: no need to flush them */
}

static void read_mon_memo(PnMon *out, const struct Pokemon *src, int priv, int which, int slot)
{
    struct PnMonMemo *m = &sMonMemo[which][slot];
    if (!sMemoEnabled) {
        read_mon(out, src, priv);
        return;
    }
    if (!m->valid || memcmp(&m->key, src, sizeof(m->key)) != 0) {
        memcpy(&m->key, src, sizeof(m->key));
        memset(&m->val, 0, sizeof(m->val));
        read_mon(&m->val, &m->key, 1);
        m->valid = 1;
    }
    *out = m->val;
    if (!priv) /* the privileged fields are the tail of PnMon */
        memset((uint8_t *)out + offsetof(PnMon, species_priv), 0, sizeof(PnMon) - offsetof(PnMon, species_priv));
}

static void dex_counts(int national, uint16_t *seen, uint16_t *caught)
{
    struct PnDexMemo *d = &sDexMemo;
    const u8 *owned = gSaveBlock2Ptr->pokedex.owned, *sn = gSaveBlock2Ptr->pokedex.seen;
    const u8 *s1 = gSaveBlock1Ptr->seen1, *s2 = gSaveBlock1Ptr->seen2;
    if (sMemoEnabled && d->valid && d->national == national && memcmp(d->owned, owned, DEX_FLAGS_NO) == 0
        && memcmp(d->seen, sn, DEX_FLAGS_NO) == 0 && memcmp(d->seen1, s1, DEX_FLAGS_NO) == 0
        && memcmp(d->seen2, s2, DEX_FLAGS_NO) == 0) {
        *seen = d->seen_count;
        *caught = d->caught_count;
        return;
    }
    *seen = national ? GetNationalPokedexCount(FLAG_GET_SEEN) : GetKantoPokedexCount(FLAG_GET_SEEN);
    *caught = national ? GetNationalPokedexCount(FLAG_GET_CAUGHT) : GetKantoPokedexCount(FLAG_GET_CAUGHT);
    if (!sMemoEnabled)
        return;
    d->national = (uint8_t)national;
    memcpy(d->owned, owned, DEX_FLAGS_NO);
    memcpy(d->seen, sn, DEX_FLAGS_NO);
    memcpy(d->seen1, s1, DEX_FLAGS_NO);
    memcpy(d->seen2, s2, DEX_FLAGS_NO);
    d->seen_count = *seen;
    d->caught_count = *caught;
    d->valid = 1;
}

static void read_items(PnItem *out, const struct ItemSlot *slots, int n, int pc)
{
    int i;
    for (i = 0; i < n; i++) {
        u16 q = slots[i].quantity; /* the accessors take a pointer: pass a copy */
        out[i].id = slots[i].itemId;
        out[i].qty = pc ? GetPcItemQuantity(&q) : GetBagItemQuantity(&q);
    }
}

/* ---- v4: PC boxes (OBS_SOURCES section 1.5 "PC boxes") ----------------------------------
 * Box mons have no level/HP/stats in memory: species (SPECIES_OR_EGG), held item and the level
 * from the experience (GetLevelFromBoxMonExp), all on a copy of the 80-byte struct BoxPokemon.
 * The decode of a whole box (30 GetBoxMonData decrypts) is memoised per box, keyed by the
 * box's exact 2400 bytes, like the party (see the memo note above): a hit is a memcmp. */
static void read_box_mon(PnBoxMon *out, const struct BoxPokemon *src)
{
    struct BoxPokemon b;
    u16 species;
    memset(out, 0, sizeof(*out));
    memcpy(&b, src, sizeof(b));
    species = (u16)GetBoxMonData(&b, MON_DATA_SPECIES);
    if (species == SPECIES_NONE)
        return;
    out->species = (uint16_t)GetBoxMonData(&b, MON_DATA_SPECIES_OR_EGG);
    out->is_egg = (uint8_t)(GetBoxMonData(&b, MON_DATA_IS_EGG) != 0 || out->species == SPECIES_EGG);
    if (!out->is_egg) {
        out->held_item = (uint16_t)GetBoxMonData(&b, MON_DATA_HELD_ITEM);
        out->level = GetLevelFromBoxMonExp(&b);
    }
}

struct PnBoxMemo {
    uint8_t valid;
    uint8_t count;
    struct BoxPokemon key[IN_BOX_COUNT];
    PnBoxMon val[IN_BOX_COUNT];
};
static struct PnBoxMemo sBoxMemo[TOTAL_BOXES_COUNT];

static void read_box(PnBoxMon *val, uint8_t *count, int box)
{
    const struct BoxPokemon *src = gPokemonStoragePtr->boxes[box];
    struct PnBoxMemo *m = &sBoxMemo[box];
    int i, n = 0;
    if (sMemoEnabled && m->valid && memcmp(m->key, src, sizeof(m->key)) == 0) {
        if (val != NULL)
            memcpy(val, m->val, sizeof(m->val));
        *count = m->count;
        return;
    }
    for (i = 0; i < IN_BOX_COUNT; i++) {
        read_box_mon(&m->val[i], &src[i]);
        if (m->val[i].species != SPECIES_NONE)
            n++;
    }
    m->count = (uint8_t)n;
    if (sMemoEnabled) {
        memcpy(m->key, src, sizeof(m->key));
        m->valid = 1;
    } else {
        m->valid = 0;
    }
    if (val != NULL)
        memcpy(val, m->val, sizeof(m->val));
    *count = m->count;
}

/* v9 (docs/PC_OBS.md § 1): the mon under the storage system's cursor, in every task state of the box screen
 * (the selected-mon menu, YES/NO, the messages and the box chooser keep sCursorArea / sCursorPosition where
 * they were). IN_BOX: species / level / is_egg from the current box's decode (o->box_mons, memoised per box),
 * the moves from four GetBoxMonData calls on a copy of that slot (each decrypts, checksums and re-encrypts the
 * substruct: the only extra work of v9, and only while the cursor is on a non-egg box mon); IN_PARTY: the party
 * record (filled by fill_progress before fill_boxes). The box
 * title / buttons areas and empty slots report species 0 and moves 0; an egg SPECIES_EGG, is_egg 1, level 0,
 * moves 0. A mon in hand (box_holding) has left its slot, so the slot under the cursor reads empty. */
static void fill_box_cursor(PnObs *o)
{
    const s8 *area = SYMPTR(const s8 *, sCursorArea_pss);
    const s8 *pos = SYMPTR(const s8 *, sCursorPosition_pss);
    int k;
    if (area == NULL || pos == NULL || *area < CURSOR_AREA_IN_BOX || *area > CURSOR_AREA_BUTTONS || *pos < 0)
        return;
    o->box_cursor_area = (uint8_t)(*area + 1);
    o->box_cursor_pos = (uint8_t)*pos;
    if (*area == CURSOR_AREA_IN_BOX && *pos < IN_BOX_COUNT) {
        const PnBoxMon *bm = &o->box_mons[*pos];
        if (bm->species == SPECIES_NONE)
            return;
        o->box_cursor_species = bm->species;
        o->box_cursor_is_egg = bm->is_egg;
        o->box_cursor_level = bm->level;
        if (!bm->is_egg) {
            struct BoxPokemon b;
            memcpy(&b, &gPokemonStoragePtr->boxes[o->box_current][*pos], sizeof(b)); /* GetBoxMonData decrypts in place */
            for (k = 0; k < MAX_MON_MOVES; k++)
                o->box_cursor_moves[k] = (uint16_t)GetBoxMonData(&b, MON_DATA_MOVE1 + k);
        }
    } else if (*area == CURSOR_AREA_IN_PARTY && *pos < PARTY_SIZE) {
        const PnMon *pm = &o->party[*pos];
        if (pm->species == SPECIES_NONE)
            return;
        o->box_cursor_species = pm->species;
        o->box_cursor_is_egg = pm->is_egg;
        if (!pm->is_egg) {
            o->box_cursor_level = pm->level;
            memcpy(o->box_cursor_moves, pm->moves, sizeof(o->box_cursor_moves));
        }
    }
}

static void fill_boxes(PnObs *o, int ingame)
{
    int b;
    if (!ingame || gPokemonStoragePtr == NULL)
        return;
    o->box_current = gPokemonStoragePtr->currentBox < TOTAL_BOXES_COUNT ? gPokemonStoragePtr->currentBox : 0;
    for (b = 0; b < TOTAL_BOXES_COUNT; b++)
        read_box(b == o->box_current ? o->box_mons : NULL, &o->box_counts[b], b);
    /* the box screen: mode, the mon in hand */
    if (SYM(CB2_PokeStorage) != 0 && (uint32_t)(uintptr_t)gMain.callback2 == SYM(CB2_PokeStorage) && gStorage != NULL) {
        const bool8 *held = SYMPTR(const bool8 *, sIsMonBeingMoved_pss);
        if (gStorage->boxOption < OPTION_EXIT)
            o->box_option = (uint8_t)(gStorage->boxOption + 1);
        if (held != NULL && *held) {
            struct Pokemon m;
            u16 sp;
            memcpy(&m, &gStorage->movingMon, sizeof(m));
            sp = (u16)GetMonData(&m, MON_DATA_SPECIES_OR_EGG);
            o->box_holding = 1;
            o->box_held_species = sp;
            if (sp != SPECIES_NONE && sp != SPECIES_EGG && !GetMonData(&m, MON_DATA_IS_EGG))
                o->box_held_level = (uint8_t)GetMonData(&m, MON_DATA_LEVEL);
        }
        fill_box_cursor(o);
    }
}

/* ---- dialog text (section 1.3) -------------------------------------------------------- */

/* Copies the page currently shown by printer slot w into out (0xFF padded) if the
 * printer's string lies in gStringVar4 or gDisplayedStringBattle. */
static int dialog_page(uint8_t *out, int w)
{
    const struct TextPrinter *tp = SYMPTR(const struct TextPrinter *, sTextPrinters);
    const u8 *end, *base = NULL, *p, *page;
    int cap = 0, len, i;

    memset(out, EOS, PN_OBS_DIALOG_LEN);
    if (tp == NULL)
        return 0;
    end = tp[w].printerTemplate.currentChar;
    if (end >= gStringVar4 && end <= gStringVar4 + 1000) {
        base = gStringVar4;
        cap = 1000;
    } else if (end >= gDisplayedStringBattle && end <= gDisplayedStringBattle + 300) {
        base = gDisplayedStringBattle;
        cap = 300;
    }
    if (base == NULL)
        return 0;
    page = base;
    p = base;
    while (p < end && p < base + cap) {
        u8 c = *p;
        if (c == EXT_CTRL_CODE_BEGIN) {
            u8 n = (p + 1 < base + cap) ? GetExtCtrlCodeLength(p[1]) : 0;
            p += 1 + (n ? n : 1);
            continue;
        }
        if (c == PLACEHOLDER_BEGIN || c == CHAR_KEYPAD_ICON || c == CHAR_EXTRA_SYMBOL) {
            p += 2;
            continue;
        }
        if (c == CHAR_PROMPT_CLEAR && p + 1 < end)
            page = p + 1;
        p++;
    }
    if (p > end)
        p = end;
    len = (int)(p - page);
    while (len > 0 && (page[len - 1] == EOS || page[len - 1] == CHAR_PROMPT_CLEAR))
        len--;
    if (len > PN_OBS_DIALOG_LEN - 1) {
        /* keep the tail, starting after a line break */
        const u8 *s = page + len - (PN_OBS_DIALOG_LEN - 1);
        const u8 *q;
        for (q = s; q < page + len; q++) {
            if (*q == CHAR_NEWLINE || *q == CHAR_PROMPT_SCROLL) {
                s = q + 1;
                break;
            }
        }
        len = (int)(page + len - s);
        page = s;
    }
    for (i = 0; i < len; i++)
        out[i] = page[i];
    return len > 0;
}

static int dialog_state_of(int w)
{
    const struct TextPrinter *tp = SYMPTR(const struct TextPrinter *, sTextPrinters);
    u8 st;
    if (tp == NULL || !tp[w].active)
        return PN_DIALOG_NONE;
    st = tp[w].state;
    if (st >= RENDER_STATE_WAIT && st <= RENDER_STATE_SCROLL_START)
        return gTextFlags.autoScroll ? PN_DIALOG_AUTO : PN_DIALOG_PROMPT;
    return PN_DIALOG_PRINTING;
}

/* ---- HP bar as drawn (section 1.6) ------------------------------------------------------- */

static void hp_bar(PnBattler *out, int b)
{
    const struct BattleBarInfo *bar;
    s32 maxv, target, cur, pixels;
    if (gBattleSpritesDataPtr == NULL || gBattleSpritesDataPtr->battleBars == NULL)
        return;
    bar = &gBattleSpritesDataPtr->battleBars[b];
    maxv = bar->maxValue;
    if (maxv <= 0 || !out->is_active)
        return;
    target = bar->oldValue - bar->receivedValue;
    if (target < 0)
        target = 0;
    if (target > maxv)
        target = maxv;
    if (bar->currValue == -32768) {
        cur = bar->oldValue;
        pixels = cur * 48 / maxv;
    } else if (bar->currValue == 0) {
        cur = target;
        pixels = cur * 48 / maxv;
    } else {
        cur = bar->currValue;
        pixels = maxv < 48 ? ((cur * 48 / maxv) >> 8) : cur * 48 / maxv;
    }
    if (pixels < 0)
        pixels = 0;
    if (pixels > 48)
        pixels = 48;
    if (pixels == 0 && target > 0)
        pixels = 1;
    out->hp_bar_pixels = (uint8_t)pixels;
    out->hp_bar_color = pixels > 24 ? 1 : (pixels > 9 ? 2 : 3);
    out->hp_bar_fraction = (uint8_t)(pixels * 255 / 48);
}

static uint8_t status_category(u32 s1)
{
    if (s1 & STATUS1_SLEEP)
        return PN_AILMENT_SLP;
    if (s1 & STATUS1_PSN_ANY)
        return PN_AILMENT_PSN;
    if (s1 & STATUS1_BURN)
        return PN_AILMENT_BRN;
    if (s1 & STATUS1_FREEZE)
        return PN_AILMENT_FRZ;
    if (s1 & STATUS1_PARALYSIS)
        return PN_AILMENT_PRZ;
    return PN_AILMENT_NONE;
}

static uint32_t status2_visible(u32 s2)
{
    const u32 counters = STATUS2_CONFUSION | STATUS2_UPROAR | STATUS2_BIDE | STATUS2_LOCK_CONFUSE | STATUS2_WRAPPED;
    u32 v = s2 & ~counters;
    if (s2 & STATUS2_CONFUSION)
        v |= 1u << 0;
    if (s2 & STATUS2_UPROAR)
        v |= 1u << 4;
    if (s2 & STATUS2_BIDE)
        v |= 1u << 8;
    if (s2 & STATUS2_LOCK_CONFUSE)
        v |= 1u << 10;
    if (s2 & STATUS2_WRAPPED)
        v |= 1u << 13;
    return v;
}

/* Opponent reveals (docs/OBSERVATION.md "Opponent reveals"), stateless:
 *   moves: a move is revealed once its PP is below the maximum. Cmd_ppreduce is the only
 *     in-battle PP decrement of a chosen move and runs only after its "X used Y!" message has
 *     finished printing (it waits for gBattleControllerExecFlags; Cmd_attackstring before it
 *     in every script, or the charging-turn message right after it for the two-turn moves);
 *     PP survive switches through the party copy. Moves that are named but not used (Disable,
 *     Taunt, Sleep Talk/Metronome callees) stay unrevealed (false negatives, never leaks). The
 *     opponent's mons start every battle with full PP (wild, trainer and roamer mons are
 *     created at the battle start). Output: ascending ids (the slot order is not shown).
 *     A transformed opponent holds the target's (the player's) moves: all reported.
 *   ability: the species' only possible ability (static species data), 0 for species with
 *     two abilities: the observer has no record of ability messages. */
static void opponent_reveals(PnBattler *out, const struct BattlePokemon *bm)
{
    u16 rev[MAX_MON_MOVES];
    int k, n = 0, transformed = (bm->status2 & STATUS2_TRANSFORMED) != 0;
    for (k = 0; k < MAX_MON_MOVES; k++) {
        u16 mv = bm->moves[k];
        int j;
        if (mv == MOVE_NONE || mv >= MOVES_COUNT)
            continue;
        if (!transformed && bm->pp[k] >= CalculatePPWithBonus(mv, bm->ppBonuses, (u8)k))
            continue;
        for (j = n; j > 0 && rev[j - 1] > mv; j--) /* insertion sort */
            rev[j] = rev[j - 1];
        rev[j] = mv;
        n++;
    }
    for (k = 0; k < n; k++)
        out->moves[k] = rev[k];
    if (bm->species != SPECIES_NONE && bm->species < NUM_SPECIES) {
        u8 a0 = gSpeciesInfo[bm->species].abilities[0], a1 = gSpeciesInfo[bm->species].abilities[1];
        if (a1 == ABILITY_NONE || a1 == a0)
            out->ability = a0;
    }
}

#define PN_BATTLE_TYPE_VISIBLE (BATTLE_TYPE_DOUBLE | BATTLE_TYPE_TRAINER | BATTLE_TYPE_FIRST_BATTLE | BATTLE_TYPE_MULTI \
    | BATTLE_TYPE_SAFARI | BATTLE_TYPE_BATTLE_TOWER | BATTLE_TYPE_OLD_MAN_TUTORIAL | BATTLE_TYPE_EREADER_TRAINER \
    | BATTLE_TYPE_GHOST | BATTLE_TYPE_POKEDUDE | BATTLE_TYPE_TRAINER_TOWER)

/* v10: the slot of gPlayerParty (in the order it has now: the battle party menu permutes it) holding a battler's mon.
 * Personality and OT id survive Transform (Cmd_transformdataexecute copies gBattleMons only up to .pp). */
static uint8_t battler_party_slot(const struct BattlePokemon *bm)
{
    int i;
    for (i = 0; i < PARTY_SIZE; i++)
        if (gPlayerParty[i].box.personality == bm->personality && gPlayerParty[i].box.otId == bm->otId)
            return (uint8_t)i;
    return 255;
}

static void read_battler(PnBattler *out, int b, int priv)
{
    const struct BattlePokemon *bm = &gBattleMons[b];
    int own = GetBattlerSide((u8)b) == B_SIDE_PLAYER;
    int k, ghost = 0;
    u8 idx = (u8)gBattlerPartyIndexes[b];

    out->is_player_side = (uint8_t)own;
    out->is_active = (uint8_t)(b < gBattlersCount && !(gAbsentBattlerFlags & (1u << b))
                               && !gSprites[gHealthboxSpriteIds[b]].invisible);
    out->party_slot = 255; /* v10 */
    if (own && b < gBattlersCount && !(gAbsentBattlerFlags & (1u << b)) && bm->species != SPECIES_NONE
        && !(gBattleTypeFlags & BATTLE_TYPE_SAFARI))
        out->party_slot = battler_party_slot(bm);
    if (!own && idx < PARTY_SIZE) {
        struct Pokemon m;
        memcpy(&m, &gEnemyParty[idx], sizeof(m));
        ghost = CheckBattleTypeGhost(&m, (u8)b);
    }
    if (out->is_active) {
        out->species = ghost ? 0 : bm->species;
        out->level = bm->level;
        out->status = status_category(bm->status1);
        out->type1 = ghost ? 0 : bm->type1;
        out->type2 = ghost ? 0 : bm->type2;
        out->status2 = status2_visible(bm->status2);
        for (k = 0; k < NUM_BATTLE_STATS && k < 8; k++)
            out->stat_stages[k] = (int8_t)(bm->statStages[k] - DEFAULT_STAT_STAGE);
        out->gender = MON_GENDERLESS;
        if (!ghost && idx < PARTY_SIZE && bm->species != SPECIES_NONE && bm->species < NUM_SPECIES) {
            /* GetMonGender on a copy; while the in-battle party menu is open gPlayerParty is
             * permuted into battle order (only read it under BattleMainCB2) */
            if (!own || (SYM(BattleMainCB2) != 0 && (uint32_t)(uintptr_t)gMain.callback2 == SYM(BattleMainCB2))) {
                struct Pokemon m;
                memcpy(&m, own ? &gPlayerParty[idx] : &gEnemyParty[idx], sizeof(m));
                if (!GetMonData(&m, MON_DATA_SANITY_IS_BAD_EGG))
                    out->gender = GetMonGender(&m);
            }
        }
        hp_bar(out, b);
        out->last_move = ghost ? 0 : gLastPrintedMoves[b];
        if (!own && !ghost)
            opponent_reveals(out, bm);
        if (own) {
            out->hp = bm->hp;
            out->max_hp = bm->maxHP;
            out->item = bm->item;
            out->ability = bm->ability;
            out->attack = bm->attack;
            out->defense = bm->defense;
            out->speed = bm->speed;
            out->sp_attack = bm->spAttack;
            out->sp_defense = bm->spDefense;
            for (k = 0; k < MAX_MON_MOVES; k++) {
                out->moves[k] = bm->moves[k];
                out->pp[k] = bm->pp[k];
            }
            /* own side: the bar holds the EXP bar in singles; use the exact HP */
            if (bm->maxHP > 0) {
                int px = bm->hp * 48 / bm->maxHP;
                if (px == 0 && bm->hp > 0)
                    px = 1;
                out->hp_bar_pixels = (uint8_t)px;
                out->hp_bar_color = px > 24 ? 1 : (px > 9 ? 2 : 3);
                out->hp_bar_fraction = (uint8_t)(px * 255 / 48);
            }
        }
    }
    if (priv) {
        out->species_priv = bm->species;
        out->hp_priv = bm->hp;
        out->max_hp_priv = bm->maxHP;
        out->item_priv = bm->item;
        for (k = 0; k < MAX_MON_MOVES; k++) {
            out->moves_priv[k] = bm->moves[k];
            out->pp_priv[k] = bm->pp[k];
        }
        out->attack_priv = bm->attack;
        out->defense_priv = bm->defense;
        out->speed_priv = bm->speed;
        out->sp_attack_priv = bm->spAttack;
        out->sp_defense_priv = bm->spDefense;
        out->ability_priv = bm->ability;
        out->party_index_priv = idx;
        out->status1_priv = bm->status1;
        out->status2_priv = bm->status2;
        out->status3_priv = gStatuses3[b];
        out->personality_priv = bm->personality;
    }
}

/* ---- the blocks -------------------------------------------------------------------------- */

static uint8_t player_motion(const struct ObjectEvent *p)
{
    u8 id = p->movementActionId;
    if (!p->heldMovementActive && !p->singleMovementActive)
        return PN_MOTION_IDLE;
    if (!p->heldMovementActive)
        return PN_MOTION_OTHER;
    if (p->heldMovementFinished)
        return PN_MOTION_IDLE;
    if (id >= MOVEMENT_ACTION_JUMP_2_DOWN && id <= MOVEMENT_ACTION_JUMP_2_RIGHT)
        return PN_MOTION_LEDGE;
    if (id >= MOVEMENT_ACTION_PLAYER_RUN_DOWN && id <= MOVEMENT_ACTION_PLAYER_RUN_RIGHT_SLOW)
        return PN_MOTION_RUN;
    if (id >= MOVEMENT_ACTION_WALK_IN_PLACE_SLOW_DOWN && id <= MOVEMENT_ACTION_WALK_IN_PLACE_NORMAL_RIGHT)
        return PN_MOTION_BUMP;
    if (id >= MOVEMENT_ACTION_WALK_IN_PLACE_FAST_DOWN && id <= MOVEMENT_ACTION_WALK_IN_PLACE_FAST_RIGHT)
        return PN_MOTION_TURN;
    if (id <= MOVEMENT_ACTION_FACE_RIGHT_FAST
        || (id >= MOVEMENT_ACTION_DELAY_1 && id <= MOVEMENT_ACTION_DELAY_16))
        return PN_MOTION_TURN;
    return PN_MOTION_STEP;
}

/* ---- tile map (layout v3, docs/OBSERVATION.md section "Tile map") ----------------------
 * Pure reads of the loaded map grid (VMap: the current map plus the MAP_OFFSET = 7 tile
 * connection margin), the current layout's metatile attributes, gMapHeader.events->warps and
 * gObjectEvents. Coordinates below are border coordinates (map coordinates + MAP_OFFSET), the
 * ones VMap and ObjectEvent.currentCoords use. The grid accessors of fieldmap.c are
 * re-implemented inline (one VMap read per cell instead of four lookups); each step cites the
 * game code it copies. */

_Static_assert(PN_OBS_TILES == PN_OBS_TILE_ROWS * PN_OBS_TILE_COLS, "tile window");
_Static_assert(sizeof(((PnObs *)0)->tiles) == PN_OBS_TILES, "tiles field");

/* Behaviour-only part of the classification, one byte per behaviour value, computed once (at
 * init) by calling the game's pure MetatileBehavior_* predicates on every u8 (they take a u8,
 * as the game passes them): the class the behaviour alone decides (ledge, forced movement,
 * waterfall, water, ice; 0 = none) | TILE_BEH_SURFABLE | TILE_BEH_WARP. A host-side table of
 * a pure function of its index, like the screen table; ~0.1 us per observation instead of ~1.2
 * us of predicate calls per window. */
#define TILE_BEH_CLASS 0x1F
#define TILE_BEH_SURFABLE 0x40
#define TILE_BEH_WARP 0x80
static uint8_t sTileBehavior[256];

static uint8_t tile_behavior_entry(u8 b)
{
    uint8_t cls = 0, flags = 0;
    /* a behaviour on which the game uses a warp event: stepped on (IsWarpMetatileBehavior,
     * field_control_avatar.c:901, static there: the same predicates), arrow and directional
     * stair warps (TryArrowWarp, field_control_avatar.c:825), doors entered northwards
     * (TryDoorWarp, field_control_avatar.c:987; MB_WARP_DOOR is in IsWarpMetatileBehavior) */
    if (MetatileBehavior_IsWarpDoor(b) || MetatileBehavior_IsLadder(b) || MetatileBehavior_IsEscalator(b)
        || MetatileBehavior_IsNonAnimDoor(b) || MetatileBehavior_IsLavaridgeB1FWarp(b)
        || MetatileBehavior_IsLavaridge1FWarp(b) || MetatileBehavior_IsWarpPad(b) || MetatileBehavior_IsFallWarp(b)
        || MetatileBehavior_IsUnionRoomWarp(b) || MetatileBehavior_IsArrowWarp(b)
        || MetatileBehavior_IsDirectionalStairWarp(b))
        flags |= TILE_BEH_WARP;
    if (MetatileBehavior_IsSurfable(b))
        flags |= TILE_BEH_SURFABLE;
    if (MetatileBehavior_IsJumpSouth(b))
        cls = PN_TILE_LEDGE_S;
    else if (MetatileBehavior_IsJumpWest(b))
        cls = PN_TILE_LEDGE_W;
    else if (MetatileBehavior_IsJumpEast(b))
        cls = PN_TILE_LEDGE_E;
    else if (MetatileBehavior_IsJumpNorth(b))
        cls = PN_TILE_LEDGE_N;
    /* forced movement: spin tiles (Rocket Hideout, Viridian Gym), walk/slide tiles, currents */
    else if (MetatileBehavior_IsSpinRight(b) || MetatileBehavior_IsWalkEast(b) || MetatileBehavior_IsSlideEast(b)
             || MetatileBehavior_IsEastwardCurrent(b))
        cls = PN_TILE_FORCED_E;
    else if (MetatileBehavior_IsSpinLeft(b) || MetatileBehavior_IsWalkWest(b) || MetatileBehavior_IsSlideWest(b)
             || MetatileBehavior_IsWestwardCurrent(b))
        cls = PN_TILE_FORCED_W;
    else if (MetatileBehavior_IsSpinUp(b) || MetatileBehavior_IsWalkNorth(b) || MetatileBehavior_IsSlideNorth(b)
             || MetatileBehavior_IsNorthwardCurrent(b))
        cls = PN_TILE_FORCED_N;
    else if (MetatileBehavior_IsSpinDown(b) || MetatileBehavior_IsWalkSouth(b) || MetatileBehavior_IsSlideSouth(b)
             || MetatileBehavior_IsSouthwardCurrent(b))
        cls = PN_TILE_FORCED_S;
    else if (MetatileBehavior_IsWaterfall(b))
        cls = PN_TILE_WATERFALL;
    /* sBehaviorSurfable (metatile_behavior.c:5) minus the waterfall and the currents above:
     * pond, fast, deep, ocean, Cycling Road water. Puddles and shallow water are walkable. */
    else if (MetatileBehavior_IsSurfable(b))
        cls = PN_TILE_WATER;
    else if (MetatileBehavior_IsIce(b) || MetatileBehavior_IsThinIce(b) || MetatileBehavior_IsCrackedIce(b))
        cls = PN_TILE_ICE;
    return (uint8_t)(cls | flags);
}

_Static_assert(PN_TILE_COUNT <= TILE_BEH_CLASS, "tile class fits the behaviour table");

static void tile_init(void)
{
    int b;
    for (b = 0; b < 256; b++)
        sTileBehavior[b] = tile_behavior_entry((u8)b);
}

/* terrain class of the in-bounds, defined (not MAPGRID_UNDEFINED) block at border coordinates
 * (x, y); warp: the current map has a warp event there usable at the player's elevation */
static uint8_t tile_terrain(const struct MapLayout *layout, u16 block, s32 x, s32 y, u8 player_elev, int surfing,
                            int warp)
{
    u16 metatile = block & MAPGRID_METATILE_ID_MASK;
    u32 attrs;
    u8 b, enc, elev, beh, cls;

    /* GetMapBorderIdAt (fieldmap.c:580): the interior is CONNECTION_NONE; a margin cell is
     * CONNECTION_INVALID unless that side has a connection */
    if ((x >= VMap.Xsize - (MAP_OFFSET + 1) || x < MAP_OFFSET || y >= VMap.Ysize - MAP_OFFSET || y < MAP_OFFSET)
        && GetMapBorderIdAt(x, y) == CONNECTION_INVALID)
        return PN_TILE_OUTSIDE;

    /* GetAttributeByMetatileIdAndMapLayout (fieldmap.c:432) + ExtractMetatileAttribute
     * (fieldmap.c:377, masks fieldmap.c:63): behaviour bits 0-8, encounter type bits 24-26;
     * an id beyond both tilesets reads 0xFF for every attribute */
    if (metatile < NUM_METATILES_IN_PRIMARY) {
        attrs = layout->primaryTileset->metatileAttributes[metatile];
        b = (u8)(attrs & 0x1FF); /* the MetatileBehavior_* predicates take a u8, as in the game */
        enc = (u8)((attrs & 0x07000000) >> 24);
    } else if (metatile < NUM_METATILES_TOTAL) {
        attrs = layout->secondaryTileset->metatileAttributes[metatile - NUM_METATILES_IN_PRIMARY];
        b = (u8)(attrs & 0x1FF);
        enc = (u8)((attrs & 0x07000000) >> 24);
    } else {
        b = 0xFF;
        enc = 0xFF;
    }
    beh = sTileBehavior[b];
    cls = beh & TILE_BEH_CLASS;

    /* a warp event the game acts on (a warp event on a plain floor or wall tile does nothing:
     * those stay terrain); door tiles have their collision bit set */
    if (warp && (beh & TILE_BEH_WARP))
        return PN_TILE_WARP;

    /* ledges: CheckForObjectEventCollision (field_player_avatar.c:574) jumps before it looks at
     * the collision result, so a ledge stays a ledge whatever its collision bit (Route 1's
     * ledges have it set) */
    if (cls >= PN_TILE_LEDGE_S && cls <= PN_TILE_LEDGE_N)
        return cls;

    /* MapGridGetCollisionAt (fieldmap.c:357) */
    if ((block & MAPGRID_COLLISION_MASK) >> MAPGRID_COLLISION_SHIFT)
        return PN_TILE_IMPASSABLE;

    /* IsElevationMismatchAt (event_object_movement.c:8351): elevation 0 (object) or 0/15 (map)
     * matches anything; MapGridGetElevationAt (fieldmap.c:347). Surfable water is exempt: on
     * foot the mismatch (water 1, land 3) is what stops the player at the shore, Surf is the
     * way in, and the class says so. While surfing, an elevation-3 tile is where Surf ends
     * (CanStopSurfing, field_player_avatar.c:597), not a wall. */
    elev = (u8)(block >> MAPGRID_ELEVATION_SHIFT);
    if (player_elev != 0 && elev != 0 && elev != 15 && elev != player_elev && !(beh & TILE_BEH_SURFABLE)
        && !(surfing && elev == 3))
        return PN_TILE_IMPASSABLE;

    /* forced movement, waterfall, water, ice (in that order, sTileBehavior) */
    if (cls != 0)
        return cls;
    /* wild encounters: StandardWildEncounter (wild_encounter.c:373) tests the metatile's
     * encounter type, not its behaviour (cave, Mansion and Pokemon Tower floors are MB_NORMAL or
     * MB_CAVE with TILE_ENCOUNTER_LAND; no tileset uses MB_INDOOR_ENCOUNTER) */
    if (enc == TILE_ENCOUNTER_LAND)
        return PN_TILE_WILD;
    return PN_TILE_WALKABLE; /* incl. the MB_IMPASSABLE_* one-way walls and MB_STRENGTH_BUTTON */
}

static uint8_t tile_object_class(u8 gfx)
{
    if (gfx >= OBJ_EVENT_GFX_VARS) {
        /* active objects hold resolved ids (SetObjectEventDynamicGraphicsId,
         * event_object_movement.c:2051); otherwise resolve as GetObjectEventGraphicsInfo
         * (event_object_movement.c:2040) does, by a raw read of VAR_OBJ_GFX_ID_x (VarGet has a
         * Quest Log side effect) */
        if (gSaveBlock1Ptr == NULL)
            return PN_TILE_NPC;
        gfx = (u8)gSaveBlock1Ptr->vars[VAR_OBJ_GFX_ID_0 + (gfx - OBJ_EVENT_GFX_VARS) - VARS_START];
    }
    if (gfx >= NUM_OBJ_EVENT_GFX)
        return PN_TILE_NPC; /* drawn as OBJ_EVENT_GFX_LITTLE_BOY */
    switch (gfx) {
    case OBJ_EVENT_GFX_ITEM_BALL:
        return PN_TILE_ITEM_BALL;
    case OBJ_EVENT_GFX_CUT_TREE:
        return PN_TILE_CUT_TREE;
    case OBJ_EVENT_GFX_PUSHABLE_BOULDER:
        return PN_TILE_BOULDER;
    case OBJ_EVENT_GFX_ROCK_SMASH_ROCK:
        return PN_TILE_ROCK;
    case OBJ_EVENT_GFX_TOWN_MAP:
    case OBJ_EVENT_GFX_POKEDEX:
    case OBJ_EVENT_GFX_FOSSIL:
    case OBJ_EVENT_GFX_RUBY:
    case OBJ_EVENT_GFX_SAPPHIRE:
    case OBJ_EVENT_GFX_OLD_AMBER:
    case OBJ_EVENT_GFX_GYM_SIGN:
    case OBJ_EVENT_GFX_SIGN:
    case OBJ_EVENT_GFX_TRAINER_TIPS:
    case OBJ_EVENT_GFX_CLIPBOARD:
    case OBJ_EVENT_GFX_METEORITE:
    case OBJ_EVENT_GFX_LAPRAS_DOLL:
    case OBJ_EVENT_GFX_SEAGALLOP:
    case OBJ_EVENT_GFX_SS_ANNE:
        return PN_TILE_OBJECT_OTHER;
    }
    /* constants/event_objects.h: every id from SNORLAX (109) to DEOXYS_N (150) is a Pokemon sprite */
    if (gfx >= OBJ_EVENT_GFX_SNORLAX && gfx <= OBJ_EVENT_GFX_DEOXYS_N)
        return PN_TILE_POKEMON;
    return PN_TILE_NPC;
}

static void tile_mark(uint8_t *tiles, s32 px, s32 py, s32 x, s32 y, uint8_t cls)
{
    s32 c = x - px + PN_OBS_TILE_COLS / 2, r = y - py + PN_OBS_TILE_ROWS / 2;
    if (c < 0 || c >= PN_OBS_TILE_COLS || r < 0 || r >= PN_OBS_TILE_ROWS)
        return;
    if (c == PN_OBS_TILE_COLS / 2 && r == PN_OBS_TILE_ROWS / 2)
        return; /* the player's own cell keeps its terrain class */
    tiles[r * PN_OBS_TILE_COLS + c] = cls;
}

static void fill_tiles(uint8_t *tiles, const struct ObjectEvent *p)
{
    const struct MapLayout *layout = gMapHeader.mapLayout;
    const struct MapEvents *events = gMapHeader.events;
    s32 px = p->currentCoords.x, py = p->currentCoords.y, r, c, i;
    u8 elev = p->currentElevation;
    int surfing = (gPlayerAvatar.flags & PLAYER_AVATAR_FLAG_SURFING) != 0;
    uint8_t warp[PN_OBS_TILES];

    if (layout == NULL || VMap.map == NULL || layout->primaryTileset == NULL || layout->secondaryTileset == NULL)
        return;
    /* warp events of the current map (map coordinates) in the window, matched as
     * GetWarpEventAtPosition (field_control_avatar.c:1008) does: same tile, elevation 0 or the
     * player's */
    memset(warp, 0, sizeof(warp));
    if (events != NULL && events->warps != NULL) {
        for (i = 0; i < events->warpCount; i++) {
            const struct WarpEvent *we = &events->warps[i];
            c = we->x + MAP_OFFSET - px + PN_OBS_TILE_COLS / 2;
            r = we->y + MAP_OFFSET - py + PN_OBS_TILE_ROWS / 2;
            if (c >= 0 && c < PN_OBS_TILE_COLS && r >= 0 && r < PN_OBS_TILE_ROWS
                && (we->elevation == 0 || we->elevation == elev))
                warp[r * PN_OBS_TILE_COLS + c] = 1;
        }
    }
    for (r = 0; r < PN_OBS_TILE_ROWS; r++) {
        s32 y = py - PN_OBS_TILE_ROWS / 2 + r;
        uint8_t *row = tiles + r * PN_OBS_TILE_COLS;
        if (y < 0 || y >= VMap.Ysize)
            continue; /* beyond the loaded grid: OUTSIDE (0) */
        for (c = 0; c < PN_OBS_TILE_COLS; c++) {
            s32 x = px - PN_OBS_TILE_COLS / 2 + c;
            u16 block;
            if (x < 0 || x >= VMap.Xsize)
                continue;
            block = VMap.map[x + VMap.Xsize * y]; /* GetMapGridBlockAt in bounds (fieldmap.c:59) */
            if (block == MAPGRID_UNDEFINED)
                continue; /* GetMapBorderIdAt: CONNECTION_INVALID (fieldmap.c:582) */
            row[c] = tile_terrain(layout, block, x, y, elev, surfing, warp[r * PN_OBS_TILE_COLS + c]);
        }
    }
    /* objects replace the terrain: every active, drawn object event but the player, at its
     * current tile and, while it moves, the tile it is leaving (DoesObjectCollideWithObjectAt,
     * event_object_movement.c:4904, blocks both) */
    for (i = 0; i < OBJECT_EVENTS_COUNT; i++) {
        const struct ObjectEvent *e = &gObjectEvents[i];
        uint8_t cls;
        if (!e->active || e->isPlayer || e->invisible || e->localId == LOCALID_CAMERA)
            continue;
        cls = tile_object_class(e->graphicsId);
        tile_mark(tiles, px, py, e->currentCoords.x, e->currentCoords.y, cls);
        if (e->previousCoords.x != e->currentCoords.x || e->previousCoords.y != e->currentCoords.y)
            tile_mark(tiles, px, py, e->previousCoords.x, e->previousCoords.y, cls);
    }
}

static void fill_field(PnObs *o, int priv, int ingame, int field)
{
    const struct ObjectEvent *p = NULL;
    int i, n = 0;

    if (gSaveBlock1Ptr == NULL || !ingame)
        return;
    o->map_group = (uint8_t)gSaveBlock1Ptr->location.mapGroup;
    o->map_num = (uint8_t)gSaveBlock1Ptr->location.mapNum;
    o->camera_x = gSaveBlock1Ptr->pos.x;
    o->camera_y = gSaveBlock1Ptr->pos.y;
    o->map_layout_id = gSaveBlock1Ptr->mapLayoutId;
    o->flash_level = gSaveBlock1Ptr->flashLevel;
    o->mapsec = gMapHeader.regionMapSectionId;
    o->floor = gMapHeader.floorNum;
    if (pn_obs_raw_flag(FLAG_SYS_SAFARI_MODE)) {
        o->safari_steps = gSafariZoneStepCounter;
        o->safari_balls = gNumSafariBalls;
    }
    if (gPlayerAvatar.objectEventId < OBJECT_EVENTS_COUNT) {
        p = &gObjectEvents[gPlayerAvatar.objectEventId];
        if (!p->active || !p->isPlayer)
            p = NULL;
    }
    if (p != NULL) {
        o->x = (int16_t)(p->currentCoords.x - MAP_OFFSET);
        o->y = (int16_t)(p->currentCoords.y - MAP_OFFSET);
        o->facing = p->facingDirection;
        o->player_flags = gPlayerAvatar.flags & ~PLAYER_AVATAR_FLAG_CONTROLLABLE;
        o->player_motion = player_motion(p);
    }
    if (!field)
        return;
    o->weather = GetCurrentWeather();
    o->controls_locked = ArePlayerFieldControlsLocked();
    for (i = 0; i < OBJECT_EVENTS_COUNT && n < PN_OBS_NUM_OBJECTS; i++) {
        const struct ObjectEvent *e = &gObjectEvents[i];
        PnObjectEvent *d;
        if (!e->active || e->isPlayer || e->localId == LOCALID_CAMERA)
            continue;
        d = &o->objects[n++];
        d->x = (int16_t)(e->currentCoords.x - MAP_OFFSET);
        d->y = (int16_t)(e->currentCoords.y - MAP_OFFSET);
        d->graphics_id = e->graphicsId;
        d->facing = e->facingDirection;
        d->visible = (uint8_t)(!e->invisible && !e->offScreen);
        d->moving = (uint8_t)((e->heldMovementActive && !e->heldMovementFinished) || e->singleMovementActive);
        if (priv) {
            d->local_id_priv = e->localId;
            d->map_num_priv = e->mapNum;
            d->map_group_priv = e->mapGroup;
            d->movement_type_priv = e->movementType;
            d->trainer_type_priv = e->trainerType;
            d->trainer_range_priv = e->trainerRange_berryTreeId;
        }
    }
    o->object_count = (uint8_t)n;
    if (p != NULL)
        fill_tiles(o->tiles, p);
}

static void fill_dialog(PnObs *o, const PnWait *w, int field)
{
    int win = -1, shown = 0;
    if (field) {
        /* dialog frame corner tile of window 0 at (0,14) of BG0 (DLG_WINDOW_BASE_TILE_NUM) */
        const u16 *tm = (const u16 *)GetBgTilemapBuffer(0);
        int wo = w->whiteout_win; /* v3.1: the whiteout message has its own window */
        shown = tm != NULL && (tm[14 * 32 + 0] & 0x3FF) == 0x200;
        if (wo >= 0)
            shown = 1;
        o->in_dialog = (uint8_t)shown;
        if (shown)
            win = wo >= 0 ? wo : 0;
    } else if (w->screen == PN_SCREEN_BATTLE && pn_obs_battle_gate()) {
        if (dialog_state_of(B_WIN_MSG) != PN_DIALOG_NONE)
            win = B_WIN_MSG;
        else if (dialog_state_of(B_WIN_OAK_OLD_MAN) != PN_DIALOG_NONE)
            win = B_WIN_OAK_OLD_MAN;
        o->in_dialog = win >= 0;
    } else if (w->screen == PN_SCREEN_NEW_GAME_SPEECH) {
        if (dialog_state_of(0) != PN_DIALOG_NONE)
            win = 0;
        o->in_dialog = win >= 0;
    }
    if (win < 0)
        return;
    o->dialog_state = (uint8_t)dialog_state_of(win);
    if (o->dialog_state == PN_DIALOG_NONE)
        o->dialog_state = PN_DIALOG_SHOWN;
    if (w->ctx == PN_WAIT_DLG_BUTTON)
        o->dialog_state = PN_DIALOG_WAIT_BUTTON;
    {
        const struct TextPrinter *tp = SYMPTR(const struct TextPrinter *, sTextPrinters);
        if (tp != NULL)
            o->dialog_font = tp[win].printerTemplate.fontId;
    }
    if (!dialog_page(o->dialog_text, win)) {
        o->dialog_font = 0;
        if (o->dialog_state == PN_DIALOG_SHOWN)
            o->dialog_state = PN_DIALOG_NONE;
    }
}

static void fill_progress(PnObs *o, int priv)
{
    int i;
    const u8 *flags = gSaveBlock1Ptr->flags;
    o->money = GetMoney(&gSaveBlock1Ptr->money);
    o->coins = GetCoins();
    o->badges = flags[FLAG_BADGE01_GET >> 3];
    /* v5: FLAG_TEMP_1..FLAG_TEMP_6, raw (FlagGet writes the Quest Log hook; rule 4) */
    for (i = 0; i < 6; i++) {
        unsigned f = FLAG_TEMP_1 + (unsigned)i;
        if ((flags[f >> 3] >> (f & 7)) & 1)
            o->temp_flags |= (uint8_t)(1u << i);
    }
    o->pokedex_obtained = (uint8_t)pn_obs_raw_flag(FLAG_SYS_POKEDEX_GET);
    if (o->pokedex_obtained) { /* the counts are shown only once the Pokedex is owned */
        int national = gSaveBlock2Ptr->pokedex.nationalMagic == 0xB9
            && gSaveBlock1Ptr->vars[VAR_NATIONAL_DEX - VARS_START] == 0x6258 && pn_obs_raw_flag(FLAG_SYS_NATIONAL_DEX);
        uint16_t seen, caught;
        dex_counts(national, &seen, &caught);
        o->dex_seen = seen;
        o->dex_caught = caught;
    }
    o->play_hours = gSaveBlock2Ptr->playTimeHours;
    o->play_minutes = gSaveBlock2Ptr->playTimeMinutes;
    o->player_gender = gSaveBlock2Ptr->playerGender;
    copy_name(o->player_name, gSaveBlock2Ptr->playerName, PN_OBS_NAME_LEN);
    copy_name(o->rival_name, gSaveBlock1Ptr->rivalName, PN_OBS_NAME_LEN);
    o->trainer_id_lo = (uint16_t)(gSaveBlock2Ptr->playerTrainerId[0] | (gSaveBlock2Ptr->playerTrainerId[1] << 8));
    o->registered_item = gSaveBlock1Ptr->registeredItem;

    for (i = 0; i < PARTY_SIZE; i++)
        read_mon_memo(&o->party[i], &gPlayerParty[i], priv, 0, i);
    for (i = 0; i < PARTY_SIZE && o->party[i].species != SPECIES_NONE; i++)
        ;
    o->party_count = (uint8_t)i;
    read_items(o->items, gSaveBlock1Ptr->bagPocket_Items, BAG_ITEMS_COUNT, 0);
    read_items(o->key_items, gSaveBlock1Ptr->bagPocket_KeyItems, BAG_KEYITEMS_COUNT, 0);
    read_items(o->balls, gSaveBlock1Ptr->bagPocket_PokeBalls, BAG_POKEBALLS_COUNT, 0);
    read_items(o->tmhm, gSaveBlock1Ptr->bagPocket_TMHM, BAG_TMHM_COUNT, 0);
    read_items(o->berries, gSaveBlock1Ptr->bagPocket_Berries, BAG_BERRIES_COUNT, 0);
    read_items(o->pc_items, gSaveBlock1Ptr->pcItems, PC_ITEMS_COUNT, 1);
}

static int battle_phase(void)
{
    uint32_t f = (uint32_t)(uintptr_t)gBattleMainFunc;
    uint32_t cb2 = (uint32_t)(uintptr_t)gMain.callback2;
    if (!gMain.inBattle) {
        if (cb2 == SYM(CB2_OverworldBasic) && cb2 != 0)
            return PN_BATTLE_PENDING;
        if (SYM(CB2_Overworld) != 0 && cb2 == SYM(CB2_Overworld) && pn_obs_task_find(SYM(Task_BattleStart)) >= 0)
            return PN_BATTLE_PENDING;
        return PN_BATTLE_NONE;
    }
    if ((cb2 == SYM(CB2_EvolutionSceneUpdate) && cb2 != 0) || (f == SYM(WaitForEvoSceneToFinish) && f != 0)
        || (f == SYM(TryEvolvePokemon) && f != 0))
        return PN_BATTLE_EVOLUTION;
    if ((f == SYM(FreeResetData_ReturnToOvOrDoEvolutions) && f != 0) || (f == SYM(ReturnFromBattleToOverworld) && f != 0))
        return PN_BATTLE_ENDING;
    if (!pn_obs_battle_gate())
        return PN_BATTLE_INTRO;
    if (gBattleOutcome != 0)
        return PN_BATTLE_ENDING;
    return PN_BATTLE_ACTIVE;
}

static void fill_battle(PnObs *o, const PnWait *w, int priv)
{
    int b, n, i;
    o->battle_phase = (uint8_t)battle_phase();
    if (!pn_obs_battle_gate())
        return;
    o->in_battle = 1;
    o->valid |= PN_VALID_BATTLE;
    o->battle_outcome = gBattleOutcome;
    o->battle_turn = gBattleResults.battleTurnCounter;
    o->battler_count = gBattlersCount;
    o->battle_weather = gBattleWeather;
    o->battle_type_flags = gBattleTypeFlags & PN_BATTLE_TYPE_VISIBLE;
    o->battle_input_battler = w->battle_input_battler;
    o->yesno_cursor = w->yesno_cursor;
    o->target_cursor = 255;
    if (w->battle_input_battler < MAX_BATTLERS_COUNT) {
        b = w->battle_input_battler;
        o->action_cursor = gActionSelectionCursor[b];
        o->move_cursor = gMoveSelectionCursor[b];
        if (w->ctx == PN_WAIT_BATTLE_TARGET || w->ctx == PN_WAIT_BATTLE_SWAP)
            o->target_cursor = gMultiUsePlayerCursor;
    } else {
        o->action_cursor = gActionSelectionCursor[0];
        o->move_cursor = gMoveSelectionCursor[0];
    }
    if ((gBattleTypeFlags & BATTLE_TYPE_TRAINER) && gTrainerBattleOpponent_A < NUM_TRAINERS) {
        const struct Trainer *t = &gTrainers[gTrainerBattleOpponent_A];
        o->trainer_class = t->trainerClass;
        if (t->trainerClass == TRAINER_CLASS_RIVAL_EARLY || t->trainerClass == TRAINER_CLASS_RIVAL_LATE
            || t->trainerClass == TRAINER_CLASS_CHAMPION)
            copy_name(o->trainer_name, gSaveBlock1Ptr->rivalName, PN_OBS_NAME_LEN);
        else
            copy_name(o->trainer_name, t->trainerName, (int)sizeof(t->trainerName));
        o->trainer_name[PN_OBS_TRAINER_NAME_LEN - 1] = EOS;
        for (i = 0; i < PARTY_SIZE; i++) {
            struct Pokemon m;
            u16 sp;
            memcpy(&m, &gEnemyParty[i], sizeof(m));
            sp = (u16)GetMonData(&m, MON_DATA_SPECIES_OR_EGG);
            if (sp != SPECIES_NONE && sp != SPECIES_EGG)
                o->opponent_party_count++;
        }
    } else {
        n = gBattlersCount > MAX_BATTLERS_COUNT ? MAX_BATTLERS_COUNT : gBattlersCount;
        for (b = 0; b < n; b++)
            if (GetBattlerSide((u8)b) != B_SIDE_PLAYER)
                o->opponent_party_count++;
    }
    if (pn_obs_battle_mons_gate()) {
        o->valid |= PN_VALID_BATTLE_MONS;
        n = gBattlersCount > MAX_BATTLERS_COUNT ? MAX_BATTLERS_COUNT : gBattlersCount;
        for (b = 0; b < n; b++)
            read_battler(&o->battlers[b], b, priv);
    }
    if (priv) {
        o->battle_type_flags_priv = gBattleTypeFlags;
        o->trainer_id_priv = gTrainerBattleOpponent_A;
        for (i = 0; i < PARTY_SIZE; i++)
            read_mon_memo(&o->enemy_party_priv[i], &gEnemyParty[i], 1, 1, i);
        for (i = 0; i < PARTY_SIZE && o->enemy_party_priv[i].species_priv != SPECIES_NONE; i++)
            ;
        o->enemy_party_count_priv = (uint8_t)i;
        /* read_mon fills the visible part too; enemy_party_priv is privileged as a whole */
    }
}

static void fill_priv(PnObs *o, int ingame)
{
    const struct ScriptContext *c = SYMPTR(const struct ScriptContext *, sGlobalScriptContext);
    const u8 *status = SYMPTR(const u8 *, sGlobalScriptContextStatus);
    const u32 *wild = SYMPTR(const u32 *, sWildEncounterData);
    int b, i;
    (void)ingame;
    o->rng_priv = gRngValue;
    o->rng_wild_priv = wild != NULL ? wild[0] : 0;
    o->callback1_priv = (uint32_t)(uintptr_t)gMain.callback1;
    o->callback2_priv = (uint32_t)(uintptr_t)gMain.callback2;
    o->callback_saved_priv = (uint32_t)(uintptr_t)gMain.savedCallback;
    o->battle_main_func_priv = (uint32_t)(uintptr_t)gBattleMainFunc;
    for (b = 0; b < MAX_BATTLERS_COUNT; b++)
        o->controller_funcs_priv[b] = (uint32_t)(uintptr_t)gBattlerControllerFuncs[b];
    if (c != NULL) {
        o->script_ptr_priv = (uint32_t)(uintptr_t)c->scriptPtr;
        o->script_native_priv = (uint32_t)(uintptr_t)c->nativePtr;
        o->script_mode_priv = c->mode;
    }
    o->script_status_priv = status != NULL ? *status : 0;
    o->main_state_priv = gMain.state;
    o->text_flags_priv = (uint8_t)(gTextFlags.canABSpeedUpPrint | (gTextFlags.useAlternateDownArrow << 1)
                                   | (gTextFlags.autoScroll << 2) | (gTextFlags.forceMidTextSpeed << 3));
    if (gSaveBlock1Ptr == NULL || gSaveBlock2Ptr == NULL)
        return;
    o->trainer_id_hi_priv = (uint16_t)(gSaveBlock2Ptr->playerTrainerId[2] | (gSaveBlock2Ptr->playerTrainerId[3] << 8));
    o->play_seconds_priv = gSaveBlock2Ptr->playTimeSeconds;
    o->repel_steps_priv = (uint8_t)gSaveBlock1Ptr->vars[VAR_REPEL_STEP_COUNT - VARS_START];
    for (i = 0; i < NUM_USED_GAME_STATS; i++)
        o->game_stats_priv[i] = GetGameStat((u8)i);
    memcpy(o->flags_priv, gSaveBlock1Ptr->flags, sizeof(o->flags_priv));
    memcpy(o->vars_priv, gSaveBlock1Ptr->vars, sizeof(o->vars_priv));
}

/* ---- entry point ------------------------------------------------------------------------- */

int pn_obs_waiting(void)
{
    PnWait w;
    uint64_t frame = 0;
    uint8_t soft_reset = 0;

    pn_obs_init();
    pn_obs_host_info(&frame, &soft_reset);
    pn_obs_wait(&w, soft_reset);
    return w.waiting;
}

void pn_observe(PnObs *out, int mode)
{
    PnWait w;
    uint64_t frame = 0;
    uint8_t soft_reset = 0;
    int priv = mode == PN_OBS_PRIVILEGED;
    int ingame, field;

    pn_obs_init();
    memset(out, 0, sizeof(*out));
    memset(out->dialog_text, EOS, sizeof(out->dialog_text));
    memset(out->player_name, EOS, sizeof(out->player_name));
    memset(out->rival_name, EOS, sizeof(out->rival_name));
    memset(out->trainer_name, EOS, sizeof(out->trainer_name));
    for (int i = 0; i < PARTY_SIZE; i++) {
        memset(out->party[i].nickname, EOS, sizeof(out->party[i].nickname));
        memset(out->party[i].ot_name, EOS, sizeof(out->party[i].ot_name));
    }
    pn_obs_host_info(&frame, &soft_reset);

    out->magic = PN_OBS_MAGIC;
    out->version = PN_OBS_VERSION;
    out->mode = (uint16_t)(priv ? PN_OBS_PRIVILEGED : PN_OBS_VISIBLE);
    out->size = (uint32_t)sizeof(PnObs);
    out->frame = frame;
    out->priv_valid = (uint8_t)priv;
    out->soft_reset_happened = soft_reset;
    out->held_keys_prev = gMain.heldKeysRaw;

    pn_obs_wait(&w, soft_reset);
    out->screen = w.screen;
    out->waiting_for_input = w.waiting;
    out->input_kind = w.waiting ? w.kind : PN_INPUT_NONE;
    out->input_keys = w.waiting ? w.keys : 0;
    out->input_flags = w.flags;
    out->wait_ctx = w.waiting ? w.ctx : PN_WAIT_NONE;
    out->field_state = w.field_state;
    out->menu_open = w.menu_open;
    out->menu_cursor = w.menu_cursor;
    out->menu_count = w.menu_count;
    out->menu_id = w.menu_id;
    /* v4 */
    out->cursor_item = w.cursor_item;
    out->context_item = w.context_item;
    out->quantity = w.quantity;
    out->quantity_max = w.quantity_max;
    out->menu_action_count = w.menu_action_count;
    memcpy(out->menu_actions, w.menu_actions, sizeof(out->menu_actions));
    out->menu_action = w.menu_cursor < w.menu_action_count ? w.menu_actions[w.menu_cursor] : PN_ACTION_NONE;
    out->box_choose_cursor = w.box_choose_cursor;
    out->yesno_cursor = 255;
    out->target_cursor = 255;
    out->battle_input_battler = 255;
    out->start_menu_cursor = 255;
    /* v7: the learn-move prompt's offer (set only by the predicates of those waits) */
    out->offered_move = w.waiting ? w.offered_move : 0;
    out->offered_slot = w.waiting && w.offered_move != 0 ? w.offered_slot : 255;
    out->summary_cursor = w.waiting ? w.summary_cursor : 0; /* v8 */
    out->party_cursor = w.waiting && w.party_cursor != 0 ? (uint8_t)(w.party_cursor - 1) : 255; /* v10 */

    ingame = pn_obs_ingame() && gSaveBlock1Ptr != NULL && gSaveBlock2Ptr != NULL;
    field = pn_obs_field_gate();
    if (ingame)
        out->valid |= PN_VALID_INGAME;
    if (field)
        out->valid |= PN_VALID_FIELD;

    if (field && (pn_obs_task_find(SYM(Task_StartMenuHandleInput)) >= 0 || pn_obs_task_find(SYM(task50_startmenu)) >= 0
                  || pn_obs_task_find(SYM(Task_WaitFadeAndCreateStartMenuTask)) >= 0)) {
        const u8 *pos = SYMPTR(const u8 *, sStartMenuCursorPos);
        const u8 *order = SYMPTR(const u8 *, sStartMenuOrder);
        if (pos != NULL && order != NULL && *pos < 9)
            out->start_menu_cursor = order[*pos];
    }

    fill_field(out, priv, ingame, field && ingame);
    fill_dialog(out, &w, field);
    if (ingame)
        fill_progress(out, priv);
    fill_battle(out, &w, priv);
    fill_boxes(out, ingame);
    if (priv)
        fill_priv(out, ingame);
}
