/* fireredsim: the skill-retention events of reward v0.11 and v0.12 (mart, mart_buy) (obs/skill_events.h). Mirrors
 * reward v0.11's event derivation (since v5 kept as the reference rl/pokerl/reward_ref.py SkillRefV011.reset /
 * .events, with v0.12's mart event) line by line; the comments name the reference's variables. The
 * use_consumable (bit 13, 2026-10-28) and use_battle_item (bit 14, 2026-10-29) events are the exception: they
 * come from the game's item-use latch (pn_skill_record_ex's item_use), not from the records, so the reference cannot
 * derive them. */
#include "global.h"
#include "item.h"
#include "party_menu.h"
#include "constants/items.h"

#include "observation.h"
#include "skill_events.h"

_Static_assert(PN_SKILL_COUNT == 16 && PN_SKILL_MART == 12 && PN_SKILL_USE_CONSUMABLE == 13 &&
               PN_SKILL_USE_BATTLE_ITEM == 14 && PN_SKILL_MART_BUY == 15,
               "reward.SKILL_EVENTS has 16 events: mart, use_consumable, use_battle_item, mart_buy");
_Static_assert(PN_SKILL_COUNT <= PN_SKILL_SUCCESS_SHIFT, "event bits below the reserved succeeded variants");
_Static_assert(POCKET_ITEMS == 1 && POCKET_KEY_ITEMS == 2 && POCKET_BERRY_POUCH == 5, "reward.POCKET_*");

#define TILE_CENTER (5 * PN_OBS_TILE_COLS + 7)

uint8_t pn_skill_item_pocket(uint16_t item)
{
    /* reward.ITEM_POCKET = tables.ITEM_POCKET (gen_tables.item_pockets: items.json, 0 for ITEM_NONE entries) */
    if (item >= ITEMS_COUNT || gItems[item].itemId == ITEM_NONE)
        return 0;
    return gItems[item].pocket;
}

uint16_t pn_skill_tm_move(uint16_t item)
{
    /* reward.TM_MOVE: party_menu.c ItemIdToBattleMoveId (sTMHMMoves) for TM01 .. HM08 */
    if (item < ITEM_TM01 || item > ITEM_HM08)
        return 0;
    return ItemIdToBattleMoveId(item);
}

/* the fields of a record the events read (reward.Parsed) */
typedef struct Cur {
    uint8_t screen, ctx, waiting, action, pocket, ahead, inb, holding, in_dialog, ingame;
    uint16_t ctx_item;
    uint64_t key[PN_OBS_PARTY_SIZE];
} Cur;

static void parse(const PnObs *r, Cur *c)
{
    int j, facing = r->facing;
    c->screen = r->screen;
    c->ctx = r->wait_ctx;
    c->waiting = r->waiting_for_input != 0;
    c->action = r->menu_action;
    c->ctx_item = r->context_item;
    c->pocket = pn_skill_item_pocket(r->context_item);
    c->inb = r->in_battle != 0;
    c->holding = r->box_holding != 0;
    c->in_dialog = r->in_dialog != 0;
    c->ingame = (r->valid & PN_VALID_INGAME) != 0;
    /* reward.AHEAD_OFFSET / AHEAD_VALID: the tile the player faces, 0 for another facing value */
    c->ahead = 0;
    if (facing >= 1 && facing <= 4) {
        static const int d[5] = {0, PN_OBS_TILE_COLS, -PN_OBS_TILE_COLS, -1, 1};
        c->ahead = r->tiles[TILE_CENTER + d[facing]];
    }
    for (j = 0; j < PN_OBS_PARTY_SIZE; j++)
        c->key[j] = ((uint64_t)r->party[j].species << 32) | (uint64_t)r->party[j].pid_priv;
}

/* reward_ref._skill_pc_bits: the PC / shop session bits after this record */
static uint8_t pc_bits(const Cur *c, uint8_t bits)
{
    uint8_t out = (c->screen == PN_SCREEN_OVERWORLD && c->waiting && c->ctx == PN_WAIT_OW_FREE) ? 0 : bits;
    if (c->waiting && c->ctx == PN_WAIT_PC_STORAGE_TOP)
        out |= PN_SKILL_IN_PC_STORAGE;
    if (c->waiting && c->ctx == PN_WAIT_PC_PLAYER)
        out |= PN_SKILL_IN_PC_PLAYER;
    if (c->waiting && c->ctx == PN_WAIT_SHOP_MENU)
        out |= PN_SKILL_IN_MART;
    return out;
}

/* reward._skill_tm */
static uint16_t skill_tm(const Cur *c, uint16_t tm)
{
    uint16_t mv = pn_skill_tm_move(c->ctx_item);
    return (mv != 0 && c->screen == PN_SCREEN_PARTY_MENU) ? mv : tm;
}

/* reward._skill_row (+ the PC / shop session bits in [7]) */
static void set_row(uint8_t *row, const Cur *c, uint8_t bits)
{
    row[0] = c->screen;
    row[1] = c->ctx;
    row[2] = c->waiting;
    row[3] = c->action;
    row[4] = c->pocket;
    row[5] = c->ahead;
    row[6] = c->inb;
    row[7] = bits;
}

static void copy_moves(uint16_t *dst, const PnObs *r)
{
    int j, k;
    for (j = 0; j < PN_OBS_PARTY_SIZE; j++)
        for (k = 0; k < 4; k++)
            dst[j * 4 + k] = r->party[j].moves[k];
}

/* np.sort(a) == np.sort(b) over the 6 identities */
static int same_set6(const uint64_t *a, const uint64_t *b)
{
    uint64_t x[PN_OBS_PARTY_SIZE], y[PN_OBS_PARTY_SIZE];
    int i, j;
    for (j = 0; j < PN_OBS_PARTY_SIZE; j++) {
        uint64_t v = a[j], w = b[j];
        for (i = j - 1; i >= 0 && x[i] > v; i--)
            x[i + 1] = x[i];
        x[i + 1] = v;
        for (i = j - 1; i >= 0 && y[i] > w; i--)
            y[i + 1] = y[i];
        y[i + 1] = w;
    }
    for (j = 0; j < PN_OBS_PARTY_SIZE; j++)
        if (x[j] != y[j])
            return 0;
    return 1;
}

void pn_skill_reset(PnSkillState *st)
{
    st->valid = 0;
}

uint32_t pn_skill_record(PnSkillState *st, const PnObs *rec)
{
    return pn_skill_record_ex(st, rec, 0);
}

uint32_t pn_skill_record_ex(PnSkillState *st, const PnObs *rec, unsigned item_use)
{
    Cur c;
    int j, k, q;
    uint32_t ev = 0;
    parse(rec, &c);
    if (!st->valid) {
        /* RewardV0.reset: the first record is the "previous step"; the PC / shop session bits from its waits;
         * the party identities whatever box_holding says; no in-game gate */
        set_row(st->prev, &c, pc_bits(&c, 0));
        for (j = 0; j < PN_OBS_PARTY_SIZE; j++)
            st->key[j] = c.key[j];
        copy_moves(st->moves, rec);
        st->tm = skill_tm(&c, 0);
        st->valid = 1;
        return 0;
    }
    {
        const uint8_t *prev = st->prev;
        const int pscreen = prev[0], pctx = prev[1], pwait = prev[2] != 0, pact = prev[3], ppock = prev[4],
                  pahead = prev[5], pinb = prev[6] != 0;
        const uint8_t bits = prev[7];
        const int same_set = same_set6(c.key, st->key);
        int same_mon[PN_OBS_PARTY_SIZE], changed = 0, tm_hit = 0;
        int left, used, left_bp, berry, inter;
        for (j = 0; j < PN_OBS_PARTY_SIZE; j++) {
            same_mon[j] = c.key[j] == st->key[j];
            changed |= !same_mon[j];
        }
        left = c.ctx != PN_WAIT_BAG_CONTEXT && c.ctx != PN_WAIT_BAG_LIST && c.ctx != PN_WAIT_BAG_MOVE;
        used = pwait && pctx == PN_WAIT_BAG_CONTEXT && pact == PN_ACTION_USE && left && !pinb;
        left_bp = c.ctx != PN_WAIT_BERRY_POUCH_MENU && c.ctx != PN_WAIT_BERRY_POUCH;
        berry = pwait && pctx == PN_WAIT_BERRY_POUCH_MENU && pact == PN_ACTION_USE && left_bp && !pinb
                && ppock == POCKET_BERRY_POUCH;
        /* tm_hit: a move of a same-identity slot, not among that slot's previous moves, equal to skill_tm */
        if (st->tm != 0) {
            for (j = 0; j < PN_OBS_PARTY_SIZE && !tm_hit; j++) {
                if (!same_mon[j])
                    continue;
                for (k = 0; k < 4 && !tm_hit; k++) {
                    uint16_t mv = rec->party[j].moves[k];
                    int known = 0;
                    if (mv == 0 || mv != st->tm)
                        continue;
                    for (q = 0; q < 4; q++)
                        known |= st->moves[j * 4 + q] == mv;
                    tm_hit = !known;
                }
            }
        }
        inter = pwait && pctx == PN_WAIT_OW_FREE && c.in_dialog && c.screen == PN_SCREEN_OVERWORLD;
#define EV(bit, cond) ev |= ((cond) ? 1u << (bit) : 0u)
        EV(PN_SKILL_PARTY_MENU, c.screen == PN_SCREEN_PARTY_MENU && pscreen != PN_SCREEN_PARTY_MENU && !c.inb);
        EV(PN_SKILL_BAG_MENU, c.screen == PN_SCREEN_BAG && pscreen != PN_SCREEN_BAG && !c.inb);
        EV(PN_SKILL_PARTY_ORDER, same_set && changed && !c.inb && c.screen != PN_SCREEN_PC_BOX && !c.holding);
        EV(PN_SKILL_USE_ITEM, (used && ppock == POCKET_ITEMS) || berry);
        EV(PN_SKILL_USE_KEY_ITEM, used && ppock == POCKET_KEY_ITEMS);
        EV(PN_SKILL_USE_TM, tm_hit && !c.inb && c.screen != PN_SCREEN_BATTLE && c.screen != PN_SCREEN_EVOLUTION
                                && c.screen != PN_SCREEN_MOVE_RELEARNER);
        EV(PN_SKILL_PC_STORAGE, c.waiting && c.ctx == PN_WAIT_PC_STORAGE_TOP && !(bits & PN_SKILL_IN_PC_STORAGE));
        EV(PN_SKILL_PC_PLAYER, c.waiting && c.ctx == PN_WAIT_PC_PLAYER && !(bits & PN_SKILL_IN_PC_PLAYER));
        EV(PN_SKILL_PC_PARTY, !same_set && (c.screen == PN_SCREEN_PC_BOX || pscreen == PN_SCREEN_PC_BOX) && !c.holding);
        EV(PN_SKILL_TREE, inter && pahead == PN_TILE_CUT_TREE);
        EV(PN_SKILL_BOULDER, inter && pahead == PN_TILE_BOULDER);
        EV(PN_SKILL_ROCK, inter && pahead == PN_TILE_ROCK);
        /* v0.12 (2026-10-20): the mart's BUY / SELL / QUIT menu (Task_ShopMenu, any clerk) waits, once per shop
         * session (a session ends when the player is free on the field), as pc_storage */
        EV(PN_SKILL_MART, c.waiting && c.ctx == PN_WAIT_SHOP_MENU && !(bits & PN_SKILL_IN_MART));
        /* 2026-10-28: a consumable the player used on the field was removed from the bag during the step (the
         * game's PORT_NOTE_ITEM_USE latch, taken by the worker after the step's frames); 2026-10-29: one used in
         * battle is its own event (the latch counts the uses made while gMain.inBattle apart: the medicine path
         * is shared with the battle's party menu) */
        EV(PN_SKILL_USE_CONSUMABLE, (item_use & PN_ITEM_USED_FIELD) != 0);
        EV(PN_SKILL_USE_BATTLE_ITEM, (item_use & PN_ITEM_USED_BATTLE) != 0);
        /* 2026-10-29: a mart purchase completed: the "Here you go" acknowledgement, which only a successful
         * purchase reaches (item in the bag, money paid), starts waiting */
        EV(PN_SKILL_MART_BUY, c.waiting && c.ctx == PN_WAIT_SHOP_ACK && !(pwait && pctx == PN_WAIT_SHOP_ACK));
#undef EV
        /* ev &= ingame; rows outside the game hold their state */
        if (!c.ingame)
            return 0;
        set_row(st->prev, &c, pc_bits(&c, bits));
        if (!c.holding)
            for (j = 0; j < PN_OBS_PARTY_SIZE; j++)
                st->key[j] = c.key[j];
        copy_moves(st->moves, rec);
        st->tm = skill_tm(&c, st->tm);
    }
    return ev;
}
