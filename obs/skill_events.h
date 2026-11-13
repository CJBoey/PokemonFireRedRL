/* fireredsim: the skill-retention events of reward v0.11 (+ v0.12's mart) from consecutive records (observer v5,
 * PnObs.skill_events_priv; docs/OBSERVATION.md "Changes in v5").
 *
 * The detection is reward v0.11's boolean event array before the cooldown (since v5 the reference
 * rl/pokerl/reward_ref.py SkillRefV011.events; the reward reads these bits instead of deriving them),
 * WITH its in-game gate: an event bit is set only on a record with PN_VALID_INGAME. The state is the
 * v0.11 reward's per-env skill state (skill_prev, skill_party, skill_moves, skill_tm); the caller (the
 * worker, one state per env) owns it. The functions read records only (and the game's constant item
 * table / TM move table), never game state, so they are pure functions of (state, record).
 *
 *   pn_skill_reset(st)        the next record initialises the state (a new game, a reset, a
 *                             snapshot restore: RewardV0.reset)
 *   pn_skill_record(st, rec)  the events of `rec` against the state, then the state becomes rec's
 *                             (RewardV0.step); the first record after pn_skill_reset initialises
 *                             the state from itself and reports 0. The value is
 *                             PnObs.skill_events_priv: bits 0-15 (1 << PN_SKILL_x); the bits
 *                             PN_SKILL_SUCCESS_SHIFT + x reserved for "succeeded" variants are 0
 *   pn_skill_record_ex(st, rec, item_use)
 *                             the same with the consumable uses the game reported during the step that
 *                             produced `rec` (fireredsim_port.h Port_TakeItemUse; PN_ITEM_USED_* bits, 0
 *                             none): bit PN_SKILL_USE_CONSUMABLE for a use outside battle (2026-10-28),
 *                             bit PN_SKILL_USE_BATTLE_ITEM for one in battle (2026-10-29).
 *                             pn_skill_record is pn_skill_record_ex with 0.
 *
 * `rec` must be a PRIVILEGED record (party PIDs); a visible record has no events (the worker leaves
 * skill_events_priv 0 in visible mode, as every _priv field).
 */
#ifndef FIREREDSIM_SKILL_EVENTS_H
#define FIREREDSIM_SKILL_EVENTS_H

#include <stdint.h>

#include "observation.h"

#ifdef __cplusplus
extern "C" {
#endif

/* bits of PnSkillState.prev[7] (reward.SKILL_IN_PC_STORAGE / SKILL_IN_PC_PLAYER / SKILL_IN_MART): the PC and
 * shop sessions in progress (set while their menu waits, cleared when the player is free on the field) */
#define PN_SKILL_IN_PC_STORAGE 1
#define PN_SKILL_IN_PC_PLAYER 2
#define PN_SKILL_IN_MART 4

typedef struct PnSkillState {
    uint64_t key[PN_OBS_PARTY_SIZE];       /* party identities (species << 32 | PID), held while a mon is in hand */
    uint16_t moves[PN_OBS_PARTY_SIZE * 4]; /* party moves of the previous record */
    uint16_t tm;                           /* move of the TM/HM last selected for a party mon (0 none) */
    uint8_t prev[8];                       /* previous record: screen, wait_ctx, waiting, menu_action, pocket of
                                              context_item, tile ahead, in_battle, PC / shop session bits */
    uint8_t valid;                         /* 0: the next record initialises the state */
    uint8_t _pad[5];
} PnSkillState;

void pn_skill_reset(PnSkillState *st);
uint32_t pn_skill_record(PnSkillState *st, const PnObs *rec);
/* pn_skill_record_ex's `item_use`: a consumable used outside battle / in battle during the step */
#define PN_ITEM_USED_FIELD 1u
#define PN_ITEM_USED_BATTLE 2u
uint32_t pn_skill_record_ex(PnSkillState *st, const PnObs *rec, unsigned item_use);

/* reward.ITEM_POCKET / reward.TM_MOVE: the bag pocket of an item id (POCKET_*; 0 for ITEM_NONE entries
 * and ids >= ITEMS_COUNT) and the move of a TM/HM item (0 for other items), from the game's tables */
uint8_t pn_skill_item_pocket(uint16_t item);
uint16_t pn_skill_tm_move(uint16_t item);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FIREREDSIM_SKILL_EVENTS_H */
