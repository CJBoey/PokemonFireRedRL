/* fireredsim: the moves party mons gained during a step (observer v7, PnObs.learned_move_priv;
 * docs/OBSERVATION.md "Changes in v7").
 *
 * The worker keeps one state per env: the party of the env's last processed record (identities and moves).
 * A STEP's record is compared with it slot by slot, the mons matched by identity (personality and OT id, the
 * record's pid_priv / ot_id_priv: not the species, so a mon that evolves during the step keeps its identity),
 * so a reorder, a deposit / withdrawal, a catch or a trade never reports a learn. A pure function of (state,
 * record): it reads records only, never game state.
 *
 *   pn_learn_reset(st)                 the next record initialises the state (RESET / RESTORE ops)
 *   pn_learn_record(st, rec, out)      a STEP's record: out[i] = the move party slot i's mon gained since the
 *                                      state's record (0 none), then the state becomes rec's. The first record
 *                                      after pn_learn_reset initialises the state and reports 0. A record without
 *                                      PN_VALID_INGAME (party not filled: title screen, Quest Log playback, Hall
 *                                      of Fame) reports 0 and empties the state: the next in-game record
 *                                      initialises it (a CONTINUE after a soft reset reloads an older party)
 *   pn_learn_init(st, rec)             a record of another op (OBSERVE / SNAPSHOT / RESET / RESTORE): initialises
 *                                      an empty state from an in-game record (reports nothing); a valid state is
 *                                      left as it was
 *
 * "Gained": a nonzero move of the current record's slot that is not among the matched previous mon's four moves.
 * A mon that gained two moves in one step reports the one in the lower move slot (moves[0..3] order). A slot whose
 * mon has no match in the previous party (new identity, empty before) reports nothing.
 *
 * `rec` must be a PRIVILEGED record (the identities); the worker computes nothing in visible mode.
 */
#ifndef FIREREDSIM_LEARNED_MOVES_H
#define FIREREDSIM_LEARNED_MOVES_H

#include <stdint.h>

#include "observation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PnLearnState {
    uint64_t key[PN_OBS_PARTY_SIZE];       /* ot_id << 32 | pid of each slot */
    uint16_t moves[PN_OBS_PARTY_SIZE * 4]; /* the slots' moves */
    uint8_t present;                       /* bit i: slot i holds a mon (species != 0) */
    uint8_t valid;                         /* 0: the next record initialises the state */
    uint8_t _pad[6];
} PnLearnState;

void pn_learn_reset(PnLearnState *st);
void pn_learn_record(PnLearnState *st, const PnObs *rec, uint16_t out[PN_OBS_PARTY_SIZE]);
/* v8: pn_learn_record, and *replaced (may be NULL) = bit i set when out[i] != 0 and the matched previous mon knew
 * four moves (the gained move replaced one: PnObs.learned_replaced_priv) */
void pn_learn_record_ex(PnLearnState *st, const PnObs *rec, uint16_t out[PN_OBS_PARTY_SIZE], uint8_t *replaced);
void pn_learn_init(PnLearnState *st, const PnObs *rec);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FIREREDSIM_LEARNED_MOVES_H */
