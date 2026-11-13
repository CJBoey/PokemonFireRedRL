/* fireredsim: the moves party mons gained during a step (obs/learned_moves.h; observer v7). */
#include <string.h>

#include "observation.h"
#include "learned_moves.h"

static void load(PnLearnState *st, const PnObs *rec)
{
    int j, k;
    st->present = 0;
    for (j = 0; j < PN_OBS_PARTY_SIZE; j++) {
        const PnMon *m = &rec->party[j];
        st->key[j] = ((uint64_t)m->ot_id_priv << 32) | (uint64_t)m->pid_priv;
        if (m->species != 0)
            st->present |= (uint8_t)(1u << j);
        for (k = 0; k < 4; k++)
            st->moves[j * 4 + k] = m->moves[k];
    }
    st->valid = 1;
}

void pn_learn_reset(PnLearnState *st)
{
    st->valid = 0;
}

void pn_learn_init(PnLearnState *st, const PnObs *rec)
{
    if (!st->valid && (rec->valid & PN_VALID_INGAME))
        load(st, rec);
}

void pn_learn_record(PnLearnState *st, const PnObs *rec, uint16_t out[PN_OBS_PARTY_SIZE])
{
    pn_learn_record_ex(st, rec, out, NULL);
}

void pn_learn_record_ex(PnLearnState *st, const PnObs *rec, uint16_t out[PN_OBS_PARTY_SIZE], uint8_t *replaced)
{
    int i, j, k, q, changed = 0;
    uint8_t present = 0;
    memset(out, 0, PN_OBS_PARTY_SIZE * sizeof(out[0]));
    if (replaced)
        *replaced = 0;
    if (!(rec->valid & PN_VALID_INGAME)) {
        /* party not filled (title screen, Quest Log playback, Hall of Fame): the game left play; the party it comes
         * back with (a CONTINUE after a soft reset loads the save) is a new baseline, not a change */
        st->valid = 0;
        return;
    }
    if (!st->valid) {
        load(st, rec);
        return;
    }
    for (i = 0; i < PN_OBS_PARTY_SIZE; i++) {
        const PnMon *m = &rec->party[i];
        uint64_t key;
        const uint16_t *prev = NULL;
        if (m->species == 0)
            continue;
        present |= (uint8_t)(1u << i);
        key = ((uint64_t)m->ot_id_priv << 32) | (uint64_t)m->pid_priv;
        /* the same slot first (no reorder: the common case), then the others */
        if ((st->present & (1u << i)) && st->key[i] == key) {
            prev = &st->moves[i * 4];
            if (memcmp(prev, m->moves, sizeof(m->moves)) == 0)
                continue; /* the same mon with the same moves (almost every record): one 8-byte compare */
        } else {
            for (j = 0; j < PN_OBS_PARTY_SIZE; j++) {
                if ((st->present & (1u << j)) && st->key[j] == key) {
                    prev = &st->moves[j * 4];
                    break;
                }
            }
        }
        changed = 1;
        if (prev == NULL)
            continue; /* not in the previous party: caught, withdrawn, traded in */
        for (k = 0; k < 4; k++) {
            uint16_t mv = m->moves[k];
            int known = 0;
            if (mv == 0)
                continue;
            for (q = 0; q < 4; q++)
                known |= prev[q] == mv;
            if (!known) {
                out[i] = mv; /* the lowest move slot of a mon that gained two */
                if (replaced && prev[0] && prev[1] && prev[2] && prev[3])
                    *replaced |= (uint8_t)(1u << i); /* v8: it knew four moves: one was forgotten */
                break;
            }
        }
    }
    if (changed || present != st->present)
        load(st, rec); /* else the state already is rec's party */
}
