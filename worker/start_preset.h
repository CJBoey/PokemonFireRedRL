/* fireredsim worker: the new-game preset (EnvPoolOptions::preset_flags / preset_vars,
 * fireredsim-worker --preset-flag / --preset-var, docs/IPC_PROTOCOL.md "New-game preset").
 * Compiled with the game's headers and flags (pn_add_game_reader_library,
 * cmake/GameReader.cmake) for the save block layout and the flag / var constants.
 * Unlike the other reader libraries pn_start_preset_apply WRITES game memory: the
 * EnvPool calls it on the live state regions between frames, right after a script
 * template's replay (bedroom included), before the start state is cached. */
#ifndef PN_START_PRESET_H
#define PN_START_PRESET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ids a preset accepts (from the game's headers): flags 1 .. FLAGS_COUNT - 1 (the saved
 * flags of gSaveBlock1Ptr->flags; 0 is no flag, SPECIAL_FLAGS_START and up are not saved),
 * vars VARS_START .. VARS_END (gSaveBlock1Ptr->vars). */
uint16_t pn_preset_flag_max(void); /* FLAGS_COUNT - 1 */
uint16_t pn_preset_var_min(void);  /* VARS_START */
uint16_t pn_preset_var_max(void);  /* VARS_END */

/* Sets flags[0..n_flags) and vars var_ids[i] = var_values[i] in the live save block 1, as
 * FlagSet / VarSet do outside Quest Log playback (tools/synth synth_op_flag / synth_op_var),
 * in that order (a var listed twice: the last value wins). Ids must be in the ranges above
 * (the caller validates them; out-of-range ids are skipped). */
void pn_start_preset_apply(const uint16_t *flags, int n_flags, const uint16_t *var_ids,
                           const uint16_t *var_values, int n_vars);

#ifdef __cplusplus
}
#endif

#endif /* PN_START_PRESET_H */
