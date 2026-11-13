// fireredsim worker: the new-game preset (start_preset.h). Compiled with the game's headers and
// flags; writes the live save block 1 directly (no game function runs, nothing else changes).
#include "global.h"
#include "constants/flags.h"
#include "constants/vars.h"

#include "start_preset.h"

uint16_t pn_preset_flag_max(void)
{
    return FLAGS_COUNT - 1;
}

uint16_t pn_preset_var_min(void)
{
    return VARS_START;
}

uint16_t pn_preset_var_max(void)
{
    return VARS_END;
}

void pn_start_preset_apply(const uint16_t *flags, int n_flags, const uint16_t *var_ids,
                           const uint16_t *var_values, int n_vars)
{
    int i;

    for (i = 0; i < n_flags; i++)
    {
        u16 id = flags[i];
        if (id == 0 || id >= FLAGS_COUNT)
            continue;
        gSaveBlock1Ptr->flags[id / 8] |= 1 << (id & 7);
    }
    for (i = 0; i < n_vars; i++)
    {
        u16 id = var_ids[i];
        if (id < VARS_START || id > VARS_END)
            continue;
        gSaveBlock1Ptr->vars[id - VARS_START] = var_values[i];
    }
}
