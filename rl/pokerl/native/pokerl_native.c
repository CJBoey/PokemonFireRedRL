/* Native reward, action masks and trail window of the training env (pokerl/reward.py has the terms, mask.py the
 * mask rules, env.py the trail). Built 64-bit by rl/pokerl/native/build.py (outside the -m32 game CMake tree).
 *
 * Every record offset comes from the caller (`off`: int32 byte offsets laid out as gen_spec.h's G_* / B_* / R_*
 * positions, computed by pokerl.native.spec.offsets from the record dtype), every table from the Python modules
 * (pn_set_tables). The float arithmetic is fixed: float32 or double then a cast as written, the reward a pairwise sum
 * of the terms in a fixed order (row_sum). Build without -ffast-math and with -ffp-contract=off so that no operation
 * is fused or reordered.
 */
#include <emmintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "gen_spec.h"

#define EXPORT __attribute__((visibility("default")))

_Static_assert(N_SKILL == 16 && SKILL_EVENT_MASK == (1 << N_SKILL) - 1 && N_SKILL_EVENTS == 1, "skill events");
/* skill-retention cooldown per event in steps (reward.SKILL_COOLDOWNS) */
static const int64_t skill_cooldown[N_SKILL] = SKILL_COOLDOWNS;
/* each term's column in the reward's summation layout (reward.TERM_SUM_COL) */
static const int term_sum_col[N_TERMS] = TERM_SUM_COL;
_Static_assert(N_TERMS == 19 && OBJ_TERM0 == 13 && INTERACT_TERM == 17 && MOVE_LEARN_TERM == 18 && SUM_COLS == 26,
               "terms");
_Static_assert(N_INTERACT_KIND == 1 && N_INTERACT_MAP == 2 && N_INTERACT_LOCAL_ID == 1 && N_INTERACT_XY == 2,
               "interaction fields");
_Static_assert(INTERACT_N > 0 && INTERACT_BYTES * 8 >= INTERACT_N && INTERACT_OBJECT == 1 && INTERACT_SIGN == 2 &&
               INTERACT_HIDDEN_ITEM == 3 && INTERACT_SIGN_WALK == 4, "interactions");
_Static_assert(N_FACING == 1 && N_IN_DIALOG == 1 && N_WAITING == 1 && N_WAIT_CTX == 1 && N_SCREEN == 1, "puzzle fields");
_Static_assert(OBJ_N > 0 && OBJ_BYTES * 8 >= OBJ_N, "object bitmap");
_Static_assert(OBJ_TRASH_CAN == 0 && OBJ_MANSION_SWITCH == 1 && OBJ_SPINNER == 2 && OBJ_WARP_PAD == 3, "object kinds");
_Static_assert(N_P_SPECIES == N_PARTY && N_P_IS_EGG == N_PARTY && N_P_LEVEL == N_PARTY && N_P_HP == N_PARTY &&
               N_P_PID == N_PARTY, "party");
_Static_assert(KEY_BYTES * 8 >= ITEMS_COUNT, "key item bitmap");
_Static_assert(FLAG_BYTES % 8 == 0 && SCENE_BLOCK % 8 == 0, "u64 blocks");

/* ---- tables (numpy arrays owned by the Python module) ------------------------------------------------ */

struct pn_tables {
    const int64_t *map_index;      /* [65536] map_group * 256 + map_num -> dense map id (reward.MAP_INDEX) */
    const float *flag_matrix;      /* [FLAGS_COUNT, 5] flags, trainers, badges, elite4 weights, earned (weight > 0) */
    const uint8_t *hm_learn;       /* [65536] species -> the HMs it can learn (bits) */
    const int32_t *scene_cols;     /* [N_SCENE] */
    const int64_t *map_w;          /* [MAP_COUNT + 1] */
    const int64_t *map_h;
    const int64_t *map_off;
    const uint16_t *action_buttons; /* [NUM_ACTIONS] */
    const float *skill_weights;    /* [N_SKILL]: per-event skill reward (reward.SKILL_WEIGHTS) */
    const int64_t *obj_t;          /* [OBJ_N] global tile index of each object tile, sorted (reward.OBJ_T) */
    const uint8_t *obj_kind;       /* [OBJ_N] OBJ_* kind */
    const uint8_t *obj_end_src;    /* [OBJ_N] OBJ_END_*: 0 none, 1 a seen-flag bit, 2 a key item bit */
    const int32_t *obj_end_bit;    /* [OBJ_N] the bit (flag index / item id) */
    const uint8_t *obj_can_mask;   /* [OBJ_BYTES] bits of the trash can entries (cleared on a new switch draw) */
    const uint8_t *map_obj_flags;  /* [MAP_COUNT + 1] 1: object tiles on the map (fast path) */
    const float *obj_weights;      /* [4] per kind reward (trash, mansion, spinner, warp pad) */
    const float *var_weights;      /* [N_SCENE] per scene variable weight of the vars term (reward.VAR_WEIGHTS) */
    const int64_t *interact_keys;  /* [INTERACT_N] interaction keys, sorted (reward.INTERACT_KEYS, interact_key) */
    const float *interact_weights; /* [1] per interaction key first started in a life (reward.INTERACT_WEIGHTS) */
    const int64_t *cap_bytes;      /* [8]: the seen bytes holding ladder flags (first N_CAP_BYTES) */
    const int32_t *opp_table;      /* [8, 256]: per byte value, min over its ladder entries of the opponent key */
    const int32_t *opp_size;       /* [OPP_IDX + 1]: party size K per ladder index (reward.OPP_SIZE) */
    const uint16_t *hm_moves;      /* [8]: move id of HM bit i (0 unused) */
};

static struct pn_tables TB;
static int TB_SET = 0;

EXPORT int64_t pn_spec_hash(void) { return SPEC_HASH; }
EXPORT int pn_num_offsets(void) { return N_OFFSETS; }

EXPORT void pn_set_tables(const struct pn_tables *t) {
    TB = *t;
    TB_SET = 1;
}

/* ---- record access ------------------------------------------------------------------------------------ */

static inline uint8_t rd8(const uint8_t *r, int32_t o) { return r[o]; }
static inline uint16_t rd16(const uint8_t *r, int32_t o) { uint16_t v; memcpy(&v, r + o, 2); return v; }
static inline int16_t rdi16(const uint8_t *r, int32_t o) { int16_t v; memcpy(&v, r + o, 2); return v; }
static inline uint32_t rd32(const uint8_t *r, int32_t o) { uint32_t v; memcpy(&v, r + o, 4); return v; }

/* prefetch the cache lines of record `r` listed in pf (spec.prefetch_offsets) */
#ifndef PF_REC
#define PF_REC 8 /* records ahead (reward parse pass) */
#endif
#ifndef PF_ST
#define PF_ST 8 /* envs ahead (reward state lines) */
#endif
static inline void prefetch_record(const uint8_t *r, const int32_t *pf, int64_t npf) {
    for (int64_t k = 0; k < npf; k++) __builtin_prefetch(r + pf[k], 0, 3);
}

/* ---- action masks -------------------------------------------------------------------------------------- */

/* mask.py: A on a start-menu entry of START_MENU_NO_A (bit set), in the save prompt, or on a RELEASE entry of the
 * Pokemon Storage System (menu_action) */
static inline int forbids_a(const uint8_t *r, const int32_t *off) {
    int ctx = rd8(r, off[G_WAIT_CTX]);
    int release = rd8(r, off[G_MENU_ACTION]) == MENU_ACTION_RELEASE;
    if (ctx == WAIT_START_MENU) {
        unsigned c = rd8(r, off[G_START_CURSOR]);
        return (c < 32u && ((START_MENU_NO_A_BITS >> c) & 1u)) || release;
    }
    return ctx == WAIT_SAVE_YESNO || ctx == WAIT_SAVE_TEXT || release;
}

/* the action mask: NOOP only while the game does not wait for input; else the buttons it reads (NOOP excluded), A
 * not where forbids_a; NOOP when nothing else is left */
EXPORT int pn_action_mask(const uint8_t *rec, int64_t n, int64_t rs, const int32_t *off, uint8_t *mask) {
    if (!TB_SET) return -1;
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *r = rec + i * rs;
        uint8_t *m = mask + i * NUM_ACTIONS;
        uint16_t keys = rd16(r, off[G_INPUT_KEYS]);
        int waiting = rd8(r, off[G_WAITING]) != 0;
        int forbid = forbids_a(r, off);
        int any = 0;
        for (int a = 1; a < NUM_ACTIONS; a++) {
            int v = (keys & TB.action_buttons[a]) != 0 && waiting;
            if (a == A_INDEX) v = v && !forbid;
            m[a] = (uint8_t)v;
            any |= v;
        }
        m[0] = !any;
    }
    return 0;
}

/* the mask fallback's mask: every action, A not where forbids_a */
EXPORT int pn_fallback_mask(const uint8_t *rec, int64_t n, int64_t rs, const int32_t *off, uint8_t *mask) {
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *r = rec + i * rs;
        uint8_t *m = mask + i * NUM_ACTIONS;
        for (int a = 0; a < NUM_ACTIONS; a++) m[a] = 1;
        m[A_INDEX] = !forbids_a(r, off);
    }
    return 0;
}

/* ---- reward -------------------------------------------------------------------------------------------- */

struct pn_reward_state {
    int64_t num_envs;
    uint8_t *seen;          /* [n, FLAG_BYTES] flags seen this life */
    uint8_t *tile_bm;       /* [n, TILE_BYTES] tiles visited this life */
    uint32_t *tile_count;   /* [TILE_COUNT] the run's first visits (shared by all envs) */
    uint8_t *map_seen;      /* [n, MAP_COUNT + 1] numpy bool */
    int32_t *cap;           /* the next opponent's ace level */
    int32_t *opp_size;      /* K, the next opponent's party size */
    int64_t *best;          /* the life's maximum of the capped level sum */
    int32_t *cover;
    int32_t *known;
    uint8_t *fainted;       /* numpy bool */
    uint16_t *var_max;      /* [n, N_SCENE] */
    uint8_t *var_last;      /* [n, SCENE_BLOCK] */
    uint8_t *key_seen;      /* [n, KEY_BYTES] key item ids seen this life (bit i of byte j = item 8 j + i) */
    int64_t *skill_step;    /* [1] the skill term's step counter (every env steps on every call) */
    int64_t *skill_next;    /* [n, N_SKILL] the first step at which each event may pay again (0: allowed) */
    int64_t *maps_visited;
    int64_t *flags_earned;
    int64_t *scratch;       /* [9, n] int64 work space of pn_reward_step */
    uint8_t *obj_bm;        /* [n, OBJ_BYTES] object tiles used this life (bit = entry of OBJ_T) */
    int64_t *prev_obj;      /* [n] the object entry faced at the previous step while free on the field, -1 none */
    int64_t *prev_t;        /* [n] the previous step's tile index, -1 none */
    uint32_t *trash_key;    /* [n] VAR_TEMP_0 | VAR_TEMP_1 << 16 of the trash cans' current attempt */
    uint8_t *interact_bm;   /* [n, INTERACT_BYTES] keys started this life (bit = entry of INTERACT_KEYS) */
    uint16_t *trail_t;      /* [n, TILE_COUNT] the move number (low 16 bits) of each tile's last entry */
    uint32_t *trail_now;    /* [n] the move counter */
    int64_t *trail_last;    /* [n] the tile of the last move, -1 none */
    uint32_t *trail_epoch;  /* [n] trail_now >> TRAIL_EPOCH_SHIFT at the last refresh */
    uint8_t *trail_view;    /* [n, TRAIL_VIEW] the last trail window (unmasked: 0 only outside the grid), rows of 16 */
    int64_t *trail_vkey;    /* [n, 4] map, x, y, trail_now of trail_view (map -1: none) */
    uint8_t *hm_seen;       /* [n, 2] the HM bits seen this life (known moves, coverage) */
    uint32_t *lv_pid;       /* [n, N_PARTY] the non-egg party mons' personalities at the last in-game record */
};
#define TRAIL_VIEW (SEEN_ROWS * 16)

/* sizeof(struct pn_reward_state): the Python side refuses a library built with another layout */
EXPORT int64_t pn_reward_state_size(void) { return (int64_t)sizeof(struct pn_reward_state); }

/* max(N, S) / S of a tile with c earlier first visits (N = c + 1, S = TILE_COUNT_SCALE); the count bonus is
 * TILE_COUNT_REWARD / sqrt of it */
static inline float tile_count_q(uint32_t c) {
    const float s = (float)TILE_COUNT_SCALE, v = (float)c + 1.0f;
    return (v > s ? v : s) / s;
}

/* the trail refresh of env e: when its move counter entered a new epoch, every stamp of age >= TRAIL_K gets age
 * TRAIL_REFRESH_AGE (16-bit stamps never alias; reward.py "Trail"); ~60 us once per 2^TRAIL_EPOCH_SHIFT moves */
static void trail_refresh(struct pn_reward_state *st, int64_t e) {
    const uint32_t ep = st->trail_now[e] >> TRAIL_EPOCH_SHIFT;
    if (ep == st->trail_epoch[e]) return;
    st->trail_epoch[e] = ep;
    const uint16_t now16 = (uint16_t)st->trail_now[e], old = (uint16_t)(now16 - TRAIL_REFRESH_AGE);
    uint16_t *s = st->trail_t + e * TILE_COUNT;
    const __m128i nowv = _mm_set1_epi16((short)now16), oldv = _mm_set1_epi16((short)old);
    const __m128i bias = _mm_set1_epi16((short)0x8000), lim = _mm_set1_epi16((short)((TRAIL_K - 1) ^ 0x8000));
    int64_t i = 0;
    for (; i + 8 <= TILE_COUNT; i += 8) { /* stale: (now - s) as unsigned >= TRAIL_K (a biased signed compare) */
        const __m128i v = _mm_loadu_si128((const __m128i *)(s + i));
        const __m128i stale = _mm_cmpgt_epi16(_mm_xor_si128(_mm_sub_epi16(nowv, v), bias), lim);
        _mm_storeu_si128((__m128i *)(s + i), _mm_or_si128(_mm_and_si128(stale, oldv), _mm_andnot_si128(stale, v)));
    }
    for (; i < TILE_COUNT; i++)
        if ((uint16_t)(now16 - s[i]) >= TRAIL_K) s[i] = old;
}

/* one record's position */
struct parsed {
    int ingame, warp;
    int64_t map;    /* MAP_INDEX of the record */
    int64_t m, t;   /* dense map id (0 outside the game), global tile index */
    int64_t x, y;
    int ok;         /* the record marks a tile: in the game, inside the map grid, not in a warp */
};

static inline void parse(const uint8_t *r, const int32_t *off, struct parsed *p) {
    p->ingame = (rd8(r, off[G_VALID]) & VALID_INGAME) != 0;
    p->map = TB.map_index[(int)rd8(r, off[G_MAP_GROUP]) * 256 + rd8(r, off[G_MAP_NUM])];
    p->warp = rd8(r, off[G_FIELD_STATE]) == FIELD_WARP;
    int64_t x = rdi16(r, off[G_XY]), y = rdi16(r, off[G_XY + 1]);
    int64_t m = p->ingame ? p->map : 0;
    int64_t w = TB.map_w[m], h = TB.map_h[m];
    p->m = m;
    p->x = x;
    p->y = y;
    p->ok = m > 0 && x >= 0 && x < w && y >= 0 && y < h && !p->warp;
    p->t = p->ok ? TB.map_off[m] + y * w + x : 0;
}

/* ---- the puzzle terms (reward.py term 8) ----------------------------------------------------------------- */

/* the OBJ_T entry of a global tile index (-1 none; a -1 input gives -1) */
static inline int64_t obj_index(int64_t t) {
    int64_t lo = 0, hi = OBJ_N;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (TB.obj_t[mid] < t) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= OBJ_N) lo = OBJ_N - 1;
    return TB.obj_t[lo] == t ? lo : -1;
}

/* the entry of the tile the player faces while free on the field (-1 none) */
static inline int64_t ahead_entry(const uint8_t *r, const int32_t *off, const struct parsed *p) {
    static const int64_t dx[5] = {0, 0, 0, -1, 1}, dy[5] = {0, 1, -1, 0, 0};
    if (!p->ingame || !p->ok) return -1;
    if (rd8(r, off[G_WAITING]) == 0 || rd8(r, off[G_WAIT_CTX]) != WAIT_OW_FREE ||
        rd8(r, off[G_SCREEN]) != SCREEN_OVERWORLD)
        return -1;
    const int f = rd8(r, off[G_FACING]);
    if (f < 1 || f > 4) return -1;
    const int64_t ax = p->x + dx[f], ay = p->y + dy[f], w = TB.map_w[p->m], h = TB.map_h[p->m];
    if (ax < 0 || ax >= w || ay < 0 || ay >= h) return -1;
    return obj_index(TB.map_off[p->m] + ay * w + ax);
}

/* entry k of env e pays unless ended (its flag seen / key item held) or already used; marks it */
static inline int claim(struct pn_reward_state *st, int64_t e, int64_t k) {
    const uint8_t src = TB.obj_end_src[k];
    const int32_t bit = TB.obj_end_bit[k];
    if (src == OBJ_END_FLAG && ((st->seen[e * FLAG_BYTES + (bit >> 3)] >> (bit & 7)) & 1)) return 0;
    if (src == OBJ_END_KEY_ITEM && ((st->key_seen[e * KEY_BYTES + (bit >> 3)] >> (bit & 7)) & 1)) return 0;
    uint8_t *bm = st->obj_bm + e * OBJ_BYTES + (k >> 3);
    const uint8_t b = (uint8_t)(1u << (k & 7));
    if (*bm & b) return 0;
    *bm |= b;
    return 1;
}

/* ---- the overworld interactions (reward.py term 9) -------------------------------------------------------- */

/* reward.interact_key of a record reporting the interaction kind `kind` (!= 0) looked up in INTERACT_KEYS: its
 * entry, -1 for none (an unknown kind, a key of no listed object / background event) */
static inline int64_t interact_entry(const uint8_t *r, const int32_t *off, uint8_t kind) {
    const int64_t mk = (int64_t)rd8(r, off[G_INTERACT_MAP]) << 8 | rd8(r, off[G_INTERACT_MAP + 1]);
    int64_t key;
    if (kind == INTERACT_OBJECT)
        key = (int64_t)INTERACT_TAG_OBJECT << 48 | mk << 32 | rd8(r, off[G_INTERACT_LOCAL_ID]);
    else if (kind == INTERACT_SIGN || kind == INTERACT_HIDDEN_ITEM || kind == INTERACT_SIGN_WALK)
        key = (int64_t)INTERACT_TAG_BG << 48 | mk << 32 | (int64_t)(uint16_t)rdi16(r, off[G_INTERACT_XY]) << 16 |
              (int64_t)(uint16_t)rdi16(r, off[G_INTERACT_XY + 1]);
    else
        return -1;
    int64_t lo = 0, hi = INTERACT_N;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (TB.interact_keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo < INTERACT_N && TB.interact_keys[lo] == key ? lo : -1;
}

/* ---- the levels and HM terms (reward.py term 3) ------------------------------------------------------------ */

/* the next opponent of one env's seen bytes: the min over the ladder of (OPP_KEY_AFTER if beaten else the entry's
 * key ace << OPP_SHIFT | index), as per-byte tables (spec.tables opp_table); cap = key >> OPP_SHIFT, K =
 * opp_size[key & OPP_IDX] */
static inline void next_opponent(const uint8_t *seen, int32_t *cap, int32_t *k) {
    _Static_assert(N_CAP_BYTES >= 1 && N_CAP_BYTES <= 8, "cap bytes");
    int32_t best = TB.opp_table[seen[TB.cap_bytes[0]]];
    for (int b = 1; b < N_CAP_BYTES; b++) {
        int32_t v = TB.opp_table[b * 256 + seen[TB.cap_bytes[b]]];
        best = v < best ? v : best;
    }
    *cap = best >> OPP_SHIFT;
    *k = TB.opp_size[best & OPP_IDX];
}

/* the HM moves known by the party (eggs and empty slots included): bit i set iff some move equals hm_moves[i]; rows
 * of 4 packed u16 (spec.ROWS P_MOVES) */
static inline unsigned hm_known(const uint8_t *r, const int32_t *off) {
    _Static_assert(N_PARTY == 6 && N_MOVES == 4 && N_HM <= 8, "move lanes");
    const uint8_t *b = r + off[R_P_MOVES];
    const int32_t so = off[R_P_MOVES + 1];
    __m128i v[3];
    for (int k = 0; k < 3; k++) {
        uint64_t w0, w1;
        memcpy(&w0, b + (2 * k) * so, 8);
        memcpy(&w1, b + (2 * k + 1) * so, 8);
        v[k] = _mm_unpacklo_epi64(_mm_cvtsi64_si128((long long)w0), _mm_cvtsi64_si128((long long)w1));
    }
    unsigned kn = 0;
    for (int h = 0; h < N_HM; h++) {
        __m128i m = _mm_set1_epi16((short)TB.hm_moves[h]);
        __m128i e = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi16(v[0], m), _mm_cmpeq_epi16(v[1], m)),
                                 _mm_cmpeq_epi16(v[2], m));
        kn |= (unsigned)(_mm_movemask_epi8(e) != 0) << h;
    }
    return kn;
}

/* one record's party: the non-egg mons' levels sorted descending into lv (0 for the other slots), the HM coverage
 * bits (the HMs the party can learn) and the all-fainted flag */
static inline void party_scan(const uint8_t *r, const int32_t *off, int32_t lv[N_PARTY], unsigned *cover,
                              int *fainted) {
    unsigned cv = 0;
    int anymon = 0, alive = 0;
    for (int j = 0; j < N_PARTY; j++) {
        uint16_t sp = rd16(r, off[G_P_SPECIES + j]);
        int mon = sp != 0 && rd8(r, off[G_P_IS_EGG + j]) == 0 && sp != SPECIES_EGG;
        lv[j] = 0;
        if (mon) {
            lv[j] = rd8(r, off[G_P_LEVEL + j]);
            cv |= TB.hm_learn[sp];
            anymon = 1;
            if (rd16(r, off[G_P_HP + j]) > 0) alive = 1;
        }
    }
    for (int j = 1; j < N_PARTY; j++) { /* insertion sort, descending */
        int32_t v = lv[j];
        int i = j - 1;
        for (; i >= 0 && lv[i] < v; i--) lv[i + 1] = lv[i];
        lv[i + 1] = v;
    }
    *cover = cv;
    *fainted = anymon && !alive;
}

/* S: the sum of min(level, cap) over the k highest levels (k clipped to 0..N_PARTY) */
static inline int64_t top_sum(const int32_t lv[N_PARTY], int32_t cap, int32_t k) {
    k = k < 0 ? 0 : (k > N_PARTY ? N_PARTY : k);
    int64_t s = 0;
    for (int j = 0; j < k; j++) s += lv[j] < cap ? lv[j] : cap;
    return s;
}

/* one record's non-egg party mons' personalities, 0 for the other slots */
static inline void pid_scan(const uint8_t *r, const int32_t *off, uint32_t pid[N_PARTY]) {
    for (int j = 0; j < N_PARTY; j++) {
        uint16_t sp = rd16(r, off[G_P_SPECIES + j]);
        int mon = sp != 0 && rd8(r, off[G_P_IS_EGG + j]) == 0 && sp != SPECIES_EGG;
        pid[j] = mon ? rd32(r, off[G_P_PID + j]) : 0;
    }
}

/* for one in-game record: whether a non-egg party mon's personality was not among env e's lv_pid (the previous
 * in-game record's: a mon entered the party), then lv_pid moves to this record */
static inline int imported(struct pn_reward_state *st, int64_t e, const uint8_t *r, const int32_t *off) {
    uint32_t pid[N_PARTY];
    pid_scan(r, off, pid);
    uint32_t *lp = st->lv_pid + e * N_PARTY;
    int any = 0;
    for (int j = 0; j < N_PARTY; j++) {
        if (!pid[j]) continue;
        int seen = 0;
        for (int q = 0; q < N_PARTY; q++) seen |= lp[q] == pid[j];
        any |= !seen;
    }
    for (int j = 0; j < N_PARTY; j++) lp[j] = pid[j];
    return any;
}

/* one record's party: S, the coverage and known HM counts, the all-fainted flag */
static inline void party(const uint8_t *r, const int32_t *off, int32_t cap, int32_t k, int64_t *s, int32_t *cover,
                         int32_t *known, int *fainted) {
    int32_t lv[N_PARTY];
    unsigned cv;
    party_scan(r, off, lv, &cv, fainted);
    *s = top_sum(lv, cap, k);
    *cover = __builtin_popcount(cv & 0xFF);
    *known = __builtin_popcount(hm_known(r, off) & 0xFF);
}

/* the key item ids of the record's key-items pocket as bits (0 and ids >= ITEMS_COUNT ignored) */
static inline void key_bits(const uint8_t *r, const int32_t *off, uint8_t *bm) {
    for (int s = 0; s < N_KEY_ID; s++) {
        unsigned id = rd16(r, off[G_KEY_ID + s]);
        if (id > 0 && id < ITEMS_COUNT) bm[id >> 3] |= (uint8_t)(1u << (id & 7));
    }
}

static inline void scene_vars(const uint8_t *vb, uint16_t *out) {
    for (int k = 0; k < N_SCENE; k++) memcpy(out + k, vb + 2 * TB.scene_cols[k], 2);
}

/* NativeReward.reset(obs, ids): rows of `rec` are the first records of envs ids[0..k) (a new life or a rewind
 * restore): every per-life state from them, nothing paid. Phased (duplicate ids: last wins; the tile bit is assigned
 * into the zeroed bitmap). */
EXPORT int pn_reward_reset(struct pn_reward_state *st, const int64_t *ids, int64_t k, const uint8_t *rec, int64_t rs,
                           const int32_t *off) {
    if (!TB_SET) return -1;
    const int64_t MS = MAP_COUNT + 1;
    for (int64_t j = 0; j < k; j++) {
        const uint8_t *r = rec + j * rs;
        int64_t e = ids[j];
        memcpy(st->seen + e * FLAG_BYTES, r + off[B_FLAGS], FLAG_BYTES);
        memcpy(st->var_last + e * SCENE_BLOCK, r + off[B_SCENE], SCENE_BLOCK);
        scene_vars(r + off[B_SCENE], st->var_max + e * N_SCENE);
        memset(st->tile_bm + e * TILE_BYTES, 0, TILE_BYTES);
        memset(st->map_seen + e * MS, 0, MS);
        memset(st->key_seen + e * KEY_BYTES, 0, KEY_BYTES);
        key_bits(r, off, st->key_seen + e * KEY_BYTES);
    }
    for (int64_t j = 0; j < k; j++) {
        struct parsed p;
        parse(rec + j * rs, off, &p);
        int64_t e = ids[j];
        if (p.ok) st->tile_bm[e * TILE_BYTES + (p.t >> 3)] = (uint8_t)(1u << (p.t & 7));
        st->map_seen[e * MS + p.m] = p.m > 0;
    }
    /* the trail: a fresh one (reward.py "Trail") */
    for (int64_t j = 0; j < k; j++) st->trail_now[ids[j]] += (uint32_t)TRAIL_RESET;
    for (int64_t j = 0; j < k; j++) {
        struct parsed p;
        parse(rec + j * rs, off, &p);
        const int64_t e = ids[j];
        if (p.ok) st->trail_t[e * TILE_COUNT + p.t] = (uint16_t)st->trail_now[e];
        st->trail_last[e] = p.ok ? p.t : -1;
        trail_refresh(st, e);
    }
    for (int64_t j = 0; j < k; j++) st->map_seen[ids[j] * MS] = 0;
    for (int64_t j = 0; j < k; j++) {
        const uint8_t *r = rec + j * rs;
        int64_t e = ids[j];
        struct parsed p;
        parse(r, off, &p);
        int32_t cap, kk;
        next_opponent(st->seen + e * FLAG_BYTES, &cap, &kk);
        int64_t s;
        int32_t cover, known;
        int fainted;
        party(r, off, cap, kk, &s, &cover, &known, &fainted);
        {   /* the first record's HM bits, no pay */
            int32_t lv[N_PARTY];
            unsigned cv;
            int fa;
            party_scan(r, off, lv, &cv, &fa);
            st->hm_seen[2 * e] = (uint8_t)hm_known(r, off);
            st->hm_seen[2 * e + 1] = (uint8_t)cv;
        }
        pid_scan(r, off, st->lv_pid + e * N_PARTY); /* the first record's mons */
        st->cap[e] = cap;
        st->opp_size[e] = kk;
        st->best[e] = s;
        st->cover[e] = cover;
        st->known[e] = known;
        st->fainted[e] = (uint8_t)fainted;
        st->maps_visited[e] = p.m > 0;
        st->flags_earned[e] = 0;
        /* skill term: no cooldowns */
        for (int q = 0; q < N_SKILL; q++) st->skill_next[e * N_SKILL + q] = 0;
        /* puzzles: a fresh bitmap; the pad / spinner under the player marked without pay */
        memset(st->obj_bm + e * OBJ_BYTES, 0, OBJ_BYTES);
        st->trash_key[e] = rd32(r, off[B_VARS]);
        const int64_t cur_t = p.ok ? p.t : -1;
        const int64_t own = obj_index(cur_t);
        if (own >= 0 && TB.obj_kind[own] >= OBJ_SPINNER)
            st->obj_bm[e * OBJ_BYTES + (own >> 3)] |= (uint8_t)(1u << (own & 7));
        st->prev_obj[e] = ahead_entry(r, off, &p);
        st->prev_t[e] = cur_t;
        /* no interaction yet */
        memset(st->interact_bm + e * INTERACT_BYTES, 0, INTERACT_BYTES);
    }
    return 0;
}

/* The reward of one env: the float32 sum of its N_TERMS terms in numpy's pairwise order (eight accumulators over the
 * blocks of 8, their pairwise total, then the tail one by one) over the SUM_COLS columns of the v76 term layout, term k
 * in column term_sum_col[k] and the other columns 0 (reward.TERM_SUM_COL: the order of the float additions decides
 * the last bit of a few rewards, so the step rewards stay bit-identical to the v76 run's) */
static inline float row_sum(const float *o) {
    _Static_assert(SUM_COLS >= 8 && SUM_COLS <= 128, "row sum");
    float a[SUM_COLS] = {0};
    for (int k = 0; k < N_TERMS; k++) a[term_sum_col[k]] = o[k];
    float r[8];
    int i;
    for (int j = 0; j < 8; j++) r[j] = a[j];
    for (i = 8; i < SUM_COLS - (SUM_COLS % 8); i += 8)
        for (int j = 0; j < 8; j++) r[j] += a[i + j];
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < SUM_COLS; i++) res += a[i];
    return res;
}

/* NativeReward.step(obs) for all num_envs rows of `rec`. Outputs: out [n, N_TERMS] (reward.TERMS order), reward [n],
 * progress [n], champion [n] (bool bytes), tile_bonus [n] (the count-based part of the tiles term). The term numbers
 * below are reward.py's. */
EXPORT int pn_reward_step(struct pn_reward_state *st, const uint8_t *rec, int64_t rs, const int32_t *off,
                          const int32_t *pf, int64_t npf, float *out, float *reward, uint8_t *progress,
                          uint8_t *champion, float *bonus) {
    if (!TB_SET) return -1;
    const int64_t n = st->num_envs, MS = MAP_COUNT + 1;
    const float tile_reward = (float)TILE_REWARD, tile_count_reward = (float)TILE_COUNT_REWARD;
    int any_rows = 0;
    int64_t n_new = 0;
    /* scratch rows: new-tile indices, m, t, ok | ingame << 1, party sums with the old / the new opponent, the
     * new cap, cover | known << 8 | fainted << 16 | imported << 17, the new K */
    int64_t *new_t = st->scratch, *sm = st->scratch + n, *stt = st->scratch + 2 * n, *sf = st->scratch + 3 * n;
    int64_t *s_old = st->scratch + 4 * n, *s_new = st->scratch + 5 * n, *cap_new = st->scratch + 6 * n;
    int64_t *pk = st->scratch + 7 * n, *k_new = st->scratch + 8 * n;
    const int64_t step = ++st->skill_step[0]; /* the skill term's step counter */

    /* One pass over the records (each record's lines are touched while they are in cache): the record of
     * env i + PF_REC is prefetched, env i + PF_ST is parsed and its state lines prefetched, env i is
     * processed. The next opponent (cap, K) of a step depends on whether any env saw a new flag (recomputed for all
     * envs, else held), so the party sums are computed with both and chosen afterwards. */
    for (int64_t i = 0; i < n && i < PF_REC; i++) prefetch_record(rec + i * rs, pf, npf);
    for (int64_t i = 0; i < n && i < PF_ST; i++) {
        struct parsed p;
        parse(rec + i * rs, off, &p);
        sm[i] = p.m;
        stt[i] = p.t;
        sf[i] = p.ok | (p.ingame << 1);
    }
    for (int64_t i = 0; i < n; i++) {
        if (i + PF_REC < n) prefetch_record(rec + (i + PF_REC) * rs, pf, npf);
        if (i + PF_ST < n) {
            int64_t e = i + PF_ST;
            struct parsed p;
            parse(rec + e * rs, off, &p);
            sm[e] = p.m;
            stt[e] = p.t;
            sf[e] = p.ok | (p.ingame << 1);
            __builtin_prefetch(st->tile_bm + e * TILE_BYTES + (p.t >> 3), 1, 3);
            __builtin_prefetch(st->tile_count + p.t, 0, 3);
            __builtin_prefetch(st->trail_t + e * TILE_COUNT + p.t, 1, 3); /* the stamp to write */
            __builtin_prefetch(st->map_seen + e * MS + p.m, 1, 3);
            for (int b = 0; b < FLAG_BYTES; b += 64) __builtin_prefetch(st->seen + e * FLAG_BYTES + b, 1, 3);
            __builtin_prefetch(st->seen + e * FLAG_BYTES + FLAG_BYTES - 1, 1, 3);
            __builtin_prefetch(st->var_last + e * SCENE_BLOCK, 1, 3);
            __builtin_prefetch(st->var_last + e * SCENE_BLOCK + SCENE_BLOCK - 1, 1, 3);
            __builtin_prefetch(st->key_seen + e * KEY_BYTES, 1, 3);
            __builtin_prefetch(st->key_seen + e * KEY_BYTES + KEY_BYTES - 1, 1, 3);
        }
        const uint8_t *r = rec + i * rs;
        float *o = out + i * N_TERMS;
        for (int k = 0; k < N_TERMS; k++) o[k] = 0.0f;
        uint8_t earned = 0;
        champion[i] = 0;
        /* 1. flags: bits set now and never seen this life */
        const uint8_t *fl = r + off[B_FLAGS];
        uint8_t *seen = st->seen + i * FLAG_BYTES;
        _Static_assert(FLAG_BYTES % 32 == 0, "flag blocks");
        __m128i acc0 = _mm_setzero_si128(), acc1 = _mm_setzero_si128();
        for (int w = 0; w < FLAG_BYTES; w += 32) {
            acc0 = _mm_or_si128(acc0, _mm_andnot_si128(_mm_loadu_si128((const __m128i *)(seen + w)),
                                                       _mm_loadu_si128((const __m128i *)(fl + w))));
            acc1 = _mm_or_si128(acc1, _mm_andnot_si128(_mm_loadu_si128((const __m128i *)(seen + w + 16)),
                                                       _mm_loadu_si128((const __m128i *)(fl + w + 16))));
        }
        int anynew = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_or_si128(acc0, acc1), _mm_setzero_si128())) != 0xFFFF;
        if (anynew) {
            any_rows = 1;
            float acc[5] = {0, 0, 0, 0, 0};
            for (int b = 0; b < FLAG_BYTES; b++) {
                uint8_t nb = fl[b] & (uint8_t)~seen[b];
                if (!nb) continue;
                if (b == CHAMP_BYTE && (nb & CHAMP_MASK)) champion[i] = 1;
                seen[b] |= nb;
                while (nb) {
                    int bit = __builtin_ctz(nb);
                    nb &= (uint8_t)(nb - 1);
                    const float *row = TB.flag_matrix + (int64_t)(b * 8 + bit) * 5;
                    for (int c = 0; c < 5; c++) acc[c] += row[c]; /* integer weights: exact in any order */
                }
            }
            for (int c = 0; c < 4; c++) o[c] = acc[c];
            earned = acc[4] > 0;
            st->flags_earned[i] += (int64_t)acc[4];
        }
        /* 2. tiles and maps (the tile counts are read before any of this step's increments) */
        int64_t m = sm[i], t = stt[i];
        int ok = sf[i] & 1, ingame = (sf[i] >> 1) & 1;
        uint8_t *bm = st->tile_bm + i * TILE_BYTES;
        uint8_t bit = (uint8_t)(1u << (t & 7));
        int new_tile = ok && (bm[t >> 3] & bit) == 0;
        float bo = 0.0f, wt = (float)new_tile;
        if (new_tile) {
            bm[t >> 3] |= bit;
            bo = tile_count_reward / __builtin_sqrtf(tile_count_q(st->tile_count[t]));
            new_t[n_new++] = t;
        }
        bonus[i] = bo;
        if (ok && t != st->trail_last[i]) { /* the trail: a move (reward.py "Trail") */
            st->trail_t[i * TILE_COUNT + t] = (uint16_t)++st->trail_now[i];
            st->trail_last[i] = t;
            trail_refresh(st, i);
        }
        uint8_t *ms = st->map_seen + i * MS;
        int new_map = m > 0 && !ms[m];
        double wm = (double)new_map;
        if (new_map) {
            ms[m] = 1;
            st->maps_visited[i] += 1;
        }
        o[4] = wt * tile_reward + bo;
        o[5] = (float)(wm * MAP_REWARD);
        /* 3. party with both candidate opponents (the seen flags now include this step's) */
        {
            int32_t c_new, kk_new;
            next_opponent(seen, &c_new, &kk_new);
            int32_t lv[N_PARTY];
            unsigned cv;
            int fainted;
            party_scan(r, off, lv, &cv, &fainted);
            unsigned kn = hm_known(r, off);
            s_old[i] = top_sum(lv, st->cap[i], st->opp_size[i]);
            s_new[i] = top_sum(lv, c_new, kk_new);
            cap_new[i] = c_new;
            k_new[i] = kk_new;
            pk[i] = (int64_t)(cv & 0xFF) | ((int64_t)(kn & 0xFF) << 8) | ((int64_t)fainted << 16); /* (bits) */
            /* a mon entered the party (bit 17; rows outside the game: held, 0) */
            if (ingame && imported(st, i, r, off)) pk[i] |= (int64_t)1 << 17;
        }
        /* 5. scene variables above their life maximum (rows outside the game: held, compared later) */
        int story = 0;
        const uint8_t *vb = r + off[B_SCENE];
        uint8_t *vl = st->var_last + i * SCENE_BLOCK;
        if (ingame && memcmp(vb, vl, SCENE_BLOCK) != 0) {
            uint16_t *vmax = st->var_max + i * N_SCENE;
            int cnt = 0;
            double vacc = 0.0; /* per-variable weights, accumulated in variable order */
            for (int k = 0; k < N_SCENE; k++) {
                uint16_t v;
                memcpy(&v, vb + 2 * TB.scene_cols[k], 2);
                if (v > vmax[k]) {
                    cnt++;
                    vacc += (double)TB.var_weights[k];
                    vmax[k] = v;
                }
            }
            o[10] = (float)vacc;
            memcpy(vl, vb, SCENE_BLOCK);
            story = cnt > 0;
        }
        /* 6. key items: ids in the key-items pocket not seen this life (rows outside the game: held) */
        int n_key = 0;
        if (ingame) {
            uint8_t *ks = st->key_seen + i * KEY_BYTES;
            for (int s = 0; s < N_KEY_ID; s++) {
                unsigned id = rd16(r, off[G_KEY_ID + s]);
                if (id == 0 || id >= ITEMS_COUNT) continue;
                uint8_t b = (uint8_t)(1u << (id & 7));
                if (!(ks[id >> 3] & b)) {
                    ks[id >> 3] |= b;  /* a duplicate slot of the same id counts once */
                    n_key++;
                }
            }
            o[11] = (float)((double)n_key * KEY_ITEM_REWARD);
        }
        progress[i] = earned | (uint8_t)new_map | (uint8_t)story | (uint8_t)(n_key > 0);
        /* 7. skill events: the worker's event bits (skill_events_priv) ANDed with the in-game bit, each event on its
         * own cooldown, only the set bits visited; the weights accumulated in double in event order */
        {
            uint32_t ev = ingame ? rd32(r, off[G_SKILL_EVENTS]) & SKILL_EVENT_MASK : 0;
            if (ev) {
                int64_t *nx = st->skill_next + i * N_SKILL;
                double acc = 0.0;
                while (ev) {
                    const int q = __builtin_ctz(ev);
                    ev &= ev - 1;
                    if (step >= nx[q]) {
                        nx[q] = step + skill_cooldown[q];
                        acc += (double)TB.skill_weights[q];
                    }
                }
                o[12] = (float)acc;
            }
        }
        /* 8. the puzzle terms (after this step's flags and key items: the end conditions). Fast path: on a map without
         * object tiles, with nothing faced at the previous step, no lookup can hit: only the previous tile is
         * updated */
        if (st->prev_obj[i] < 0 && TB.map_obj_flags[m] == 0) {
            if (ingame) st->prev_t[i] = ok ? t : -1;
        } else {
            struct parsed p;
            parse(r, off, &p);
            const int64_t cur_t = p.ok ? p.t : -1;
            if (ingame && m == VERMILION_MAP) { /* the trash cans' attempt key: a new switch draw clears their bits */
                const uint32_t key = rd32(r, off[B_VARS]);
                if (key != st->trash_key[i]) {
                    uint8_t *ob = st->obj_bm + i * OBJ_BYTES;
                    for (int b = 0; b < OBJ_BYTES; b++) ob[b] &= (uint8_t)~TB.obj_can_mask[b];
                    st->trash_key[i] = key;
                }
            }
            int64_t cnt[4] = {0, 0, 0, 0};
            const int64_t po = st->prev_obj[i];
            if (ingame && po >= 0) {
                const uint8_t kd = TB.obj_kind[po];
                int hit;
                if (kd <= OBJ_MANSION_SWITCH) /* an interaction: a dialog opened on the overworld screen */
                    hit = rd8(r, off[G_IN_DIALOG]) != 0 && rd8(r, off[G_SCREEN]) == SCREEN_OVERWORLD;
                else /* a spinner / pad stepped on: the tile changed */
                    hit = cur_t != st->prev_t[i];
                if (hit && claim(st, i, po)) cnt[kd]++;
            }
            const int64_t own = ingame ? obj_index(cur_t) : -1; /* a pad / spinner under the player */
            if (own >= 0 && TB.obj_kind[own] >= OBJ_SPINNER && claim(st, i, own)) cnt[TB.obj_kind[own]]++;
            for (int k = 0; k < 4; k++) o[OBJ_TERM0 + k] = (float)((double)cnt[k] * (double)TB.obj_weights[k]);
            if (ingame) {
                st->prev_obj[i] = ahead_entry(r, off, &p);
                st->prev_t[i] = cur_t;
            }
        }
        /* 10. move_learn: computed by the Python wrapper (reward.move_learn_pass; o[MOVE_LEARN_TERM] stays 0 here) */
        /* 9. overworld interactions: the common path is the kind byte and a branch; a found key not yet started this
         * life is marked and pays */
        {
            const uint8_t ik = rd8(r, off[G_INTERACT_KIND]);
            if (__builtin_expect(ik != 0, 0) && ingame) {
                const int64_t k = interact_entry(r, off, ik);
                if (k >= 0) {
                    uint8_t *ib = st->interact_bm + i * INTERACT_BYTES + (k >> 3);
                    const uint8_t b = (uint8_t)(1u << (k & 7));
                    if (!(*ib & b)) {
                        *ib |= b;
                        o[INTERACT_TERM] = TB.interact_weights[0];
                    }
                }
            }
        }
    }
    for (int64_t k = 0; k < n_new; k++) st->tile_count[new_t[k]] += 1u;

    /* 3./4. capped levels, HM coverage/taught, whiteout (state only, no record access); the row sums */
    for (int64_t i = 0; i < n; i++) {
        float *o = out + i * N_TERMS;
        int ingame = (sf[i] >> 1) & 1;
        int32_t cap = any_rows ? (int32_t)cap_new[i] : st->cap[i];
        int32_t kk = any_rows ? (int32_t)k_new[i] : st->opp_size[i];
        int64_t s = any_rows ? s_new[i] : s_old[i];
        const unsigned cbits = (unsigned)(pk[i] & 0xFF), kbits = (unsigned)((pk[i] >> 8) & 0xFF);
        int32_t cover = __builtin_popcount(cbits), known = __builtin_popcount(kbits);
        int fainted = (int)((pk[i] >> 16) & 1);
        /* hm_cover / hm_taught pay the bits first seen this life (rows outside the game: held) */
        uint8_t *hs = st->hm_seen + 2 * i;
        int32_t d_known = 0, d_cover = 0;
        if (!ingame) {
            s = st->best[i];
            cover = st->cover[i];
            known = st->known[i];
            fainted = st->fainted[i];
        } else {
            const unsigned nk = kbits & ~(unsigned)hs[0], nc = cbits & ~(unsigned)hs[1];
            hs[0] |= (uint8_t)kbits;
            hs[1] |= (uint8_t)cbits;
            d_known = __builtin_popcount(nk & HM_PAY_MASK);  /* only the paid HMs (hm_reward_moves) */
            d_cover = __builtin_popcount(nc & HM_PAY_MASK);
        }
        int64_t gain = s - st->best[i];
        /* no levels reward when the opponent (cap, K) changes or a mon entered the party: re-based */
        int rebase = kk != st->opp_size[i] || cap != st->cap[i] || ((pk[i] >> 17) & 1);
        int64_t lv = rebase ? 0 : (gain > 0 ? gain : 0);
        if (s > st->best[i]) st->best[i] = s;
        st->cap[i] = cap;
        st->opp_size[i] = kk;
        o[6] = (float)(LEVEL_REWARD * (double)lv);
        o[7] = (float)(HM_REWARD * (double)d_cover);
        o[8] = (float)(HM_TAUGHT_REWARD * (double)d_known);
        o[9] = (float)((double)(fainted && !st->fainted[i]) * WHITEOUT_REWARD);
        st->cover[i] = cover;
        st->known[i] = known;
        st->fainted[i] = (uint8_t)fainted;
        reward[i] = row_sum(o);
        progress[i] = progress[i] | (lv > 0) | (d_cover > 0) | (d_known > 0);
    }
    return 0;
}

/* ---- the trail window (env.py: the policy's trail plane) --------------------------------------------------- */

/* one env's window geometry */
struct win {
    int ok;              /* a window to compute (the record marks a tile), else all 0 */
    int ja, jb;          /* the columns inside the map grid: [ja, jb), never empty (the player's column is inside) */
    int64_t y0, h, w;    /* the map row of window row 0, the grid */
    int64_t t0;          /* the global tile index of window cell (0, 0) (may lie outside the grid: not read there) */
    int64_t m, x, y;     /* the record's map (dense index) and position */
};

static inline void win_parse(const uint8_t *r, const int32_t *off, struct win *v) {
    struct parsed p;
    parse(r, off, &p);
    v->ok = p.ok;
    const int64_t w = TB.map_w[p.m], x0 = p.x - SEEN_COL0;
    v->ja = x0 < 0 ? (int)-x0 : 0;
    v->jb = w - x0 < SEEN_COLS ? (int)(w - x0) : SEEN_COLS;
    v->y0 = p.y - SEEN_ROW0;
    v->h = TB.map_h[p.m];
    v->w = w;
    v->t0 = TB.map_off[p.m] + v->y0 * w + x0;
    v->m = p.m;
    v->x = p.x;
    v->y = p.y;
}

/* the trail values of window row i of env e: 255 - age (uint16) while age < TRAIL_K, 0 outside the grid (rows, and
 * columns outside ``cols``) and in byte 15 */
static inline __m128i trail_row(const struct pn_reward_state *st, int64_t e, const struct win *v, int i, __m128i cols) {
    const int64_t Y = v->y0 + i;
    if (Y < 0 || Y >= v->h) return _mm_setzero_si128();
    const uint16_t *trow = st->trail_t + e * TILE_COUNT;
    const int64_t t0 = v->t0 + i * v->w;
    __m128i s0, s1;
    if (t0 >= 0 && t0 + 16 <= TILE_COUNT) {
        s0 = _mm_loadu_si128((const __m128i *)(trow + t0));
        s1 = _mm_loadu_si128((const __m128i *)(trow + t0) + 1);
    } else { /* (the table's first or last tiles: only the grid's columns are read) */
        uint16_t tmp[16] = {0};
        for (int j = v->ja; j < v->jb; j++) tmp[j] = trow[t0 + j];
        s0 = _mm_loadu_si128((const __m128i *)tmp);
        s1 = _mm_loadu_si128((const __m128i *)tmp + 1);
    }
    _Static_assert(TRAIL_K == 255, "the saturating subtraction shows ages 0 .. 254");
    const __m128i now = _mm_set1_epi16((short)(uint16_t)st->trail_now[e]), c255 = _mm_set1_epi16(255);
    const __m128i v0 = _mm_subs_epu16(c255, _mm_sub_epi16(now, s0)), v1 = _mm_subs_epu16(c255, _mm_sub_epi16(now, s1));
    return _mm_and_si128(_mm_packus_epi16(v0, v1), cols);
}

/* the trail value of window cell (i, j) of env e (0 outside the grid) */
static inline uint8_t trail_cell(const struct pn_reward_state *st, int64_t e, const struct win *v, int i, int j) {
    const int64_t Y = v->y0 + i;
    if (Y < 0 || Y >= v->h || j < v->ja || j >= v->jb) return 0;
    const uint16_t age = (uint16_t)((uint16_t)st->trail_now[e] - st->trail_t[e * TILE_COUNT + v->t0 + i * v->w + j]);
    return age < TRAIL_K ? (uint8_t)(255 - age) : 0;
}

/* Brings env e's trail view (trail_view, its key trail_vkey) to the trail of its position (m, x, y) now: unchanged
 * when nothing moved since the view; after exactly one move by one tile on the same map (the only change of the
 * trail: every age + 1, the new tile age 0) the view is shifted by the move, aged by 1 (saturating: age 254 -> 255
 * reads 0), the new tile set to 255 and only the row or column the move brings into view read from the stamps; any
 * other change (a reset, a map change, a longer jump, a missed step) rebuilds it from the stamps. Equals the direct
 * computation from the stamps in every case. */
static void trail_view_update(struct pn_reward_state *st, int64_t e, const struct win *v, int64_t m, int64_t x,
                              int64_t y, __m128i cols) {
    uint8_t *V = st->trail_view + e * TRAIL_VIEW;
    int64_t *key = st->trail_vkey + e * 4;
    const uint32_t now = st->trail_now[e];
    if (key[0] == m && key[1] == x && key[2] == y && (uint32_t)key[3] == now) return;
    const int64_t dx = x - key[1], dy = y - key[2];
    if (key[0] == m && (uint32_t)key[3] + 1u == now && ((dx == 0) != (dy == 0)) && dx >= -1 && dx <= 1 && dy >= -1 &&
        dy <= 1) {
        __m128i rows[SEEN_ROWS];
        const __m128i one = _mm_set1_epi8(1), keep15 = _mm_srli_si128(_mm_set1_epi8(-1), 1);
        for (int i = 0; i < SEEN_ROWS; i++) {
            const int src = i + (int)dy;
            __m128i r = (src >= 0 && src < SEEN_ROWS) ? _mm_loadu_si128((const __m128i *)(V + 16 * src))
                                                      : _mm_setzero_si128();
            if (dx > 0) r = _mm_srli_si128(r, 1);
            else if (dx < 0) r = _mm_and_si128(_mm_slli_si128(r, 1), keep15);
            rows[i] = _mm_subs_epu8(r, one);
        }
        if (dy != 0) { /* the row the move brings into view */
            const int ie = dy > 0 ? SEEN_ROWS - 1 : 0;
            rows[ie] = trail_row(st, e, v, ie, cols);
        }
        for (int i = 0; i < SEEN_ROWS; i++) _mm_storeu_si128((__m128i *)(V + 16 * i), rows[i]);
        if (dx != 0) { /* the column the move brings into view */
            const int je = dx > 0 ? SEEN_COLS - 1 : 0;
            for (int i = 0; i < SEEN_ROWS; i++) V[16 * i + je] = trail_cell(st, e, v, i, je);
        }
        V[16 * SEEN_ROW0 + SEEN_COL0] = 255; /* the tile just entered */
    } else {
        for (int i = 0; i < SEEN_ROWS; i++) _mm_storeu_si128((__m128i *)(V + 16 * i), trail_row(st, e, v, i, cols));
    }
    key[0] = m;
    key[1] = x;
    key[2] = y;
    key[3] = (int64_t)now;
}

#ifndef PF_WIN
#define PF_WIN 4 /* envs ahead (the trail window) */
#endif

/* the lines env e's window will read: its tile classes, its trail view */
static inline void win_prefetch(const struct pn_reward_state *st, const uint8_t *cls, int64_t e, const struct win *v) {
    if (!v->ok) return;
    for (int b = 0; b < SEEN_CELLS; b += 64) __builtin_prefetch(cls + b, 0, 3);
    __builtin_prefetch(cls + SEEN_CELLS - 1, 0, 3);
    const uint8_t *V = st->trail_view + e * TRAIL_VIEW;
    __builtin_prefetch(V, 1, 3);
    __builtin_prefetch(V + 64, 1, 3);
    __builtin_prefetch(V + 128, 1, 3);
    __builtin_prefetch(V + TRAIL_VIEW - 1, 1, 3);
}

/* one window row: a 16-byte store (its 16th byte is the next row's or env's first cell, written after it), or the
 * first SEEN_COLS bytes through a buffer (the last row of the last env) */
static inline void store_row(uint8_t *o, __m128i r, int buffered) {
    if (!buffered) {
        _mm_storeu_si128((__m128i *)o, r);
    } else {
        uint8_t tmp[16];
        _mm_storeu_si128((__m128i *)tmp, r);
        memcpy(o, tmp, SEEN_COLS);
    }
}
_Static_assert(PF_WIN < 8, "the window ring holds 8 envs");
_Static_assert(SEEN_COLS < 16, "a window row is one 16-byte vector");

/* NativeReward.trail_window(obs, out) for all num_envs rows of `rec`, out [n, SEEN_ROWS, SEEN_COLS] uint8. The window of
 * an env is the player's map grid around (x, y): cell (i, j) = tile (x - SEEN_COL0 + j, y - SEEN_ROW0 + i); 255 - age
 * while the age (trail_now - stamp, uint16) < TRAIL_K, else 0; 0 outside the grid, on cells whose tile class (the
 * record's ``tiles``) is OUTSIDE (0) and on rows without a marked tile (parse: !ok). A window row is one 16-byte vector
 * (SSE2); the 16th byte of a row's store is the next row's (or the next env's) first cell, which is written after it
 * (the last row of the last env is stored through a buffer). The geometry of env e + PF_WIN is parsed ahead and its
 * lines prefetched. */
EXPORT int pn_trail_window(struct pn_reward_state *st, const uint8_t *rec, int64_t rs, const int32_t *off,
                           uint8_t *trail_out) {
    if (!TB_SET) return -1;
    const int64_t n = st->num_envs;
    const int32_t tiles = off[B_TILES];
    /* 16-byte loads of a row's classes read up to 16 bytes past the tiles field: inside the record, else the classes
     * are copied first */
    const int cls_vec = (int64_t)tiles + SEEN_CELLS + 16 <= rs;
    const __m128i zero = _mm_setzero_si128(), col = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    struct win ring[8];
    for (int64_t e = 0; e < n && e < PF_WIN; e++) {
        win_parse(rec + e * rs, off, &ring[e & 7]);
        win_prefetch(st, rec + e * rs + tiles, e, &ring[e & 7]);
    }
    for (int64_t e = 0; e < n; e++) {
        if (e + PF_WIN < n) {
            const int64_t f = e + PF_WIN;
            win_parse(rec + f * rs, off, &ring[f & 7]);
            win_prefetch(st, rec + f * rs + tiles, f, &ring[f & 7]);
        }
        const struct win v = ring[e & 7];
        uint8_t *to = trail_out + e * SEEN_CELLS;
        if (!v.ok) {
            memset(to, 0, SEEN_CELLS);
            continue;
        }
        const uint8_t *cls = rec + e * rs + tiles;
        uint8_t cbuf[SEEN_CELLS + 16];
        if (!cls_vec) {
            memcpy(cbuf, cls, SEEN_CELLS);
            memset(cbuf + SEEN_CELLS, 0, 16);
            cls = cbuf;
        }
        const int ja = v.ja, jb = v.jb;
        /* 0xFF on the columns inside the grid (byte 15 never: jb <= 15) */
        const __m128i cols = _mm_and_si128(_mm_cmpgt_epi8(col, _mm_set1_epi8((char)(ja - 1))),
                                           _mm_cmplt_epi8(col, _mm_set1_epi8((char)jb)));
        const int last = e == n - 1;
        trail_view_update(st, e, &v, v.m, v.x, v.y, cols); /* the env's trail view, brought up to date */
        const uint8_t *tview = st->trail_view + e * TRAIL_VIEW;
        for (int i = 0; i < SEEN_ROWS; i++) {
            const int64_t Y = v.y0 + i;
            const int k0 = i * SEEN_COLS;
            if (Y < 0 || Y >= v.h) {
                memset(to + k0, 0, SEEN_COLS);
                continue;
            }
            /* the cells to fill: inside the grid and of a class other than OUTSIDE */
            const __m128i ok = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(cls + k0)), zero),
                                                cols);
            const int buffered = last && i == SEEN_ROWS - 1;
            store_row(to + k0, _mm_and_si128(_mm_loadu_si128((const __m128i *)(tview + 16 * i)), ok), buffered);
        }
    }
    return 0;
}
