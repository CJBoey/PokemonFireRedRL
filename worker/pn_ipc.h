/* pn_ipc.h -- shared-memory protocol between python/fireredsim (64-bit) and
 * fireredsim-worker (32-bit i386). docs/IPC_PROTOCOL.md is the prose version.
 *
 * Every struct has explicit fixed-width fields, no implicit padding and 8-byte
 * aligned 64-bit fields, so the 32-bit worker and the 64-bit Python process see
 * the same byte offsets (checked by the static asserts below and by
 * python/tests/test_ipc_layout.py against `fireredsim-worker --info`).
 *
 * Layout of the shared file (all section offsets page aligned):
 *
 *   PnShmHeader                       page 0
 *   PnWorkerBlock[num_workers]        128 B each
 *   PnEnvCmd[num_envs]                64 B each, written by Python
 *   PnEnvResult[num_envs]             64 B each, written by the owning worker
 *   PnEnvInfo[num_envs]               256 B each, terminal / error message
 *   obs:        num_envs x obs_size   (PnObs bytes, docs/OBSERVATION.md)
 *   pixels:     num_envs x pixel_stride (one frame per env slot: PN_PIX_* format, e.g. uint8 RGB
 *               [160][240][3] = 115200 bytes) if pixels != 0
 *   snapshots:  snapshot_slots x snapshot_stride (fireredsim snapshot files, sim.h format v1)
 *
 * Synchronisation (x86 only: TSO, plain stores are not reordered with stores):
 *   Python  : writes the env commands, sets each participating worker's
 *             block.cmd and block.seq (the new batch number), sets
 *             header.pending to the number of participating workers, then
 *             wakes them: increments block.wake and FUTEX_WAKEs it (only if, after a
 *             full barrier, block.sleeping is set: see PnWorkerBlock). When every
 *             worker participates and header.tree_fanout = F > 0, Python wakes
 *             only workers 0..F-1 and sets PN_WCMD_TREE in their cmd: worker w
 *             wakes workers F*(w+1) .. F*(w+1)+F-1 before running its own envs
 *             (a wake-up tree: no single process issues hundreds of wake-ups).
 *             Partial batches of futex-waiting workers use the same tree over the
 *             posted workers, with explicit children (PN_WCMD_TREE_LIST, tree_child).
 *   Worker  : waits (spin, then FUTEX_WAIT) for block.wake to change; if its
 *             block.seq != block.done_seq it executes the ops of its env range,
 *             writes the results, stores done_seq = seq, then atomically
 *             decrements header.pending; the worker that brings it to 0
 *             FUTEX_WAKEs header.pending.
 *   Python  : FUTEX_WAITs on header.pending until it is 0 (with a timeout to
 *             detect dead workers).
 * Asynchronous mode (Python's send/recv): batches are posted to subsets of
 * workers at different times; Python never waits on `pending` then, but on
 * `completions`, which every worker increments after each batch (and
 * FUTEX_WAKEs with PN_HDR_WAKE_EACH), and checks each worker's done_seq.
 * The futex words are in a MAP_SHARED mapping and use the shared (not
 * FUTEX_PRIVATE) operations, which work between a 64-bit and a 32-bit process.
 */
#ifndef PN_IPC_H
#define PN_IPC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_SHM_MAGIC 0x48534E50u /* "PNSH" little endian */
#define PN_SHM_VERSION 3u
#define PN_PAGE 4096u
#define PN_PIXEL_W 240u
#define PN_PIXEL_H 160u
#define PN_PIXEL_BYTES (PN_PIXEL_W * PN_PIXEL_H * 3u)
#define PN_TIMER1_MAX 8u

/* Pixel formats (docs/PIXEL_OBS.md section 1): PnShmHeader.pixels and the per-op format of
 * PnEnvCmd.flags bits 24-31. Bits 0-3 = scale k, bit 4 = gray box filter, bit 5 = gray sub-sampled
 * lines; 1 = RGB8; 0 = none (header) / the header's format (per op). Gray output: uint8
 * [ceil(160/k)][ceil(240/k)] (render/include/pn_render.h, PN_FMT_GRAY8). */
#define PN_PIX_NONE 0u
#define PN_PIX_RGB8 1u
#define PN_PIX_SCALE_MASK 0x0Fu
#define PN_PIX_GRAY_BOX 0x10u
#define PN_PIX_GRAY_SUB 0x20u
#define PN_PIX_GRAY1 0x11u  /* [160][240] luma */
#define PN_PIX_GRAY2 0x12u  /* [80][120] mean of 2x2 */
#define PN_PIX_GRAY3 0x13u  /* [54][80]  mean of 3x3 (partial boxes at the bottom / right) */
#define PN_PIX_GRAY4 0x14u  /* [40][60]  mean of 4x4 */
#define PN_PIX_GRAY2S 0x22u /* [80][120] lines 0, 2, 4, ... only; mean of 2 pixels */
#define PN_PIX_GRAY3S 0x23u /* [54][80]  lines 0, 3, 6, ... only; mean of 3 pixels */
#define PN_PIX_GRAY4S 0x24u /* [40][60]  lines 0, 4, 8, ... only; mean of 4 pixels */
#define PN_PIX_MAX_SCALE 4u /* the gray scales the worker accepts */

/* Bytes of one frame of `code`, 0 if the code is not a format the worker renders (PN_PIX_NONE
 * included): RGB8, or gray box / sub-sampled with scale 1..PN_PIX_MAX_SCALE. */
static inline uint32_t pn_pixel_bytes(uint32_t code)
{
    if (code == PN_PIX_RGB8)
        return PN_PIXEL_BYTES;
    uint32_t k = code & PN_PIX_SCALE_MASK, kind = code & ~PN_PIX_SCALE_MASK;
    if ((kind != PN_PIX_GRAY_BOX && kind != PN_PIX_GRAY_SUB) || k < 1 || k > PN_PIX_MAX_SCALE)
        return 0;
    return ((PN_PIXEL_H + k - 1) / k) * ((PN_PIXEL_W + k - 1) / k);
}

/* Header (page 0). Written by Python before the workers start, read-only for
 * them except the fields marked "worker". */
typedef struct PnShmHeader {
    uint32_t magic;            /* PN_SHM_MAGIC */
    uint32_t version;          /* PN_SHM_VERSION */
    uint32_t num_envs;
    uint32_t num_workers;
    uint32_t obs_size;         /* bytes per env in the obs section (pn_obs_size()) */
    uint32_t obs_mode;         /* 0 visible, 1 privileged */
    uint32_t pixels;           /* PN_PIX_* format of the renders (0: no pixel section; 1: RGB8) */
    uint32_t snapshot_slots;
    uint32_t snapshot_size;    /* bytes of one snapshot (header + region table + state) */
    uint32_t snapshot_stride;  /* snapshot_size rounded up to PN_PAGE */
    uint32_t state_size;       /* bytes of an environment's state regions */
    uint32_t flags;            /* PN_HDR_* */
    uint64_t off_workers;
    uint64_t off_cmds;
    uint64_t off_results;
    uint64_t off_info;
    uint64_t off_obs;
    uint64_t off_pixels;       /* 0 if pixels == 0 */
    uint64_t off_snapshots;    /* 0 if snapshot_slots == 0 */
    uint64_t total_size;
    uint64_t build_id;         /* expected executable build id (0: do not check) */
    uint32_t spin_us;          /* worker busy-wait before FUTEX_WAIT, microseconds */
    uint32_t ready;            /* worker: incremented (atomically) when a worker is ready */
    uint32_t failed;           /* worker: incremented when a worker failed to start */
    uint32_t tree_fanout;      /* F of the wake-up tree (PN_WCMD_TREE) */
    uint32_t pixel_stride;     /* offset 136: bytes per env slot of the pixel section; 0 (an older
                                  Python leaves the reserved area 0) = pn_pixel_bytes(pixels).
                                  Env k's slot: off_pixels + k * pixel_stride */
    uint8_t _reserved0[192 - 140];
    uint32_t completions;      /* offset 192: worker: incremented (atomically) after every finished
                                  batch; FUTEX_WAKEd when flags has PN_HDR_WAKE_EACH (async recv) */
    uint32_t _pad1c[15];
    /* futex words, each on its own cache line */
    uint32_t go;               /* offset 256: batch counter (Python increments it per batch; diagnostics) */
    uint32_t _pad1[15];
    uint32_t pending;          /* offset 320: workers still running the current batch */
    uint32_t _pad2[15];
} PnShmHeader;

enum PnHeaderFlags {
    PN_HDR_RENDER_HBLANK = 1u << 0, /* run gMain.hblankCallback per line when rendering */
    PN_HDR_WAKE_EACH = 1u << 1,     /* FUTEX_WAKE `completions` after every worker batch */
    PN_HDR_TRACE = 1u << 2,         /* workers write PnWorkerBlock.trace_* timestamps (diagnostics) */
};

#define PN_TREE_LIST_MAX 8u /* PnWorkerBlock.tree_child entries */

/* One per worker (two cache lines: the futex word apart from the counters). */
typedef struct PnWorkerBlock {
    uint32_t cmd;        /* PN_WCMD_* (Python), | PN_WCMD_TREE */
    uint32_t seq;        /* batch number (Python); the worker runs when seq != done_seq */
    uint32_t done_seq;   /* worker: seq of the last finished batch */
    uint32_t status;     /* worker: 0 ok, else PN_ERR_* of a batch-level failure */
    uint32_t env_begin;  /* first env (Python, fixed) */
    uint32_t env_end;    /* one past the last env (Python, fixed) */
    uint32_t pid;        /* worker */
    uint32_t state;      /* worker: PN_WSTATE_* */
    uint64_t busy_ns;    /* worker: cumulative time spent executing batches */
    uint64_t batches;    /* worker: batches executed */
    uint64_t frames;     /* worker: logical frames run */
    uint64_t swap_ns;    /* worker: cumulative time spent swapping env states */
    uint32_t wake;       /* offset 64: futex word, incremented by whoever wakes the worker */
    uint32_t sleeping;   /* worker: 1 while in (or about to enter) FUTEX_WAIT on `wake`; a waker
                            that increments `wake` with a full barrier may skip FUTEX_WAKE if 0 */
    /* worker, only with PN_HDR_TRACE (diagnostics; former padding): CLOCK_MONOTONIC ns of the
       last batch: its wake-up seen, its own ops begun (after waking its tree children), its ops
       done (before done_seq / pending) */
    uint64_t trace_wake_ns;
    uint64_t trace_begin_ns;
    uint64_t trace_end_ns;
    /* Python, with PN_WCMD_TREE_LIST (former padding): the workers this worker wakes before
       running its own ops in an asynchronous / partial batch (the wake-up tree over the posted
       workers, which Python builds per batch); tree_children of them */
    uint16_t tree_child[PN_TREE_LIST_MAX];
    uint32_t tree_children;
    uint32_t _pad[3];
} PnWorkerBlock;

enum PnWorkerCmd {
    PN_WCMD_RUN = 1,      /* execute the PnEnvCmd ops of the env range */
    PN_WCMD_SHUTDOWN = 2, /* acknowledge and exit */
    PN_WCMD_MASK = 0xFF,
    PN_WCMD_TREE = 0x100, /* wake the children in the wake-up tree first */
    PN_WCMD_TREE_LIST = 0x200, /* wake the workers listed in tree_child[0 .. tree_children) first */
};

enum PnWorkerState {
    PN_WSTATE_STARTING = 0,
    PN_WSTATE_READY = 1,
    PN_WSTATE_EXITED = 2,
    PN_WSTATE_FAILED = 3,
};

/* Per-env operation, written by Python before a batch. */
typedef struct PnEnvCmd {
    uint32_t op;          /* PN_OP_* (PN_OP_NONE: nothing for this env in this batch) */
    uint32_t flags;       /* PN_F_* */
    uint16_t buttons;     /* STEP: buttons held (KEYINPUT layout, active high) */
    uint16_t n_frames;    /* STEP: logical frames to run (0 allowed) */
    uint16_t hold_frames; /* STEP: buttons held for the first hold_frames frames, NOOP after
                             (>= n_frames: all of them) */
    uint16_t timer1_count;/* RESET: number of timer1 values (0: keep the template's) */
    uint16_t timer1[PN_TIMER1_MAX]; /* RESET: successive REG_TM1CNT_L reads (RNG seed, trainer ID) */
    uint32_t template_id; /* RESET: PN_TEMPLATE_* or PN_TEMPLATE_FILE0 + i */
    uint32_t slot;        /* SNAPSHOT / RESTORE: snapshot slot */
    uint32_t noop_frames; /* RESET: NOOP frames run after the reset */
    uint32_t _pad0;
    uint64_t rng_seed;    /* RESET with PN_F_RNG_RESEED: the env's new host entropy stream state
                             (docs/RNG_RESEED.md); ignored without the flag (former padding: 0) */
    uint16_t max_frames;  /* STEP with PN_F_AUTO_ADVANCE: cap on the op's total frames (requested
                             n_frames included; the requested frames always run in full) */
    uint16_t advance_frames; /* STEP with PN_F_AUTO_ADVANCE: frames per auto-advance chunk; the wait
                             predicate is evaluated after each chunk (0: n_frames, 1: every frame) */
    uint32_t _pad1;
} PnEnvCmd;

enum PnOp {
    PN_OP_NONE = 0,
    PN_OP_STEP = 1,     /* n_frames frames, buttons held for hold_frames of them */
    PN_OP_RESET = 2,    /* reset from a template */
    PN_OP_SNAPSHOT = 3, /* write a snapshot into slot */
    PN_OP_RESTORE = 4,  /* restore the snapshot in slot */
    PN_OP_OBSERVE = 5,  /* nothing but the PN_F_OBS / PN_F_RENDER / PN_F_HASH outputs */
};

enum PnOpFlags {
    PN_F_OBS = 1u << 0,      /* write the observation after the op */
    PN_F_RENDER = 1u << 1,   /* render the final frame into the pixel section after the op */
    PN_F_HASH = 1u << 2,     /* write the state hash into the result */
    PN_F_FAST = 1u << 3,     /* SNAPSHOT: leave the header's state hash 0 (in-memory clones);
                                RESTORE: skip the state-hash check */
    PN_F_RNG_RESEED = 1u << 4, /* rng_reseed (docs/RNG_RESEED.md). STEP: before the first frame,
                                  the next output of the env's host entropy stream replaces the
                                  game's PRNG state (gRngValue, sWildEncounterData.rngState);
                                  RESET: the stream state becomes rng_seed (after the start's
                                  noop_frames, which are not reseeded) */
    PN_F_AUTO_ADVANCE = 1u << 5, /* STEP: decision-point stepping (docs/IPC_PROTOCOL.md). After the
                                    n_frames requested frames, keep running chunks of advance_frames
                                    frames with no button held while the observer's waiting_for_input
                                    predicate is 0 at the chunk boundary, until the op's total reaches
                                    max_frames; a terminal status or a soft reset ends it. With
                                    PN_F_RNG_RESEED every chunk reseeds first (as a NOOP step would);
                                    frames_run is the total. Ignored by the other ops */
};

/* PnEnvCmd.flags bits 24-31: the PN_PIX_* format of this op's PN_F_RENDER (0: the header's
 * `pixels`). A format whose bytes exceed the slot (pixel_stride) fails with PN_ERR_NO_PIXELS. */
#define PN_F_FORMAT_SHIFT 24u
#define PN_F_FORMAT_MASK 0xFF000000u

enum PnTemplate {
    PN_TEMPLATE_POWER_ON = 0, /* power-on (state after AgbMain's initialisation) */
    PN_TEMPLATE_BEDROOM = 1,  /* tests/inputs/new_game_to_pallet.txt replayed with the timer1 values */
    PN_TEMPLATE_SAVE = 2,     /* power-on with a .sav, CONTINUE, Quest Log skipped, overworld: the
                                 save file's path is the NUL-terminated string Python writes into
                                 the env's PnEnvInfo.message before the batch (worker/envpool.h) */
    PN_TEMPLATE_FILE0 = 16,   /* snapshot files given to the worker with --template, in order */
    PN_TEMPLATE_SCRIPT0 = 1024, /* input scripts given with --template-script, in order (replayed
                                   from power-on; worker/envpool.h for the #! directives) */
};

/* Per-env result, written by the owning worker. */
typedef struct PnEnvResult {
    uint64_t frame;        /* logical frame counter after the op */
    uint64_t state_hash;   /* xxh64 of the state (PN_F_HASH) */
    uint32_t status;       /* PN_ST_* bits */
    uint32_t terminal;     /* PORT_TERMINAL_* kind (0 while the env runs) */
    uint32_t frames_run;   /* logical frames run by the op */
    uint32_t soft_resets;  /* soft resets since the env's last power-on/template */
    uint32_t error;        /* PN_ERR_* if PN_ST_ERROR */
    uint32_t seq;          /* batch number of this result */
    uint32_t op_ns;        /* time spent on the op (diagnostics) */
    uint32_t _pad[5];
} PnEnvResult;

enum PnStatus {
    PN_ST_OK = 0,
    PN_ST_SOFT_RESET = 1u << 0, /* the game executed SoftReset during this op (env powered on again) */
    PN_ST_TERMINAL = 1u << 1,   /* the env has ended (terminal != 0); reset or restore it */
    PN_ST_ERROR = 1u << 2,      /* the op failed (error, message in PnEnvInfo); env unchanged */
    PN_ST_BEDROOM_CACHED = 1u << 3, /* RESET bedroom / script / save start: served from the worker's
                                       start-state cache */
};

enum PnError {
    PN_ERR_NONE = 0,
    PN_ERR_BAD_OP = 1,
    PN_ERR_BAD_SLOT = 2,
    PN_ERR_BAD_TEMPLATE = 3,
    PN_ERR_BAD_SNAPSHOT = 4,  /* restore: not a snapshot of this executable / corrupted */
    PN_ERR_NO_PIXELS = 5,
    PN_ERR_START_FAILED = 6,  /* bedroom replay did not reach the bedroom; save start: the save
                                 could not be continued to a controllable overworld */
    PN_ERR_EXCEPTION = 7,
};

typedef struct PnEnvInfo {
    char message[256];     /* NUL-terminated: terminal status or error description */
} PnEnvInfo;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#define PN_STATIC_ASSERT static_assert
#else
#define PN_STATIC_ASSERT _Static_assert
#endif
PN_STATIC_ASSERT(sizeof(PnShmHeader) == 384, "PnShmHeader layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnShmHeader, off_workers) == 48, "PnShmHeader layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnShmHeader, pixel_stride) == 136, "PnShmHeader layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnShmHeader, completions) == 192, "PnShmHeader layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnShmHeader, go) == 256, "PnShmHeader layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnShmHeader, pending) == 320, "PnShmHeader layout");
PN_STATIC_ASSERT(sizeof(PnWorkerBlock) == 128, "PnWorkerBlock layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnWorkerBlock, wake) == 64, "PnWorkerBlock layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnWorkerBlock, busy_ns) == 32, "PnWorkerBlock layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnWorkerBlock, trace_wake_ns) == 72, "PnWorkerBlock layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnWorkerBlock, tree_child) == 96, "PnWorkerBlock layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnWorkerBlock, tree_children) == 112, "PnWorkerBlock layout");
PN_STATIC_ASSERT(sizeof(PnEnvCmd) == 64, "PnEnvCmd layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnEnvCmd, template_id) == 32, "PnEnvCmd layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnEnvCmd, rng_seed) == 48, "PnEnvCmd layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnEnvCmd, max_frames) == 56, "PnEnvCmd layout");
PN_STATIC_ASSERT(__builtin_offsetof(PnEnvCmd, advance_frames) == 58, "PnEnvCmd layout");
PN_STATIC_ASSERT(sizeof(PnEnvResult) == 64, "PnEnvResult layout");
PN_STATIC_ASSERT(sizeof(PnEnvInfo) == 256, "PnEnvInfo layout");

#endif /* PN_IPC_H */
