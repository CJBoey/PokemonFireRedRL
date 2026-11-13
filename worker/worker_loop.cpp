// fireredsim worker loop: hosts a contiguous range of environments and executes
// the batches that python/fireredsim (or fireredsim-bench) posts in shared
// memory. Protocol: worker/pn_ipc.h, docs/IPC_PROTOCOL.md. Used by
// fireredsim-worker (worker_main.cpp) and by fireredsim-bench, which runs
// itself as its workers (snapshots are only valid for one executable).
//
//   fireredsim-worker --info
//       JSON description of this executable (build id, state and snapshot sizes,
//       observation size, IPC struct layout) for the Python package.
//   fireredsim-worker --shm-fd FD | --shm-path PATH  --worker W
//                     [--template SNAPSHOT]... [--bedroom-cache N] [--cpu C]
//                     [--preset-flag ID]... [--preset-var ID=VALUE]...
//       Serve worker block W of the shared-memory file until SHUTDOWN or until
//       the parent process dies.
#include <fcntl.h>
#include <linux/futex.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool.h"
#include "obs_bridge.h"
#include "pn_ipc.h"
#include "start_preset.h"
#include "symbols.h"
#include "worker_loop.h"

namespace pn {

namespace {

uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

long futex(uint32_t *addr, int op, uint32_t val, const struct timespec *timeout)
{
    return syscall(SYS_futex, addr, op, val, timeout, nullptr, 0);
}

// Maps [offset, offset + length) of the shared file; returns a pointer to `offset`.
uint8_t *map_range(int fd, uint64_t offset, uint64_t length)
{
    if (length == 0)
        return nullptr;
    uint64_t start = offset & ~static_cast<uint64_t>(PN_PAGE - 1);
    uint64_t end = (offset + length + PN_PAGE - 1) & ~static_cast<uint64_t>(PN_PAGE - 1);
    void *p = mmap(nullptr, static_cast<size_t>(end - start), PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   static_cast<off_t>(start));
    if (p == MAP_FAILED)
        throw std::runtime_error("mmap of the shared memory failed: " + std::string(std::strerror(errno)) +
                                 " (offset " + std::to_string(offset) + ", length " + std::to_string(length) + ")");
    return static_cast<uint8_t *>(p) + (offset - start);
}

void print_info()
{
    Simulator &sim = Simulator::get();
    size_t snapshot_size = sizeof(SnapshotHeader) + sim.regions().size() * sizeof(SnapshotRegion) + sim.state_size();
    std::printf("{\n");
    std::printf("  \"tool\": \"fireredsim-worker\",\n");
    std::printf("  \"build_id\": \"%016llx\",\n", (unsigned long long)sim.build_id());
    std::printf("  \"state_size\": %zu,\n", sim.state_size());
    std::printf("  \"snapshot_size\": %zu,\n", snapshot_size);
    std::printf("  \"obs_size\": %zu,\n", obs_size());
    std::printf("  \"obs_stub\": %s,\n", obs_is_stub() ? "true" : "false");
    // skill_events_priv bits 0 .. n-1 this worker sets (obs_bridge.h; reward v0.12: 13, the mart event; absent in
    // older workers, which set bits 0-11)
    std::printf("  \"skill_events\": %d,\n", obs_skill_event_count());
    // 1: the records carry observer v6's interact_*_priv (obs_bridge.h; absent in older workers)
    std::printf("  \"interact\": %d,\n", obs_interact_fields());
    // 1: the records carry observer v7's offered_move / offered_slot / learned_move_priv (absent in older workers);
    // 2: also observer v8's learned_replaced_priv / summary_cursor
    std::printf("  \"learned_moves\": %d,\n", obs_learned_moves_fields());
    // 1: the records carry observer v9's box_cursor_* (the mon under the PC storage cursor; absent in older workers)
    std::printf("  \"pc_cursor\": %d,\n", obs_pc_cursor_fields());
    std::printf("  \"party_cursor\": %d,\n", obs_party_cursor_fields());
    std::printf("  \"pixel_bytes\": %u,\n", PN_PIXEL_BYTES);
    // Pixel formats (PN_PIX_* codes, docs/PIXEL_OBS.md) this worker renders: PnShmHeader.pixels,
    // PnShmHeader.pixel_stride and the per-op format in PnEnvCmd.flags bits 24-31.
    {
        static const uint32_t kFormats[] = {PN_PIX_RGB8,   PN_PIX_GRAY1,  PN_PIX_GRAY2,  PN_PIX_GRAY3,
                                            PN_PIX_GRAY4,  PN_PIX_GRAY2S, PN_PIX_GRAY3S, PN_PIX_GRAY4S};
        std::printf("  \"pixel_formats\": [");
        for (size_t k = 0; k < sizeof kFormats / sizeof kFormats[0]; k++)
            std::printf("%s%u", k ? ", " : "", kFormats[k]);
        std::printf("],\n");
    }
    // --preset-flag / --preset-var (the new-game preset of script starts, envpool.h): the ids they accept
    // (absent in older workers, which reject the options)
    std::printf("  \"preset\": {\"flag_min\": 1, \"flag_max\": %u, \"var_min\": %u, \"var_max\": %u},\n",
                pn_preset_flag_max(), pn_preset_var_min(), pn_preset_var_max());
    std::printf("  \"timer1_max\": %u,\n", PN_TIMER1_MAX);
    std::printf("  \"shm_version\": %u,\n", PN_SHM_VERSION);
    // rng_reseed (PN_F_RNG_RESEED, docs/RNG_RESEED.md): the PRNG words it replaces.
    std::printf("  \"rng_reseed\": {\"gRngValue\": %u, \"sWildEncounterData.rngState\": %u},\n",
                rng_value_addr(), rng_wild_addr());
    std::printf("  \"regions\": [");
    const auto &regs = sim.regions();
    for (size_t i = 0; i < regs.size(); i++)
        std::printf("%s{\"name\": \"%s\", \"addr\": %u, \"size\": %zu}", i ? ", " : "", regs[i].name,
                    static_cast<unsigned>(reinterpret_cast<uintptr_t>(regs[i].addr)), regs[i].size);
    std::printf("],\n");
    // Fields added in formerly reserved space, listed apart from "structs" so that an older Python
    // (whose layout check rejects fields it does not know) still accepts this worker.
    std::printf("  \"struct_ext\": {\"PnShmHeader\": {\"pixel_stride\": [%zu, %zu]}},\n",
                offsetof(PnShmHeader, pixel_stride), sizeof(((PnShmHeader *)0)->pixel_stride));
    std::printf("  \"structs\": {\n");
#define F(S, f) std::printf("%s\"%s\": [%zu, %zu]", first ? "" : ", ", #f, offsetof(S, f), sizeof(((S *)0)->f)), first = false
    bool first;
    std::printf("    \"PnShmHeader\": {\"size\": %zu, \"fields\": {", sizeof(PnShmHeader));
    first = true;
    F(PnShmHeader, magic); F(PnShmHeader, version); F(PnShmHeader, num_envs); F(PnShmHeader, num_workers);
    F(PnShmHeader, obs_size); F(PnShmHeader, obs_mode); F(PnShmHeader, pixels); F(PnShmHeader, snapshot_slots);
    F(PnShmHeader, snapshot_size); F(PnShmHeader, snapshot_stride); F(PnShmHeader, state_size);
    F(PnShmHeader, flags); F(PnShmHeader, off_workers); F(PnShmHeader, off_cmds); F(PnShmHeader, off_results);
    F(PnShmHeader, off_info); F(PnShmHeader, off_obs); F(PnShmHeader, off_pixels); F(PnShmHeader, off_snapshots);
    F(PnShmHeader, total_size); F(PnShmHeader, build_id); F(PnShmHeader, spin_us); F(PnShmHeader, ready);
    F(PnShmHeader, failed); F(PnShmHeader, tree_fanout); F(PnShmHeader, completions); F(PnShmHeader, go);
    F(PnShmHeader, pending);
    std::printf("}},\n    \"PnWorkerBlock\": {\"size\": %zu, \"fields\": {", sizeof(PnWorkerBlock));
    first = true;
    F(PnWorkerBlock, cmd); F(PnWorkerBlock, seq); F(PnWorkerBlock, done_seq); F(PnWorkerBlock, status);
    F(PnWorkerBlock, env_begin); F(PnWorkerBlock, env_end); F(PnWorkerBlock, pid); F(PnWorkerBlock, state);
    F(PnWorkerBlock, busy_ns); F(PnWorkerBlock, batches); F(PnWorkerBlock, frames); F(PnWorkerBlock, swap_ns);
    F(PnWorkerBlock, wake); F(PnWorkerBlock, sleeping); F(PnWorkerBlock, trace_wake_ns);
    F(PnWorkerBlock, trace_begin_ns); F(PnWorkerBlock, trace_end_ns); F(PnWorkerBlock, tree_child);
    F(PnWorkerBlock, tree_children);
    std::printf("}},\n    \"PnEnvCmd\": {\"size\": %zu, \"fields\": {", sizeof(PnEnvCmd));
    first = true;
    F(PnEnvCmd, op); F(PnEnvCmd, flags); F(PnEnvCmd, buttons); F(PnEnvCmd, n_frames); F(PnEnvCmd, hold_frames);
    F(PnEnvCmd, timer1_count); F(PnEnvCmd, timer1); F(PnEnvCmd, template_id); F(PnEnvCmd, slot);
    F(PnEnvCmd, noop_frames); F(PnEnvCmd, rng_seed); F(PnEnvCmd, max_frames); F(PnEnvCmd, advance_frames);
    std::printf("}},\n    \"PnEnvResult\": {\"size\": %zu, \"fields\": {", sizeof(PnEnvResult));
    first = true;
    F(PnEnvResult, frame); F(PnEnvResult, state_hash); F(PnEnvResult, status); F(PnEnvResult, terminal);
    F(PnEnvResult, frames_run); F(PnEnvResult, soft_resets); F(PnEnvResult, error); F(PnEnvResult, seq);
    F(PnEnvResult, op_ns);
    std::printf("}},\n    \"PnEnvInfo\": {\"size\": %zu, \"fields\": {", sizeof(PnEnvInfo));
    first = true;
    F(PnEnvInfo, message);
    std::printf("}}\n  }\n}\n");
#undef F
}

struct Args {
    int shm_fd = -1;
    std::string shm_path;
    int worker = -1;
    int cpu = -1;
    bool quiet_faults = false;
    EnvPoolOptions pool;
};

void set_message(PnEnvInfo *info, const std::string &m)
{
    if (!info)
        return;
    size_t n = std::min(m.size(), sizeof(info->message) - 1);
    std::memcpy(info->message, m.data(), n);
    info->message[n] = 0;
}

class Worker {
public:
    Worker(const Args &a) : args_(a)
    {
        int fd = a.shm_fd;
        if (fd < 0) {
            fd = open(a.shm_path.c_str(), O_RDWR);
            if (fd < 0)
                throw std::runtime_error("cannot open " + a.shm_path + ": " + std::strerror(errno));
        }
        fd_ = fd;
        hdr_ = reinterpret_cast<PnShmHeader *>(map_range(fd_, 0, sizeof(PnShmHeader)));
        if (hdr_->magic != PN_SHM_MAGIC || hdr_->version != PN_SHM_VERSION)
            throw std::runtime_error("shared memory: bad magic/version (Python package and worker differ)");
        if (a.worker < 0 || static_cast<uint32_t>(a.worker) >= hdr_->num_workers)
            throw std::runtime_error("--worker out of range");
        all_blocks_ = reinterpret_cast<PnWorkerBlock *>(
            map_range(fd_, hdr_->off_workers, static_cast<uint64_t>(hdr_->num_workers) * sizeof(PnWorkerBlock)));
        block_ = all_blocks_ + a.worker;
        begin_ = block_->env_begin;
        end_ = block_->env_end;
        if (begin_ >= end_ || end_ > hdr_->num_envs)
            throw std::runtime_error("bad env range in the worker block");
        uint32_t n = end_ - begin_;
        cmds_ = reinterpret_cast<PnEnvCmd *>(map_range(fd_, hdr_->off_cmds + begin_ * sizeof(PnEnvCmd), n * sizeof(PnEnvCmd)));
        results_ = reinterpret_cast<PnEnvResult *>(
            map_range(fd_, hdr_->off_results + begin_ * sizeof(PnEnvResult), n * sizeof(PnEnvResult)));
        info_ = reinterpret_cast<PnEnvInfo *>(map_range(fd_, hdr_->off_info + begin_ * sizeof(PnEnvInfo), n * sizeof(PnEnvInfo)));
        if (hdr_->obs_size)
            obs_ = map_range(fd_, hdr_->off_obs + static_cast<uint64_t>(begin_) * hdr_->obs_size,
                             static_cast<uint64_t>(n) * hdr_->obs_size);
        if (hdr_->pixels) {
            uint32_t need = pn_pixel_bytes(hdr_->pixels);
            if (!need)
                throw std::runtime_error("shared memory: unknown pixel format " + std::to_string(hdr_->pixels));
            pixel_stride_ = hdr_->pixel_stride ? hdr_->pixel_stride : need;
            if (pixel_stride_ < need)
                throw std::runtime_error("shared memory: pixel_stride " + std::to_string(pixel_stride_) +
                                         " is smaller than a frame of the header's pixel format (" +
                                         std::to_string(need) + " bytes)");
            pixels_ = map_range(fd_, hdr_->off_pixels + static_cast<uint64_t>(begin_) * pixel_stride_,
                                static_cast<uint64_t>(n) * pixel_stride_);
        }
        if (hdr_->snapshot_slots)
            snaps_ = map_range(fd_, hdr_->off_snapshots,
                               static_cast<uint64_t>(hdr_->snapshot_slots) * hdr_->snapshot_stride);

        EnvPoolOptions po = a.pool;
        po.render_hblank = (hdr_->flags & PN_HDR_RENDER_HBLANK) != 0;
        pool_ = std::make_unique<EnvPool>(static_cast<int>(n), po);
        skill_ = std::make_unique<SkillEvents>(static_cast<int>(n));
        learn_ = std::make_unique<LearnedMoves>(static_cast<int>(n));
        if (hdr_->obs_size != obs_size())
            throw std::runtime_error("observation size mismatch: shm " + std::to_string(hdr_->obs_size) +
                                     ", worker " + std::to_string(obs_size()));
        if (hdr_->snapshot_slots && (hdr_->snapshot_size != pool_->snapshot_size() ||
                                     hdr_->snapshot_stride < pool_->snapshot_size()))
            throw std::runtime_error("snapshot size mismatch between shm header and worker");
        if (hdr_->build_id && hdr_->build_id != pool_->build_id())
            throw std::runtime_error("worker executable build id differs from the one the Python side expects");
        if (hdr_->state_size && hdr_->state_size != pool_->state_size())
            throw std::runtime_error("state size mismatch between shm header and worker");
    }

    void run()
    {
        block_->pid = static_cast<uint32_t>(getpid());
        // Read the wake word before announcing readiness: the first batch may be posted the
        // moment the last worker is ready, and its wake-up must not be mistaken for the baseline.
        uint32_t last_wake = __atomic_load_n(&block_->wake, __ATOMIC_ACQUIRE);
        __atomic_store_n(&block_->state, static_cast<uint32_t>(PN_WSTATE_READY), __ATOMIC_RELEASE);
        __atomic_add_fetch(&hdr_->ready, 1, __ATOMIC_SEQ_CST);
        futex(&hdr_->ready, FUTEX_WAKE, INT_MAX, nullptr);
        pid_t parent = getppid();
        for (;;) {
            last_wake = wait_wake(last_wake, parent);
            bool trace = (hdr_->flags & PN_HDR_TRACE) != 0;
            uint64_t t_wake = trace ? now_ns() : 0;
            uint32_t seq = __atomic_load_n(&block_->seq, __ATOMIC_ACQUIRE);
            if (seq == block_->done_seq)
                continue;
            uint32_t cmd = block_->cmd;
            if (cmd & PN_WCMD_TREE)
                wake_children();
            else if (cmd & PN_WCMD_TREE_LIST)
                wake_listed_children();
            cmd &= PN_WCMD_MASK;
            if (cmd == PN_WCMD_SHUTDOWN) {
                block_->state = PN_WSTATE_EXITED;
                finish(seq);
                return;
            }
            uint64_t t0 = now_ns();
            uint64_t swap0 = pool_->stats().swap_ns, frames0 = pool_->stats().frames;
            block_->status = cmd == PN_WCMD_RUN ? 0 : PN_ERR_BAD_OP;
            if (cmd == PN_WCMD_RUN)
                run_batch(seq);
            uint64_t t1 = now_ns();
            block_->busy_ns += t1 - t0;
            if (trace) {
                block_->trace_wake_ns = t_wake;
                block_->trace_begin_ns = t0;
                block_->trace_end_ns = t1;
            }
            block_->swap_ns += pool_->stats().swap_ns - swap0;
            block_->frames += pool_->stats().frames - frames0;
            block_->batches++;
            finish(seq);
        }
    }

private:
    Args args_;
    int fd_ = -1;
    PnShmHeader *hdr_ = nullptr;
    PnWorkerBlock *all_blocks_ = nullptr;
    PnWorkerBlock *block_ = nullptr;
    PnEnvCmd *cmds_ = nullptr;
    PnEnvResult *results_ = nullptr;
    PnEnvInfo *info_ = nullptr;
    uint8_t *obs_ = nullptr, *pixels_ = nullptr, *snaps_ = nullptr;
    uint32_t pixel_stride_ = 0; // bytes per env slot of the pixel section
    uint32_t begin_ = 0, end_ = 0;
    std::unique_ptr<EnvPool> pool_;
    std::unique_ptr<SkillEvents> skill_; // observer v5 skill_events_priv, one state per env
    std::unique_ptr<LearnedMoves> learn_; // observer v7 learned_move_priv, one state per env

    void finish(uint32_t seq)
    {
        __atomic_store_n(&block_->done_seq, seq, __ATOMIC_RELEASE);
        __atomic_add_fetch(&hdr_->completions, 1, __ATOMIC_SEQ_CST);
        if (hdr_->flags & PN_HDR_WAKE_EACH)
            futex(&hdr_->completions, FUTEX_WAKE, INT_MAX, nullptr);
        if (__atomic_sub_fetch(&hdr_->pending, 1, __ATOMIC_SEQ_CST) == 0)
            futex(&hdr_->pending, FUTEX_WAKE, INT_MAX, nullptr);
    }

    static void wake_block(PnWorkerBlock *b)
    {
        __atomic_add_fetch(&b->wake, 1, __ATOMIC_SEQ_CST); // locked RMW: full barrier
        if (__atomic_load_n(&b->sleeping, __ATOMIC_SEQ_CST))
            futex(&b->wake, FUTEX_WAKE, 1, nullptr);
    }

    // Wake-up tree (pn_ipc.h): worker w wakes F*(w+1) .. F*(w+1)+F-1, passing the flag on.
    void wake_children()
    {
        uint32_t F = hdr_->tree_fanout;
        uint32_t w = static_cast<uint32_t>(block_ - all_blocks_);
        for (uint32_t j = 0; F && j < F; j++) {
            uint64_t c = static_cast<uint64_t>(F) * (w + 1) + j;
            if (c >= hdr_->num_workers)
                break;
            wake_block(all_blocks_ + c);
        }
    }

    // Wake-up tree of an asynchronous / partial batch: the children Python listed in the block.
    void wake_listed_children()
    {
        uint32_t n = std::min<uint32_t>(block_->tree_children, PN_TREE_LIST_MAX);
        for (uint32_t j = 0; j < n; j++) {
            uint32_t c = block_->tree_child[j];
            if (c < hdr_->num_workers)
                wake_block(all_blocks_ + c);
        }
    }

    uint32_t wait_wake(uint32_t last, pid_t parent)
    {
        uint32_t spin_us = hdr_->spin_us;
        if (spin_us) {
            uint64_t until = now_ns() + static_cast<uint64_t>(spin_us) * 1000u;
            unsigned k = 0;
            for (;;) {
                uint32_t g = __atomic_load_n(&block_->wake, __ATOMIC_ACQUIRE);
                if (g != last)
                    return g;
                __builtin_ia32_pause();
                if ((++k & 63) == 0 && now_ns() > until)
                    break;
            }
        }
        for (;;) {
            // Dekker with the waker: announce the sleep (xchg: full barrier), then re-check.
            __atomic_exchange_n(&block_->sleeping, 1u, __ATOMIC_SEQ_CST);
            uint32_t g = __atomic_load_n(&block_->wake, __ATOMIC_SEQ_CST);
            if (g != last) {
                __atomic_store_n(&block_->sleeping, 0u, __ATOMIC_RELAXED);
                return g;
            }
            struct timespec ts = {1, 0};
            futex(&block_->wake, FUTEX_WAIT, last, &ts);
            __atomic_store_n(&block_->sleeping, 0u, __ATOMIC_RELAXED);
            if (getppid() != parent) {
                std::fprintf(stderr, "fireredsim-worker %d: parent process gone, exiting\n", args_.worker);
                std::exit(0);
            }
        }
    }

    void run_batch(uint32_t seq)
    {
        for (uint32_t e = begin_; e < end_; e++) {
            uint32_t k = e - begin_;
            const PnEnvCmd c = cmds_[k];
            if (c.op == PN_OP_NONE)
                continue;
            PnEnvResult &res = results_[k];
            uint64_t t0 = now_ns();
            OpResult r;
            int i = static_cast<int>(k);
            // Observer v6 interactions (obs_bridge.h): the latch is process-wide host memory, so it is
            // cleared before every op (frames of earlier ops, of any env, and template builds are not
            // this op's) and taken right after a STEP's frames, which are env i's only. Every other op
            // reports none, also when it ran frames (RESET: start scripts, save starts, noop frames).
            PortInteraction ia;
            std::memset(&ia, 0, sizeof ia);
            Port_TakeInteraction(nullptr);
            Port_TakeItemUse(nullptr); // the consumable-use latch: the same rules (use_consumable / use_battle_item)
            try {
                switch (c.op) {
                case PN_OP_STEP:
                    skill_->advanced(i); // the step's record is a new one (also with 0 frames run)
                    if (c.flags & PN_F_AUTO_ADVANCE) {
                        EnvPool::AutoAdvance aa;
                        aa.max_frames = c.max_frames;
                        aa.advance_frames = c.advance_frames;
                        r = pool_->step(i, c.buttons, c.n_frames, c.hold_frames, (c.flags & PN_F_RNG_RESEED) != 0, &aa);
                    } else {
                        r = pool_->step(i, c.buttons, c.n_frames, c.hold_frames, (c.flags & PN_F_RNG_RESEED) != 0);
                    }
                    Port_TakeInteraction(&ia); // every frame of the step, auto-advance chunks included
                    {
                        PortItemUse iu;
                        Port_TakeItemUse(&iu);
                        if (iu.count) // the step's record (or the next processed one)
                            skill_->used_item(i, iu.count - iu.battle, iu.battle);
                    }
                    break;
                case PN_OP_RESET:
                    // every template (power-on, bedroom and other start scripts, snapshot files, save
                    // starts): the reset's record initialises the skill-event state (RewardV0.reset)
                    skill_->reset(i);
                    learn_->reset(i);
                    if (c.template_id == PN_TEMPLATE_SAVE) {
                        // The save file's path: written by Python into this env's message slot.
                        char path[sizeof(info_[k].message)];
                        std::memcpy(path, info_[k].message, sizeof path);
                        path[sizeof path - 1] = 0;
                        r = pool_->reset(i, c.template_id, c.timer1, c.timer1_count, c.noop_frames, path);
                    } else {
                        r = pool_->reset(i, c.template_id, c.timer1, c.timer1_count, c.noop_frames);
                    }
                    // rng_reseed: the episode's stream (the noop frames above never read it).
                    if ((c.flags & PN_F_RNG_RESEED) && !(r.status & PN_ST_ERROR))
                        pool_->set_rng_stream(i, c.rng_seed);
                    break;
                case PN_OP_SNAPSHOT:
                    if (c.slot >= hdr_->snapshot_slots) {
                        r.status = PN_ST_ERROR;
                        r.error = PN_ERR_BAD_SLOT;
                        r.message = "snapshot slot " + std::to_string(c.slot) + " out of range";
                    } else {
                        pool_->snapshot(i, snaps_ + static_cast<uint64_t>(c.slot) * hdr_->snapshot_stride,
                                        !(c.flags & PN_F_FAST));
                    }
                    break;
                case PN_OP_RESTORE:
                    if (c.slot >= hdr_->snapshot_slots) {
                        r.status = PN_ST_ERROR;
                        r.error = PN_ERR_BAD_SLOT;
                        r.message = "snapshot slot " + std::to_string(c.slot) + " out of range";
                    } else if (!pool_->restore(i, snaps_ + static_cast<uint64_t>(c.slot) * hdr_->snapshot_stride,
                                               hdr_->snapshot_size, !(c.flags & PN_F_FAST), &r.message)) {
                        r.status = PN_ST_ERROR;
                        r.error = PN_ERR_BAD_SNAPSHOT;
                    } else {
                        // a snapshot restore (PokeVecEnv.restore / restore_from_slots: rewind ring slots,
                        // archive starts): the restored record initialises the skill-event state
                        skill_->reset(i);
                        learn_->reset(i);
                    }
                    break;
                case PN_OP_OBSERVE:
                    break;
                default:
                    r.status = PN_ST_ERROR;
                    r.error = PN_ERR_BAD_OP;
                    r.message = "unknown op " + std::to_string(c.op);
                    break;
                }
                if (c.flags & PN_F_OBS) {
                    uint8_t *rec = obs_ + static_cast<uint64_t>(k) * hdr_->obs_size;
                    pool_->observe(i, rec, static_cast<int>(hdr_->obs_mode));
                    skill_->fill(i, rec, static_cast<int>(hdr_->obs_mode));
                    learn_->fill(i, rec, static_cast<int>(hdr_->obs_mode), c.op == PN_OP_STEP);
                    obs_fill_interaction(rec, static_cast<int>(hdr_->obs_mode), ia);
                }
                if (c.flags & PN_F_RENDER) {
                    uint32_t code = (c.flags & PN_F_FORMAT_MASK) >> PN_F_FORMAT_SHIFT;
                    if (!code)
                        code = hdr_->pixels;
                    uint32_t bytes = pn_pixel_bytes(code);
                    if (!pixels_) {
                        r.status |= PN_ST_ERROR;
                        r.error = PN_ERR_NO_PIXELS;
                        r.message = "render requested but the pixel section is disabled";
                    } else if (!bytes || bytes > pixel_stride_) {
                        r.status |= PN_ST_ERROR;
                        r.error = PN_ERR_NO_PIXELS;
                        char buf[160];
                        if (!bytes)
                            std::snprintf(buf, sizeof buf, "render: unknown pixel format 0x%x", code);
                        else
                            std::snprintf(buf, sizeof buf,
                                          "render: pixel format 0x%x needs %u bytes, the pixel slot has %u "
                                          "(pixel_stride)", code, bytes, pixel_stride_);
                        r.message = buf;
                    } else {
                        pool_->render(i, pixels_ + static_cast<uint64_t>(k) * pixel_stride_, code);
                    }
                }
                res.state_hash = (c.flags & PN_F_HASH) ? pool_->hash(i) : 0;
                PortTerminal t;
                uint32_t kind = pool_->terminal(i, &t);
                res.terminal = kind;
                if (kind != PORT_TERMINAL_NONE) {
                    r.status |= PN_ST_TERMINAL;
                    if (r.message.empty())
                        r.message = std::string(Port_TerminalKindName(kind)) + ": " + t.message + " (at " +
                                    self_symbols().describe(t.eip) + ")";
                }
                EnvHost h = pool_->host(i);
                res.frame = h.frame;
                res.soft_resets = h.soft_resets;
            } catch (const std::exception &ex) {
                r.status |= PN_ST_ERROR;
                r.error = PN_ERR_EXCEPTION;
                r.message = ex.what();
            }
            res.status = r.status;
            res.frames_run = r.frames_run;
            res.error = r.error;
            if (!r.message.empty())
                set_message(&info_[k], r.message);
            else
                info_[k].message[0] = 0;
            res.op_ns = static_cast<uint32_t>(std::min<uint64_t>(now_ns() - t0, 0xFFFFFFFFu));
            res.seq = seq;
        }
    }
};

// A number of --preset-flag / --preset-var (decimal or 0x hex), 0..65535.
uint16_t preset_u16(const std::string &opt, const std::string &text)
{
    size_t used = 0;
    unsigned long v = 0;
    try {
        v = std::stoul(text, &used, 0);
    } catch (const std::exception &) {
        used = 0;
    }
    if (text.empty() || used != text.size() || v > 0xFFFF)
        throw std::runtime_error(opt + ": '" + text + "' is not a number 0..65535 (decimal or 0x hex)");
    return static_cast<uint16_t>(v);
}

Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + s);
            return argv[++i];
        };
        if (s == "--shm-fd") a.shm_fd = std::stoi(next());
        else if (s == "--shm-path") a.shm_path = next();
        else if (s == "--worker") a.worker = std::stoi(next());
        else if (s == "--template") a.pool.template_files.push_back(next());
        else if (s == "--template-script") a.pool.template_scripts.push_back(next());
        else if (s == "--bedroom-cache") a.pool.bedroom_cache_max = static_cast<size_t>(std::stoul(next()));
        else if (s == "--save-cache") a.pool.save_cache_max = static_cast<size_t>(std::stoul(next()));
        else if (s == "--preset-flag") a.pool.preset_flags.push_back(preset_u16(s, next()));
        else if (s == "--preset-var") {
            std::string v = next();
            size_t eq = v.find('=');
            if (eq == std::string::npos)
                throw std::runtime_error("--preset-var " + v + ": expected ID=VALUE");
            a.pool.preset_vars.emplace_back(preset_u16(s, v.substr(0, eq)), preset_u16(s, v.substr(eq + 1)));
        }
        else if (s == "--cpu") a.cpu = std::stoi(next());
        else if (s == "--quiet-faults") a.quiet_faults = true;
        else throw std::runtime_error("unknown option " + s);
    }
    if ((a.shm_fd < 0 && a.shm_path.empty()) || a.worker < 0)
        throw std::runtime_error("usage: fireredsim-worker --info | (--shm-fd FD | --shm-path PATH) --worker W "
                                 "[--template SNAPSHOT]... [--template-script SCRIPT]... [--bedroom-cache N] "
                                 "[--save-cache N] [--cpu C] [--quiet-faults] [--preset-flag ID]... "
                                 "[--preset-var ID=VALUE]...\n"
                                 "       fireredsim-worker --save-start SAV [--seed A,B] [--script-out FILE]\n"
                                 "       fireredsim-worker --replay [--sav SAV] [--seed A,B] [--input SCRIPT] "
                                 "[--frames N] [--dump-state FILE] [--rng-reseed STREAM,FIRST,EVERY]");
    return a;
}

} // namespace

int worker_main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--info") {
        try {
            print_info();
            return 0;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "fireredsim-worker: %s\n", e.what());
            return 2;
        }
    }
    if (argc >= 2 && std::string(argv[1]) == "--save-start")
        return save_start_tool_main(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "--replay")
        return replay_tool_main(argc, argv);
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    Args a;
    PnShmHeader *hdr = nullptr;
    PnWorkerBlock *block = nullptr;
    try {
        a = parse_args(argc, argv);
        if (a.quiet_faults)
            Port_SetFaultReport(nullptr, 0); // no per-site notices on stderr
        if (a.cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(a.cpu, &set);
            sched_setaffinity(0, sizeof set, &set);
        }
        Worker w(a);
        w.run();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "fireredsim-worker %d: %s\n", a.worker, e.what());
        // Tell the Python side (if the header is mapped) that this worker cannot start.
        int fd = a.shm_fd >= 0 ? a.shm_fd : (a.shm_path.empty() ? -1 : open(a.shm_path.c_str(), O_RDWR));
        if (fd >= 0) {
            try {
                hdr = reinterpret_cast<PnShmHeader *>(map_range(fd, 0, sizeof(PnShmHeader)));
                if (hdr->magic == PN_SHM_MAGIC && a.worker >= 0 && static_cast<uint32_t>(a.worker) < hdr->num_workers) {
                    block = reinterpret_cast<PnWorkerBlock *>(
                        map_range(fd, hdr->off_workers + static_cast<uint64_t>(a.worker) * sizeof(PnWorkerBlock),
                                  sizeof(PnWorkerBlock)));
                    block->state = PN_WSTATE_FAILED;
                }
                if (hdr->magic == PN_SHM_MAGIC) {
                    __atomic_add_fetch(&hdr->failed, 1, __ATOMIC_SEQ_CST);
                    __atomic_add_fetch(&hdr->ready, 1, __ATOMIC_SEQ_CST);
                    futex(&hdr->ready, FUTEX_WAKE, INT_MAX, nullptr);
                }
            } catch (...) {
            }
        }
        return 2;
    }
}

} // namespace pn
