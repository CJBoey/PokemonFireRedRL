// fireredsim: environment table on top of the state regions (ARCHITECTURE.md
// sections 1.4, 4, 5; docs/MIRROR_LAYOUT.md).
//
// The game's whole mutable state is a list of memory regions of this process,
// the mirror layout: the GBA's EWRAM (0x02000000), IWRAM (0x03000000), I/O
// registers, palette RAM, VRAM, OAM and flash chip at their GBA addresses
// (every game RAM variable at its retail ROM address), plus .pnstate
// (0x10000000), the native-only platform state that includes the stack the
// game code runs on.
//
// An environment's state is the concatenation of the regions' bytes plus a few
// host-side fields; the environment being stepped is "active", i.e. its bytes
// are in the live regions. Activating another environment swaps them (two
// copies of the state size). Snapshot = header + region table + host fields +
// state bytes; restore/clone/reset are copies.
//
// The class is a process-wide singleton (there is one set of live regions per
// process) and is not thread-safe; parallelism is by processes
// (ARCHITECTURE.md 1.5).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fireredsim_port.h"

namespace pn {

// One contiguous range of live memory that belongs to an environment's state.
struct MemRegion {
    const char *name; // "ewram", "iwram", "io", "pltt", "vram", "oam", "flash", "pnstate"
    uint8_t *addr;
    size_t size;
};

// Per-environment host fields (not part of the regions).
struct EnvHost {
    uint64_t frame = 0;             // logical frames run since power-on (= ReadKeys count)
    uint64_t input_history_len = 0; // frames whose input was applied since power-on
    uint32_t soft_resets = 0;       // SoftReset events (each powers the env on again)
    uint16_t buttons = 0;           // buttons applied to the last frame
    uint8_t vblank_pending = 0;     // VBlank raised at the last boundary, not yet serviced
    uint8_t soft_reset = 0;         // the last frame ended in SoftReset
    // Host entropy stream of the rng_reseed option (splitmix64 state, docs/RNG_RESEED.md):
    // set by a reset that asks for it, advanced by every reseed, never read otherwise (0 and
    // untouched without the option, as the 8 padding bytes it replaces). Like `frame` it is
    // part of the snapshot (header), not of the state regions or their hash.
    uint64_t rng_stream = 0;
};
static_assert(sizeof(EnvHost) == 32, "EnvHost layout is part of the snapshot format");
static_assert(__builtin_offsetof(EnvHost, rng_stream) == 24, "EnvHost layout is part of the snapshot format");

// The game's PRNG state (docs/RNG_RESEED.md): gRngValue (src/random.c: Random(), every
// random decision of the game plus one call per VBlank) and the wild-encounter dice RNG
// sWildEncounterData.rngState (src/wild_encounter.c: WildEncounterRandom()).
struct RngState {
    uint32_t rng = 0;  // gRngValue
    uint32_t wild = 0; // sWildEncounterData.rngState
};

// rng_reseed's host entropy stream: one splitmix64 step of `state` (advances it, returns
// the 64-bit output). The output becomes the game's PRNG state with rng_state_from.
uint64_t rng_stream_next(uint64_t &state);
// gRngValue = low 32 bits, sWildEncounterData.rngState = high 32 bits of a stream output.
inline RngState rng_state_from(uint64_t v)
{
    return RngState{static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32)};
}
// Addresses of the two PRNG words in this executable (their retail addresses, 0x03005000
// and 0x020386D0, by the mirror layout; checked against the symbol table on first use).
uint32_t rng_value_addr();
uint32_t rng_wild_addr();
// The PRNG state in the live regions (the active environment's); no activation.
RngState live_rng_state();
void set_live_rng_state(const RngState &s);

// Snapshot file format v1:
//   SnapshotHeader
//   region_count x SnapshotRegion   (address and size of each live region)
//   state_size bytes                (the regions' bytes, concatenated in order)
// A snapshot is valid only for the executable that wrote it (build_id) and the
// same region table.
struct SnapshotHeader {
    char magic[8];          // "PNSNAP\r\n"
    uint32_t version;       // 1
    uint32_t header_size;   // sizeof(SnapshotHeader)
    uint64_t build_id;      // xxh64 of the executable file
    uint32_t region_count;
    uint32_t state_size;    // sum of the region sizes
    uint64_t state_hash;    // xxh64 of the state bytes (= Simulator::hash_state)
    EnvHost host;
};
static_assert(sizeof(SnapshotHeader) == 72, "snapshot header layout");

struct SnapshotRegion {
    uint32_t addr;
    uint32_t size;
};

struct EnvConfig {
    std::shared_ptr<const std::vector<uint8_t>> flash; // 128 KiB initial flash, or null (erased)
    // Successive reads of the timer 1 counter REG_TM1CNT_L (RNG seed and
    // trainer ID: title screen START, then naming screen exit); the last
    // value repeats. 1..PORT_TIMER1_VALUES_MAX values.
    std::vector<uint16_t> timer1_values = {0};
};

class Simulator {
public:
    static Simulator &get();

    const std::vector<MemRegion> &regions() const { return regions_; }
    size_t state_size() const { return state_size_; }
    uint64_t build_id() const { return build_id_; }

    // Creates an environment, powers it on and runs AgbMain's initialisation.
    // The new environment is active afterwards.
    int create_env(const EnvConfig &cfg = EnvConfig());
    void destroy_env(int id);
    int num_envs() const;

    // Makes `id` the environment whose bytes are in the live regions.
    void activate(int id);
    int active() const { return active_; }

    // One logical frame with host(id).buttons; handles SoftReset.
    uint32_t run_frame(int id);
    // n_frames frames, `buttons` held for the first hold_frames of them
    // (hold_frames < 0: all n_frames), as in ARCHITECTURE.md section 4.
    void step(int id, uint16_t buttons, int n_frames, int hold_frames = -1);

    // Power-on again from the environment's template (the pristine state with
    // its initial flash and timer 1 values), then AgbMain's initialisation.
    void reset(int id);

    std::vector<uint8_t> snapshot(int id);
    void restore(int id, const std::vector<uint8_t> &blob); // throws on mismatch
    int clone(int id);
    void save_snapshot(int id, const std::string &path);
    void load_snapshot(int id, const std::string &path);

    // xxh64 over the environment's state bytes (regions concatenated).
    uint64_t hash_state(int id);
    // The environment's state bytes (regions concatenated).
    void state_copy(int id, std::vector<uint8_t> &out);
    // The state bytes of an inactive environment without copying them (regions
    // concatenated, valid until the environment is activated or destroyed);
    // nullptr if `id` is the active environment (its bytes are the live regions).
    const uint8_t *parked_state(int id);

    EnvHost &host(int id);
    void probe(int id, PortProbe *out);

    // Copies the flash contents of an environment out (128 KiB).
    std::vector<uint8_t> flash_image(int id);
    // Timer 1 value sequence and read position of an environment.
    std::vector<uint16_t> timer1_values(int id, int *next = nullptr);

    // The game's PRNG state of an environment (activates it), and a direct write of it
    // (a test hook: the rng_reseed equivalence tests write the same values with the option off).
    RngState rng_state(int id);
    void set_rng_state(int id, const RngState &s);
    // rng_reseed (docs/RNG_RESEED.md): the next output of the environment's host entropy stream
    // (EnvHost::rng_stream, advanced) becomes its PRNG state. Returns the state written.
    RngState reseed_rng(int id);

private:
    Simulator();
    struct Slot;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<MemRegion> regions_;
    size_t state_size_ = 0;
    int active_ = -1;
    uint64_t build_id_ = 0;

    Slot &slot(int id);
    int new_slot();
    void park_active();                             // live regions -> active slot's state
    void save_live(std::vector<uint8_t> &out) const; // live regions -> buffer
    void load_live(const uint8_t *in);              // buffer -> live regions
    void zero_live();
    uint64_t hash_live() const;
    void power_on(Slot &s); // live regions must belong to s
};

std::vector<uint8_t> read_file(const std::string &path);
void write_file(const std::string &path, const void *data, size_t size);
// "0x9bf5,0x61c0" -> {0x9bf5, 0x61c0}; throws on errors.
std::vector<uint16_t> parse_u16_list(const std::string &text);

} // namespace pn
