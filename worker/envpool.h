// fireredsim: a pool of environments hosted by one process (the worker, the
// benchmark, native tests). ARCHITECTURE.md sections 1.5, 4, 5.
//
// One process has one set of live state regions (docs/MIRROR_LAYOUT.md), so the
// pool runs every environment in the single Simulator environment ("live") and
// keeps the others' state bytes in host buffers: activating environment i
// copies the live regions out to the previously active environment's buffer
// and i's buffer in (a "swap", 2 x 564 KiB). A pool of one environment never
// swaps. The pool writes and reads snapshots in the Simulator's file format
// (sim.h, format v1), so snapshots taken here load in fireredsim-cli and vice
// versa.
//
// Resets come from templates: power-on (the state after AgbMain's
// initialisation), input scripts replayed from power-on ("bedroom" =
// tests/inputs/new_game_to_pallet.txt, embedded; more with
// EnvPoolOptions::template_scripts) with the reset's timer-1 values (a new game
// started with that RNG seed / trainer ID) or the script's own fixed values,
// results cached per timer-1 sequence, or snapshot files. The timer-1 values
// (successive REG_TM1CNT_L reads, SeedRngAndSetTrainerId) are part of the
// environment state (.pnstate); a reset may replace them.
//
// Script templates are input scripts (tests/inputs/FORMAT.md) with optional
// directives in comments:
//   #! frames N          run N frames (default: up to the end of the last range)
//   #! timer1 A,B,...    fixed timer-1 values (default: the reset's values)
//
// New-game preset (EnvPoolOptions::preset_flags / preset_vars; docs/IPC_PROTOCOL.md): event
// flags set and game variables written into save block 1 of every script template's start
// (the bedroom included) right after its replay, before the start is cached, so cached and
// replayed starts are the same state. Not applied to power-on, save starts, snapshot file
// templates or restores (their state carries its own flags). A preset start is the script's
// state plus these writes: fireredsim-cli / --replay replays of the script do not include them.
//
// Save starts (PN_TEMPLATE_SAVE): power-on with the flash image of a .sav file
// (as fireredsim-cli --sav), then a fixed rule-based input driver (below,
// save_start_buttons) presses through the title screen and the main menu's
// CONTINUE, skips the Quest Log recap if the save has one, and stops at the first
// frame boundary where the player can move in the overworld. The result is
// cached per (save contents, timer-1 values) like the bedroom (LRU over all
// saves), and the driver's per-frame buttons are kept so that the start can be
// replayed anywhere (fireredsim-cli --sav SAVE --seed V --input SCRIPT).
#pragma once

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sim.h"

namespace pn {

struct EnvPoolOptions {
    std::vector<std::string> template_files;   // PN_TEMPLATE_FILE0 + i: snapshot files
    std::vector<std::string> template_scripts; // PN_TEMPLATE_SCRIPT0 + i: input scripts
    size_t bedroom_cache_max = 16;             // start states cached per script template (LRU)
    size_t save_cache_max = 16;                // save start states cached, all saves together (LRU)
    uint64_t save_start_max_frames = 20000;    // the CONTINUE driver fails after this many frames
    bool render_hblank = true;                 // run gMain.hblankCallback per line when rendering
    // New-game preset (above): flag ids 1 .. FLAGS_COUNT - 1 to set, (var id VARS_START .. VARS_END,
    // value) pairs to write, in this order; the constructor rejects other ids.
    std::vector<uint16_t> preset_flags;
    std::vector<std::pair<uint16_t, uint16_t>> preset_vars;
};

struct OpResult {
    uint32_t status = 0;     // PN_ST_* (pn_ipc.h)
    uint32_t frames_run = 0; // frames completed by this op
    uint32_t error = 0;      // PN_ERR_*
    std::string message;     // error / terminal description
    bool reseeded = false;   // step with rng_reseed: the PRNG state was replaced by `rng`
    RngState rng;            //   (before the first frame; docs/RNG_RESEED.md)
};

struct PoolStats {
    uint64_t swaps = 0, swap_ns = 0;
    uint64_t frames = 0;
    uint64_t start_cache_hits = 0, start_cache_misses = 0; // script templates (bedroom included)
    uint64_t renders = 0, render_hook_renders = 0;
};

// The bedroom start script (tests/inputs/new_game_to_pallet.txt, embedded at build time).
const char *bedroom_script_text();

// A save start: what the CONTINUE driver did (EnvPool::save_start_info).
struct SaveStartInfo {
    uint64_t save_hash = 0;          // xxh64 of the 128 KiB flash image
    std::vector<uint16_t> timer1;    // timer-1 values of the run
    std::vector<uint16_t> buttons;   // the driver's buttons, one per frame from power-on
    uint64_t state_hash = 0;         // state hash at the hand-over boundary
    int map_group = 0, map_num = 0, x = 0, y = 0; // where the player stands then
};

// The save start's input driver: the buttons of the next frame from the state at the
// current boundary (live regions) and the previous frame's buttons. Rules, in order:
//   - the previous frame had buttons: release (every press is a one-frame tap);
//   - copyright screen, intro, title screen (CB2_Intro, CB2_TitleScreenRun): START;
//   - main menu (CB2_MainMenu, task Task_HandleMenuInput, no palette fade): UP while the
//     cursor is not on CONTINUE, then A;
//   - Quest Log recap playing (gQuestLogState 2 or 3): B (ends the recap);
//   - otherwise nothing.
// Returns -1 when the start is complete (the boundary to hand over), -2 with *err if the
// save cannot be continued (the main menu found no valid save).
int save_start_buttons(uint16_t prev_buttons, std::string *err);
// Hand-over condition: CB2_Overworld, no Quest Log playback, no palette fade, no script,
// field controls free, the player's object exists, no button held.
bool save_start_done();

class EnvPool {
public:
    explicit EnvPool(int num_envs, const EnvPoolOptions &opt = EnvPoolOptions());
    ~EnvPool();
    EnvPool(const EnvPool &) = delete;
    EnvPool &operator=(const EnvPool &) = delete;

    int size() const { return static_cast<int>(envs_.size()); }
    size_t state_size() const { return sim_.state_size(); }
    size_t snapshot_size() const;
    uint64_t build_id() const { return sim_.build_id(); }
    int num_file_templates() const { return static_cast<int>(file_templates_.size()); }
    int num_script_templates() const { return static_cast<int>(scripts_.size()) - 1; }
    const PoolStats &stats() const { return stats_; }
    Simulator &sim() { return sim_; }

    // Makes env i the one whose bytes are in the live regions.
    void activate(int i);
    int active() const { return cur_; }

    // n_frames logical frames; `buttons` held for the first hold_frames of them
    // (hold_frames >= n_frames or < 0: all), NOOP afterwards (ARCHITECTURE.md 4).
    // Stops at a terminal status (PN_ST_TERMINAL). rng_reseed (docs/RNG_RESEED.md): before
    // the first frame the next output of env i's host entropy stream replaces the game's PRNG
    // state (gRngValue, sWildEncounterData.rngState; OpResult::rng); not on a terminal env.
    // Decision-point stepping (aa != nullptr, pn_ipc.h PN_F_AUTO_ADVANCE): after the n_frames
    // requested frames, run chunks of aa->advance_frames NOOP frames (0: n_frames, at least 1)
    // while the observer's waiting_for_input predicate (obs_waiting) is 0 at the chunk boundary
    // and the op's total stays below aa->max_frames (the last chunk is shortened to the cap); a
    // terminal status or a soft reset ends the op. With rng_reseed every chunk reseeds first, so
    // the op is exactly the requested step followed by NOOP steps of advance_frames frames.
    // OpResult::rng / reseeded describe the first reseed.
    struct AutoAdvance {
        int max_frames = 0;
        int advance_frames = 0;
    };
    OpResult step(int i, uint16_t buttons, int n_frames, int hold_frames = -1, bool rng_reseed = false,
                  const AutoAdvance *aa = nullptr);

    // rng_reseed's host entropy stream of env i (EnvHost::rng_stream, part of snapshots):
    // a reset with the option sets it from the episode's seed (pn_ipc.h PnEnvCmd::rng_seed).
    void set_rng_stream(int i, uint64_t state);
    uint64_t rng_stream(int i) const { return host(i).rng_stream; }
    // The game's PRNG state of env i, and a direct write of it (test hook: the equivalence
    // tests write the values of a reseeding run into a run without the option).
    RngState rng_state(int i);
    void set_rng_state(int i, const RngState &s);

    // Reset from a template (pn_ipc.h PN_TEMPLATE_*). timer1_count 0 keeps the
    // template's timer-1 values (power-on, bedroom and save starts: {0}).
    // PN_TEMPLATE_SAVE needs save_path (a 128 KiB .sav; longer files are truncated as
    // fireredsim-cli --sav does).
    OpResult reset(int i, uint32_t template_id, const uint16_t *timer1, int timer1_count,
                   uint32_t noop_frames = 0, const char *save_path = nullptr);

    // The save start of env i's last PN_TEMPLATE_SAVE reset (nullptr if none since).
    const SaveStartInfo *save_start_info(int i) const;

    // Snapshot file bytes (sim.h format v1) of env i into out[snapshot_size()].
    // with_hash = false leaves state_hash 0 (such a snapshot restores only
    // with verify = false; files need the hash).
    void snapshot(int i, uint8_t *out, bool with_hash = true);
    // Restores a snapshot written by this executable; false (+ err) if it is
    // not one (build id, region table, size or, with verify, the state hash).
    bool restore(int i, const uint8_t *blob, size_t size, bool verify, std::string *err);

    uint64_t hash(int i);
    EnvHost host(int i) const;
    // Terminal status of env i (PORT_TERMINAL_*; 0 while it runs).
    uint32_t terminal(int i, PortTerminal *out = nullptr);

    // Structured observation of env i (obs_bridge: pn_observe of the active env).
    void observe(int i, void *out, int mode);
    // Env i's last frame (render/, on demand) in pixel format `code` (pn_ipc.h
    // PN_PIX_*: RGB8 [160][240][3] by default, or a gray format,
    // pn_pixel_bytes(code) bytes). Throws std::invalid_argument for a code the
    // worker does not render.
    void render(int i, uint8_t *out, uint32_t code = 1 /* PN_PIX_RGB8 */);

    // Copies the state bytes of env i (regions concatenated).
    void state_bytes(int i, std::vector<uint8_t> &out);

private:
    struct Env {
        std::vector<uint8_t> state; // valid while not active
        EnvHost host;
    };
    struct Template {
        std::vector<uint8_t> state;
        EnvHost host;
    };

    Simulator &sim_;
    int live_ = -1; // Simulator env id
    int cur_ = -1;  // pool env whose bytes are live
    std::vector<Env> envs_;
    Template power_on_;
    std::vector<Template> file_templates_;
    EnvPoolOptions opt_;
    PoolStats stats_;
    std::vector<uint8_t> scratch_;

    // Script templates ([0] = the bedroom): replayed from power-on, cached per timer-1 sequence.
    struct ScriptTemplate {
        std::string name;
        std::vector<uint16_t> buttons;     // per frame
        std::vector<uint16_t> fixed_timer1; // empty: the reset's values
        bool check_bedroom = false;
        std::list<std::vector<uint16_t>> lru;
        std::map<std::vector<uint16_t>, std::pair<Template, std::list<std::vector<uint16_t>>::iterator>> cache;
    };
    std::vector<ScriptTemplate> scripts_;
    uint32_t cb2_overworld_ = 0;

    // Save starts: templates keyed by (flash image hash, timer-1 values), one LRU over all
    // saves; the file of a path is re-read only when its size/mtime/inode change.
    struct SaveFile {
        uint64_t size = 0, mtime_ns = 0, ino = 0;
        uint64_t hash = 0;
    };
    struct SaveEntry {
        Template t;
        std::shared_ptr<const SaveStartInfo> info;
        std::list<std::pair<uint64_t, std::vector<uint16_t>>>::iterator lru;
    };
    std::map<std::string, SaveFile> save_files_;
    std::list<std::pair<uint64_t, std::vector<uint16_t>>> save_lru_;
    std::map<std::pair<uint64_t, std::vector<uint16_t>>, SaveEntry> save_cache_;
    std::vector<std::shared_ptr<const SaveStartInfo>> env_save_info_;

    OpResult reset_save(int i, const std::string &path, const uint16_t *timer1, int timer1_count);
    void power_on_with_flash(int i, const uint8_t *flash, const std::vector<uint16_t> &timer1);

    void save_live(uint8_t *out) const;
    void load_live(const uint8_t *in);
    uint64_t hash_live() const;
    void load_template(int i, const Template &t);
    bool replay_script(int i, const ScriptTemplate &st, const std::vector<uint16_t> &t1, std::string *err);
    OpResult reset_script(int i, ScriptTemplate &st, const uint16_t *timer1, int timer1_count);
    void add_script(const std::string &name, const std::string &text, bool check_bedroom);
    void apply_preset(); // the new-game preset on the live regions (EnvPoolOptions::preset_*)
    void set_timer1(const uint16_t *v, int count, bool keep_position);
    bool parse_snapshot(const uint8_t *blob, size_t size, bool verify, std::string *err) const;
};

// Bedroom check (PortProbe): CB2_Overworld, map 4.1 at (6,6), no script, controls free.
bool in_bedroom(const PortProbe &p, uint32_t cb2_overworld);

} // namespace pn
