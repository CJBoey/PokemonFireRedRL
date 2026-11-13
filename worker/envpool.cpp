// fireredsim: environment pool of one process (see envpool.h).
#include "envpool.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "input_script.h"
#include "obs_bridge.h"
#include "pn_ipc.h"
#include "render_adapter.h"
#include "save_start_probe.h"
#include "start_preset.h"
#include "symbols.h"
#include "xxhash64.h"

namespace pn {

namespace {

uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

LiveRenderer &renderer()
{
    static LiveRenderer r(Simulator::get());
    return r;
}

// Code addresses the save start driver compares with (this executable's symbols).
struct SaveStartSyms {
    uint32_t intro, title, main_menu, overworld, menu_input;
};

const SaveStartSyms &ss_syms()
{
    static const SaveStartSyms s = [] {
        auto get = [](const char *name) {
            auto a = self_symbols().addr_of(name);
            if (!a)
                throw std::runtime_error(std::string("save start: symbol ") + name + " not found");
            return *a;
        };
        return SaveStartSyms{get("CB2_Intro"), get("CB2_TitleScreenRun"), get("CB2_MainMenu"),
                             get("CB2_Overworld"), get("Task_HandleMenuInput")};
    }();
    return s;
}

const char *save_status_name(unsigned st)
{
    switch (st) {
    case 0: return "empty (no save)";
    case 2: return "invalid (the game says the save file has been deleted)";
    case 4: return "no flash";
    case 0xFF: return "corrupted (the game reports a save error)";
    default: return "unknown";
    }
}

} // namespace

int save_start_buttons(uint16_t prev_buttons, std::string *err)
{
    if (save_start_done())
        return -1;
    if (prev_buttons)
        return 0; // taps: one frame pressed, one released
    const SaveStartSyms &sym = ss_syms();
    PnSaveStartProbe p;
    pn_save_start_probe(&p);
    if (p.callback2 == sym.intro || p.callback2 == sym.title)
        return BTN_START;
    if (p.callback2 == sym.main_menu) {
        if (p.saveFileStatus != pn_save_status_ok()) {
            if (err)
                *err = "the save cannot be continued: gSaveFileStatus " + std::to_string(p.saveFileStatus) + ", " +
                       save_status_name(p.saveFileStatus);
            return -2;
        }
        if (p.paletteFadeActive)
            return 0;
        for (int t = 0; t < PN_SS_TASKS; t++)
            if (p.taskFunc[t] == sym.menu_input)
                return p.taskData1[t] != 0 ? BTN_UP : BTN_A; // tCursorPos 0 = CONTINUE
        return 0;
    }
    if (p.questLogState == 2 || p.questLogState == 3)
        return BTN_B; // QL_HandleInput: B ends the recap
    return 0;
}

bool save_start_done()
{
    PnSaveStartProbe p;
    pn_save_start_probe(&p);
    return p.callback2 == ss_syms().overworld && p.questLogState != 2 && p.questLogState != 3 &&
           !p.paletteFadeActive && !p.scriptEnabled && !p.fieldControlsLocked && p.playerObjectActive &&
           p.heldKeys == 0;
}

bool in_bedroom(const PortProbe &p, uint32_t cb2_overworld)
{
    return p.callback2 == cb2_overworld && p.mapGroup == 4 && p.mapNum == 1 && p.x == 6 && p.y == 6 &&
           !p.scriptEnabled && !p.fieldControlsLocked;
}

EnvPool::EnvPool(int num_envs, const EnvPoolOptions &opt) : sim_(Simulator::get()), opt_(opt)
{
    if (num_envs < 1)
        throw std::runtime_error("EnvPool: at least one environment");
    if (sim_.num_envs() != 0)
        throw std::runtime_error("EnvPool: the process's Simulator must be unused (one pool per process)");
    // Fault guard for the renderer's HBlank hook, before the platform installs its own fault
    // handler on the first game frame (the platform chains host faults to it).
    install_render_guard();
    EnvConfig cfg; // erased flash, timer 1 {0}
    live_ = sim_.create_env(cfg);
    power_on_.state.resize(sim_.state_size());
    save_live(power_on_.state.data());
    power_on_.host = sim_.host(live_);
    envs_.resize(static_cast<size_t>(num_envs));
    for (Env &e : envs_)
        e.host = power_on_.host;
    env_save_info_.resize(static_cast<size_t>(num_envs));
    cur_ = 0; // env 0 is live, the others are "power-on" (empty state buffer = the template)

    for (const std::string &path : opt_.template_files) {
        std::vector<uint8_t> blob = read_file(path);
        std::string err;
        if (!parse_snapshot(blob.data(), blob.size(), true, &err))
            throw std::runtime_error("template " + path + ": " + err);
        Template t;
        SnapshotHeader h;
        std::memcpy(&h, blob.data(), sizeof h);
        const uint8_t *p = blob.data() + sizeof h + sim_.regions().size() * sizeof(SnapshotRegion);
        t.state.assign(p, p + sim_.state_size());
        t.host = h.host;
        file_templates_.push_back(std::move(t));
    }

    // New-game preset ids (start_preset.h): a clear start-up error instead of a write outside save block 1.
    auto hex = [](unsigned v) {
        char b[16];
        std::snprintf(b, sizeof b, "0x%X", v);
        return std::string(b);
    };
    for (uint16_t f : opt_.preset_flags)
        if (f == 0 || f > pn_preset_flag_max())
            throw std::runtime_error("preset flag " + hex(f) + ": not a saved event flag (ids 0x1.." +
                                     hex(pn_preset_flag_max()) + "; special flags from 0x4000 are not saved)");
    for (const auto &v : opt_.preset_vars)
        if (v.first < pn_preset_var_min() || v.first > pn_preset_var_max())
            throw std::runtime_error("preset var " + hex(v.first) + ": not a saved game variable (ids " +
                                     hex(pn_preset_var_min()) + ".." + hex(pn_preset_var_max()) + ")");

    add_script("new_game_to_pallet.txt (bedroom)", bedroom_script_text(), true);
    for (const std::string &path : opt_.template_scripts) {
        std::vector<uint8_t> text = read_file(path);
        add_script(path, std::string(text.begin(), text.end()), false);
    }
    auto cb = self_symbols().addr_of("CB2_Overworld");
    if (!cb)
        throw std::runtime_error("symbol CB2_Overworld not found");
    cb2_overworld_ = *cb;
}

EnvPool::~EnvPool() = default;

size_t EnvPool::snapshot_size() const
{
    return sizeof(SnapshotHeader) + sim_.regions().size() * sizeof(SnapshotRegion) + sim_.state_size();
}

void EnvPool::save_live(uint8_t *out) const
{
    for (const MemRegion &r : sim_.regions()) {
        std::memcpy(out, r.addr, r.size);
        out += r.size;
    }
}

void EnvPool::load_live(const uint8_t *in)
{
    for (const MemRegion &r : sim_.regions()) {
        std::memcpy(r.addr, in, r.size);
        in += r.size;
    }
}

uint64_t EnvPool::hash_live() const
{
    Xxh64Stream h;
    for (const MemRegion &r : sim_.regions())
        h.update(r.addr, r.size);
    return h.digest();
}

void EnvPool::activate(int i)
{
    if (i < 0 || i >= size())
        throw std::runtime_error("bad environment index " + std::to_string(i));
    if (i == cur_)
        return;
    uint64_t t0 = now_ns();
    if (cur_ >= 0) {
        Env &out = envs_[cur_];
        out.state.resize(sim_.state_size());
        save_live(out.state.data());
        out.host = sim_.host(live_);
    }
    Env &in = envs_[i];
    load_live(in.state.empty() ? power_on_.state.data() : in.state.data());
    sim_.host(live_) = in.host;
    cur_ = i;
    stats_.swaps++;
    stats_.swap_ns += now_ns() - t0;
}

void EnvPool::load_template(int i, const Template &t)
{
    if (i < 0 || i >= size())
        throw std::runtime_error("bad environment index " + std::to_string(i));
    if (cur_ != i && cur_ >= 0) {
        Env &out = envs_[cur_];
        out.state.resize(sim_.state_size());
        save_live(out.state.data());
        out.host = sim_.host(live_);
    }
    cur_ = i;
    load_live(t.state.data());
    sim_.host(live_) = t.host;
}

void EnvPool::set_timer1(const uint16_t *v, int count, bool keep_position)
{
    if (count < 1 || count > static_cast<int>(PN_TIMER1_MAX))
        throw std::runtime_error("timer 1 values: 1.." + std::to_string(PN_TIMER1_MAX) + " values");
    int next = 0;
    if (keep_position) {
        uint16_t old[PORT_TIMER1_VALUES_MAX];
        Port_GetTimer1Values(old, &next);
        if (next > count)
            next = count;
    }
    Port_SetTimer1Values(v, count, next);
}

OpResult EnvPool::step(int i, uint16_t buttons, int n_frames, int hold_frames, bool rng_reseed,
                       const AutoAdvance *aa)
{
    activate(i);
    OpResult r;
    if (Port_GetTerminal(nullptr) != PORT_TERMINAL_NONE) {
        r.status |= PN_ST_TERMINAL;
        return r;
    }
    if (hold_frames < 0 || hold_frames > n_frames)
        hold_frames = n_frames;
    EnvHost &h = sim_.host(live_);
    // One step's frames: false when the env became terminal.
    auto run = [&](uint16_t b, int frames, int hold) {
        if (rng_reseed) {
            // docs/RNG_RESEED.md: fresh host entropy at every step boundary; nothing else changes.
            RngState s = rng_state_from(rng_stream_next(h.rng_stream));
            set_live_rng_state(s);
            if (!r.reseeded) {
                r.rng = s;
                r.reseeded = true;
            }
        }
        for (int k = 0; k < frames; k++) {
            h.buttons = k < hold ? b : 0;
            uint32_t f = sim_.run_frame(live_);
            if (f & PORT_FRAME_TERMINAL) {
                r.status |= PN_ST_TERMINAL;
                return false;
            }
            r.frames_run++;
            if (f & PORT_FRAME_SOFT_RESET)
                r.status |= PN_ST_SOFT_RESET;
        }
        return true;
    };
    bool alive = run(buttons, n_frames, hold_frames);
    if (aa != nullptr && alive) {
        int chunk = aa->advance_frames > 0 ? aa->advance_frames : std::max(n_frames, 1);
        uint32_t cap = static_cast<uint32_t>(std::max(aa->max_frames, 0));
        while (!(r.status & PN_ST_SOFT_RESET) && r.frames_run < cap && !obs_waiting()) {
            int k = static_cast<int>(std::min<uint32_t>(static_cast<uint32_t>(chunk), cap - r.frames_run));
            if (!run(0, k, 0))
                break;
        }
    }
    stats_.frames += r.frames_run;
    return r;
}

// Script template: an input script plus "#! frames N" / "#! timer1 A,B" directives.
void EnvPool::add_script(const std::string &name, const std::string &text, bool check_bedroom)
{
    ScriptTemplate st;
    st.name = name;
    st.check_bedroom = check_bedroom;
    InputScript s = InputScript::parse(text, name);
    uint64_t frames = s.end_frame();
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        std::string line = text.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        pos = eol == std::string::npos ? text.size() : eol + 1;
        if (line.rfind("#!", 0) != 0)
            continue;
        std::string rest = line.substr(2);
        size_t a = rest.find_first_not_of(" \t");
        if (a == std::string::npos)
            continue;
        rest = rest.substr(a);
        size_t sp = rest.find_first_of(" \t");
        std::string key = rest.substr(0, sp);
        std::string val = sp == std::string::npos ? "" : rest.substr(rest.find_first_not_of(" \t", sp));
        while (!val.empty() && (val.back() == '\r' || val.back() == ' '))
            val.pop_back();
        if (key == "frames")
            frames = std::stoull(val, nullptr, 0);
        else if (key == "timer1")
            st.fixed_timer1 = parse_u16_list(val);
        else
            throw std::runtime_error(name + ": unknown directive '#! " + key + "'");
    }
    if (st.fixed_timer1.size() > PN_TIMER1_MAX)
        throw std::runtime_error(name + ": at most " + std::to_string(PN_TIMER1_MAX) + " timer1 values");
    st.buttons.resize(frames);
    for (uint64_t f = 0; f < frames; f++)
        st.buttons[f] = s.buttons_at(f);
    scripts_.push_back(std::move(st));
}

bool EnvPool::replay_script(int i, const ScriptTemplate &st, const std::vector<uint16_t> &t1, std::string *err)
{
    load_template(i, power_on_);
    set_timer1(t1.data(), static_cast<int>(t1.size()), false);
    EnvHost &h = sim_.host(live_);
    for (size_t f = 0; f < st.buttons.size(); f++) {
        h.buttons = st.buttons[f];
        uint32_t r = sim_.run_frame(live_);
        stats_.frames++;
        if (r & (PORT_FRAME_TERMINAL | PORT_FRAME_SOFT_RESET)) {
            if (err) {
                PortTerminal t;
                Port_GetTerminal(&t);
                *err = "start script " + st.name + ": " +
                       std::string((r & PORT_FRAME_TERMINAL) ? "terminal status " : "soft reset ") +
                       Port_TerminalKindName(t.kind) + " at frame " + std::to_string(f) + ": " + t.message;
            }
            return false;
        }
    }
    if (!st.check_bedroom)
        return true;
    PortProbe p;
    Port_Probe(&p);
    if (!in_bedroom(p, cb2_overworld_)) {
        if (err)
            *err = "bedroom start script did not end in the bedroom (map " + std::to_string(p.mapGroup) + "." +
                   std::to_string(p.mapNum) + " at (" + std::to_string(p.x) + "," + std::to_string(p.y) +
                   "), callback2 " + self_symbols().describe(p.callback2) + ")";
        return false;
    }
    return true;
}

void EnvPool::apply_preset()
{
    if (opt_.preset_flags.empty() && opt_.preset_vars.empty())
        return;
    std::vector<uint16_t> ids, values;
    for (const auto &v : opt_.preset_vars) {
        ids.push_back(v.first);
        values.push_back(v.second);
    }
    pn_start_preset_apply(opt_.preset_flags.data(), static_cast<int>(opt_.preset_flags.size()), ids.data(),
                          values.data(), static_cast<int>(ids.size()));
}

// Start state of a script template for the timer-1 values (the script's fixed values win;
// default {0}): from the LRU cache, or replayed from power-on and cached.
OpResult EnvPool::reset_script(int i, ScriptTemplate &st, const uint16_t *timer1, int timer1_count)
{
    OpResult r;
    std::vector<uint16_t> key = !st.fixed_timer1.empty() ? st.fixed_timer1
                                : timer1_count > 0      ? std::vector<uint16_t>(timer1, timer1 + timer1_count)
                                                        : std::vector<uint16_t>{0};
    auto it = st.cache.find(key);
    if (it != st.cache.end()) {
        st.lru.splice(st.lru.begin(), st.lru, it->second.second);
        load_template(i, it->second.first);
        stats_.start_cache_hits++;
        r.status |= PN_ST_BEDROOM_CACHED;
        return r;
    }
    stats_.start_cache_misses++;
    std::string err;
    if (!replay_script(i, st, key, &err)) {
        r.status = PN_ST_ERROR;
        r.error = PN_ERR_START_FAILED;
        r.message = err;
        load_template(i, power_on_); // a defined state
        return r;
    }
    apply_preset(); // (before the cache: cached starts carry it too)
    if (opt_.bedroom_cache_max > 0) {
        if (st.cache.size() >= opt_.bedroom_cache_max) {
            st.cache.erase(st.lru.back());
            st.lru.pop_back();
        }
        Template t;
        t.state.resize(sim_.state_size());
        save_live(t.state.data());
        t.host = sim_.host(live_);
        st.lru.push_front(key);
        st.cache.emplace(key, std::make_pair(std::move(t), st.lru.begin()));
    }
    return r;
}

OpResult EnvPool::reset(int i, uint32_t template_id, const uint16_t *timer1, int timer1_count,
                        uint32_t noop_frames, const char *save_path)
{
    OpResult r;
    if (timer1_count < 0 || timer1_count > static_cast<int>(PN_TIMER1_MAX)) {
        r.status = PN_ST_ERROR;
        r.error = PN_ERR_BAD_TEMPLATE;
        r.message = "timer1_count must be 0.." + std::to_string(PN_TIMER1_MAX);
        return r;
    }
    if (i < 0 || i >= size())
        throw std::runtime_error("bad environment index " + std::to_string(i));
    if (template_id == PN_TEMPLATE_SAVE) {
        if (!save_path || !*save_path) {
            r.status = PN_ST_ERROR;
            r.error = PN_ERR_BAD_TEMPLATE;
            r.message = "save start: no save file path";
            return r;
        }
        OpResult s = reset_save(i, save_path, timer1, timer1_count);
        if (s.status & PN_ST_ERROR)
            return s;
        r = s;
        if (noop_frames > 0) {
            OpResult n = step(i, 0, static_cast<int>(noop_frames), 0);
            r.status |= n.status;
            r.frames_run += n.frames_run;
        }
        return r;
    }
    env_save_info_[static_cast<size_t>(i)].reset();
    if (template_id == PN_TEMPLATE_POWER_ON) {
        load_template(i, power_on_);
        if (timer1_count > 0)
            set_timer1(timer1, timer1_count, false);
    } else if (template_id == PN_TEMPLATE_BEDROOM) {
        OpResult s = reset_script(i, scripts_[0], timer1, timer1_count);
        if (s.status & PN_ST_ERROR)
            return s;
        r = s;
    } else if (template_id >= PN_TEMPLATE_SCRIPT0 && template_id - PN_TEMPLATE_SCRIPT0 + 1 < scripts_.size()) {
        OpResult s = reset_script(i, scripts_[template_id - PN_TEMPLATE_SCRIPT0 + 1], timer1, timer1_count);
        if (s.status & PN_ST_ERROR)
            return s;
        r = s;
    } else if (template_id >= PN_TEMPLATE_FILE0 && template_id - PN_TEMPLATE_FILE0 < file_templates_.size()) {
        load_template(i, file_templates_[template_id - PN_TEMPLATE_FILE0]);
        if (timer1_count > 0)
            set_timer1(timer1, timer1_count, true);
    } else {
        r.status = PN_ST_ERROR;
        r.error = PN_ERR_BAD_TEMPLATE;
        r.message = "unknown template id " + std::to_string(template_id);
        return r;
    }
    if (noop_frames > 0) {
        OpResult s = step(i, 0, static_cast<int>(noop_frames), 0);
        r.status |= s.status;
        r.frames_run += s.frames_run;
    }
    return r;
}

// Power-on of env i with a flash image, exactly as Simulator::create_env with EnvConfig::flash and
// timer1_values (fireredsim-cli --sav --seed): zeroed regions, the hardware power-on state with the
// flash contents, the timer-1 values, AgbMain's initialisation. Env i becomes the live env.
void EnvPool::power_on_with_flash(int i, const uint8_t *flash, const std::vector<uint16_t> &timer1)
{
    if (cur_ != i && cur_ >= 0) {
        Env &out = envs_[cur_];
        out.state.resize(sim_.state_size());
        save_live(out.state.data());
        out.host = sim_.host(live_);
    }
    cur_ = i;
    for (const MemRegion &r : sim_.regions())
        std::memset(r.addr, 0, r.size);
    Port_PowerOn(flash);
    set_timer1(timer1.data(), static_cast<int>(timer1.size()), false);
    uint32_t b = Port_BootInit();
    if (b & PORT_FRAME_SOFT_RESET)
        throw std::runtime_error("SoftReset during AgbMain_Init");
    EnvHost &h = sim_.host(live_);
    h = EnvHost();
    h.vblank_pending = (b & PORT_FRAME_VBLANK_PENDING) ? 1 : 0;
}

// Save start of env i: from the cache, or power-on with the save + the CONTINUE driver (envpool.h).
OpResult EnvPool::reset_save(int i, const std::string &path, const uint16_t *timer1, int timer1_count)
{
    OpResult r;
    auto fail = [&](const std::string &m) {
        r.status = PN_ST_ERROR;
        r.error = PN_ERR_START_FAILED;
        r.message = "save start " + path + ": " + m;
        return r;
    };
    env_save_info_[static_cast<size_t>(i)].reset();
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return fail("cannot stat the file");
    uint64_t mtime = static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000000ull + static_cast<uint64_t>(st.st_mtim.tv_nsec);
    SaveFile &sf = save_files_[path];
    std::vector<uint8_t> flash;
    if (sf.hash == 0 || sf.size != static_cast<uint64_t>(st.st_size) || sf.mtime_ns != mtime ||
        sf.ino != static_cast<uint64_t>(st.st_ino)) {
        try {
            flash = read_file(path);
        } catch (const std::exception &e) {
            save_files_.erase(path);
            return fail(e.what());
        }
        if (flash.size() < PORT_FLASH_SIZE) {
            save_files_.erase(path);
            return fail("a FireRed save is 128 KiB of flash, the file has " + std::to_string(flash.size()) + " bytes");
        }
        flash.resize(PORT_FLASH_SIZE); // mGBA appends RTC data to some saves (as fireredsim-cli --sav)
        sf.size = static_cast<uint64_t>(st.st_size);
        sf.mtime_ns = mtime;
        sf.ino = static_cast<uint64_t>(st.st_ino);
        sf.hash = xxh64(flash.data(), flash.size()) | 1; // never 0 (= not loaded)
    }
    std::vector<uint16_t> t1 = timer1_count > 0 ? std::vector<uint16_t>(timer1, timer1 + timer1_count)
                                                : std::vector<uint16_t>{0};
    auto key = std::make_pair(sf.hash, t1);
    auto it = save_cache_.find(key);
    if (it != save_cache_.end()) {
        save_lru_.splice(save_lru_.begin(), save_lru_, it->second.lru);
        load_template(i, it->second.t);
        env_save_info_[static_cast<size_t>(i)] = it->second.info;
        stats_.start_cache_hits++;
        r.status |= PN_ST_BEDROOM_CACHED;
        return r;
    }
    stats_.start_cache_misses++;
    if (flash.empty()) {
        try {
            flash = read_file(path);
        } catch (const std::exception &e) {
            save_files_.erase(path);
            return fail(e.what());
        }
        if (flash.size() < PORT_FLASH_SIZE) {
            save_files_.erase(path);
            return fail("the file changed (shorter than 128 KiB)");
        }
        flash.resize(PORT_FLASH_SIZE);
        if ((xxh64(flash.data(), flash.size()) | 1) != sf.hash) { // rewritten without a new mtime/size/inode
            save_files_.erase(path);
            return fail("the file changed while it was in use; reset again");
        }
    }
    auto info = std::make_shared<SaveStartInfo>();
    info->save_hash = sf.hash;
    info->timer1 = t1;
    power_on_with_flash(i, flash.data(), t1);
    EnvHost &h = sim_.host(live_);
    uint16_t prev = 0;
    std::string err;
    for (;;) {
        int b = save_start_buttons(prev, &err);
        if (b == -1)
            break;
        if (b == -2 || info->buttons.size() >= opt_.save_start_max_frames) {
            if (b != -2) {
                PortProbe p;
                Port_Probe(&p);
                PnSaveStartProbe q;
                pn_save_start_probe(&q);
                err = "no controllable overworld after " + std::to_string(info->buttons.size()) +
                      " frames (callback2 " + self_symbols().describe(p.callback2) + ", map " +
                      std::to_string(p.mapGroup) + "." + std::to_string(p.mapNum) + " (" + std::to_string(p.x) +
                      "," + std::to_string(p.y) + "), script " + std::to_string(q.scriptEnabled) +
                      ", controls locked " + std::to_string(q.fieldControlsLocked) + ", quest log state " +
                      std::to_string(q.questLogState) + ")";
            }
            load_template(i, power_on_); // a defined state
            return fail(err);
        }
        h.buttons = static_cast<uint16_t>(b);
        uint32_t fr = sim_.run_frame(live_);
        stats_.frames++;
        info->buttons.push_back(static_cast<uint16_t>(b));
        prev = static_cast<uint16_t>(b);
        if (fr & (PORT_FRAME_TERMINAL | PORT_FRAME_SOFT_RESET)) {
            PortTerminal t;
            Port_GetTerminal(&t);
            std::string m = std::string((fr & PORT_FRAME_TERMINAL) ? "terminal status " : "soft reset ") +
                            Port_TerminalKindName(t.kind) + " at frame " + std::to_string(info->buttons.size() - 1) +
                            ": " + t.message;
            load_template(i, power_on_);
            return fail(m);
        }
    }
    PortProbe p;
    Port_Probe(&p);
    info->map_group = p.mapGroup;
    info->map_num = p.mapNum;
    info->x = p.x;
    info->y = p.y;
    info->state_hash = hash_live();
    env_save_info_[static_cast<size_t>(i)] = info;
    if (opt_.save_cache_max > 0) {
        if (save_cache_.size() >= opt_.save_cache_max) {
            save_cache_.erase(save_lru_.back());
            save_lru_.pop_back();
        }
        SaveEntry e;
        e.t.state.resize(sim_.state_size());
        save_live(e.t.state.data());
        e.t.host = sim_.host(live_);
        e.info = info;
        save_lru_.push_front(key);
        e.lru = save_lru_.begin();
        save_cache_.emplace(key, std::move(e));
    }
    return r;
}

const SaveStartInfo *EnvPool::save_start_info(int i) const
{
    if (i < 0 || i >= size())
        throw std::runtime_error("bad environment index " + std::to_string(i));
    return env_save_info_[static_cast<size_t>(i)].get();
}

void EnvPool::snapshot(int i, uint8_t *out, bool with_hash)
{
    if (i < 0 || i >= size())
        throw std::runtime_error("bad environment index " + std::to_string(i));
    const auto &regs = sim_.regions();
    SnapshotHeader h{};
    std::memcpy(h.magic, "PNSNAP\r\n", 8);
    h.version = 1;
    h.header_size = sizeof(SnapshotHeader);
    h.build_id = sim_.build_id();
    h.region_count = static_cast<uint32_t>(regs.size());
    h.state_size = static_cast<uint32_t>(sim_.state_size());
    uint8_t *p = out + sizeof h;
    for (const MemRegion &r : regs) {
        SnapshotRegion d{static_cast<uint32_t>(reinterpret_cast<uintptr_t>(r.addr)), static_cast<uint32_t>(r.size)};
        std::memcpy(p, &d, sizeof d);
        p += sizeof d;
    }
    if (i == cur_) {
        save_live(p);
        h.host = sim_.host(live_);
    } else {
        const Env &e = envs_[i];
        std::memcpy(p, e.state.empty() ? power_on_.state.data() : e.state.data(), sim_.state_size());
        h.host = e.host;
    }
    h.state_hash = with_hash ? xxh64(p, sim_.state_size()) : 0;
    std::memcpy(out, &h, sizeof h);
}

bool EnvPool::parse_snapshot(const uint8_t *blob, size_t size, bool verify, std::string *err) const
{
    auto fail = [&](const char *why) {
        if (err)
            *err = why;
        return false;
    };
    SnapshotHeader h;
    if (size < sizeof h)
        return fail("snapshot too small");
    std::memcpy(&h, blob, sizeof h);
    if (std::memcmp(h.magic, "PNSNAP\r\n", 8) != 0)
        return fail("not a fireredsim snapshot");
    if (h.version != 1 || h.header_size != sizeof h)
        return fail("unsupported snapshot version");
    if (h.build_id != sim_.build_id())
        return fail("snapshot was taken by a different executable (build id mismatch)");
    const auto &regs = sim_.regions();
    size_t table = regs.size() * sizeof(SnapshotRegion);
    if (h.region_count != regs.size() || h.state_size != sim_.state_size() || size < sizeof h + table + sim_.state_size())
        return fail("snapshot regions do not match this executable");
    const uint8_t *p = blob + sizeof h;
    for (const MemRegion &r : regs) {
        SnapshotRegion d;
        std::memcpy(&d, p, sizeof d);
        p += sizeof d;
        if (d.addr != static_cast<uint32_t>(reinterpret_cast<uintptr_t>(r.addr)) || d.size != r.size)
            return fail("snapshot regions do not match this executable");
    }
    if (verify && xxh64(p, sim_.state_size()) != h.state_hash)
        return fail("snapshot state hash mismatch (corrupted, or taken without its hash)");
    return true;
}

bool EnvPool::restore(int i, const uint8_t *blob, size_t size, bool verify, std::string *err)
{
    if (i < 0 || i >= this->size())
        throw std::runtime_error("bad environment index " + std::to_string(i));
    if (!parse_snapshot(blob, size, verify, err))
        return false;
    env_save_info_[static_cast<size_t>(i)].reset();
    SnapshotHeader h;
    std::memcpy(&h, blob, sizeof h);
    const uint8_t *p = blob + sizeof h + sim_.regions().size() * sizeof(SnapshotRegion);
    if (i == cur_) {
        load_live(p);
        sim_.host(live_) = h.host;
    } else {
        Env &e = envs_[i];
        e.state.assign(p, p + sim_.state_size());
        e.host = h.host;
    }
    return true;
}

void EnvPool::set_rng_stream(int i, uint64_t state)
{
    if (i < 0 || i >= size())
        throw std::runtime_error("bad environment index " + std::to_string(i));
    if (i == cur_)
        sim_.host(live_).rng_stream = state;
    else
        envs_[static_cast<size_t>(i)].host.rng_stream = state;
}

RngState EnvPool::rng_state(int i)
{
    activate(i);
    return live_rng_state();
}

void EnvPool::set_rng_state(int i, const RngState &s)
{
    activate(i);
    set_live_rng_state(s);
}

uint64_t EnvPool::hash(int i)
{
    if (i == cur_)
        return hash_live();
    const Env &e = envs_.at(static_cast<size_t>(i));
    return xxh64(e.state.empty() ? power_on_.state.data() : e.state.data(), sim_.state_size());
}

EnvHost EnvPool::host(int i) const
{
    if (i == cur_)
        return sim_.host(live_);
    return envs_.at(static_cast<size_t>(i)).host;
}

uint32_t EnvPool::terminal(int i, PortTerminal *out)
{
    activate(i);
    return Port_GetTerminal(out);
}

void EnvPool::observe(int i, void *out, int mode)
{
    activate(i);
    obs_fill(out, mode);
}

void EnvPool::render(int i, uint8_t *out, uint32_t code)
{
    PnRenderOpts opts;
    if (!pixel_render_opts(code, &opts))
        throw std::invalid_argument("unknown pixel format code " + std::to_string(code));
    activate(i);
    LiveRenderer &r = renderer();
    stats_.renders++;
    if (!opt_.render_hblank || !r.hblank_hook_needed()) {
        r.render(out, false, &opts);
        return;
    }
    // The hook runs game code that writes the live I/O registers (and OAM): save and restore the
    // whole state, as fireredsim-render does, so rendering never changes the simulation.
    stats_.render_hook_renders++;
    scratch_.resize(sim_.state_size());
    save_live(scratch_.data());
    EnvHost h = sim_.host(live_);
    bool ok = r.render(out, true, &opts);
    load_live(scratch_.data());
    sim_.host(live_) = h;
    if (!ok)
        r.render(out, false, &opts); // the HBlank callback faulted: the frame without it
}

void EnvPool::state_bytes(int i, std::vector<uint8_t> &out)
{
    out.resize(sim_.state_size());
    if (i == cur_) {
        save_live(out.data());
        return;
    }
    const Env &e = envs_.at(static_cast<size_t>(i));
    std::memcpy(out.data(), e.state.empty() ? power_on_.state.data() : e.state.data(), sim_.state_size());
}

} // namespace pn
