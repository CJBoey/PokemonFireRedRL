// fireredsim: environment table (see sim.h).
#include "sim.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <ucontext.h>
#include <unistd.h>

#include "symbols.h"
#include "xxhash64.h"

namespace pn {

struct Simulator::Slot {
    bool in_use = false;
    EnvHost host;
    EnvConfig cfg;
    std::vector<uint8_t> state; // the regions' bytes; valid while the slot is not active
    std::shared_ptr<const std::vector<uint8_t>> tmpl; // power-on template (state after Port_PowerOn)
};

std::vector<uint8_t> read_file(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::vector<uint16_t> parse_u16_list(const std::string &text)
{
    std::vector<uint16_t> out;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t comma = text.find(',', pos);
        std::string item = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        char *end = nullptr;
        unsigned long v = std::strtoul(item.c_str(), &end, 0);
        if (item.empty() || !end || *end || v > 0xFFFF)
            throw std::runtime_error("bad 16-bit value '" + item + "' in '" + text + "'");
        out.push_back(static_cast<uint16_t>(v));
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return out;
}

void write_file(const std::string &path, const void *data, size_t size)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot write " + path);
    f.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!f)
        throw std::runtime_error("write failed: " + path);
}

namespace {

// Loud diagnostics for hardware faults in game code. The retail ARM code
// tolerates things that fault on x86: x / 0 and x % 0 return 0 (agbcc's libgcc),
// INT_MIN / -1 returns INT_MIN, and reads through NULL hit the BIOS area. Such a
// site needs a PORTABLE patch with the retail result; never continue silently.
void fault_handler(int sig, siginfo_t *info, void *ctx)
{
    const ucontext_t *uc = static_cast<const ucontext_t *>(ctx);
    uint32_t eip = 0;
#if defined(__i386__)
    eip = static_cast<uint32_t>(uc->uc_mcontext.gregs[REG_EIP]);
#else
    (void)uc;
#endif
    const char *what = sig == SIGFPE ? "SIGFPE (integer division by zero or INT_MIN/-1: retail ARM returns 0 / INT_MIN)"
                     : sig == SIGSEGV ? "SIGSEGV (invalid memory access; retail reads through NULL hit the BIOS area)"
                     : sig == SIGBUS ? "SIGBUS" : sig == SIGILL ? "SIGILL" : "signal";
    std::string where = self_symbols().describe(eip);
    std::fflush(stdout);
    std::fprintf(stderr, "fireredsim: FATAL: %s at eip=0x%08x (%s), fault address %p\n", what, eip, where.c_str(),
                 info ? info->si_addr : nullptr);
    std::fflush(stderr);
    signal(SIGABRT, SIG_DFL);
    abort();
}

void install_fault_handlers()
{
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = fault_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
}

} // namespace

Simulator &Simulator::get()
{
    static Simulator sim;
    return sim;
}

Simulator::Simulator()
{
    // The state regions of the mirror layout, in snapshot order (sim.h). The
    // linker script (tools/mirror/gen_ld.py) must have put them at the GBA's
    // addresses. Nothing else is mapped next to them, so an access that leaves
    // a region faults instead of touching host memory (the 1 KiB I/O, palette
    // and OAM regions excepted: their 4 KiB page continues).
    struct Def {
        const char *name;
        unsigned char *start, *end;
        uintptr_t addr;
        size_t size; // 0: any size (.pnstate)
    };
    const Def defs[] = {
        {"ewram", __gba_ewram_start, __gba_ewram_end, PORT_EWRAM_ADDR, PORT_EWRAM_SIZE},
        {"iwram", __gba_iwram_start, __gba_iwram_end, PORT_IWRAM_ADDR, PORT_IWRAM_SIZE},
        {"io", __gba_io_start, __gba_io_end, PORT_IO_ADDR, PORT_IO_SIZE},
        {"pltt", __gba_pltt_start, __gba_pltt_end, PORT_PLTT_ADDR, PORT_PLTT_SIZE},
        {"vram", __gba_vram_start, __gba_vram_end, PORT_VRAM_ADDR, PORT_VRAM_SIZE},
        {"oam", __gba_oam_start, __gba_oam_end, PORT_OAM_ADDR, PORT_OAM_SIZE},
        {"flash", __gba_flash_start, __gba_flash_end, PORT_FLASH_ADDR, PORT_FLASH_SIZE},
        {"pnstate", __pnstate_start, __pnstate_end, PORT_PNSTATE_ADDR, 0},
    };
    state_size_ = 0;
    for (const Def &d : defs) {
        size_t size = static_cast<size_t>(d.end - d.start);
        if (reinterpret_cast<uintptr_t>(d.start) != d.addr || (d.size && size != d.size) || size == 0)
            throw std::runtime_error(std::string("state region ") + d.name +
                                     " is not where the mirror layout puts it (linker script)");
        regions_.push_back({d.name, d.start, size});
        state_size_ += size;
    }
    for (const MemRegion &r : regions_) {
        // The regions are NOLOAD: as loaded they are all zero, which is the
        // state the GBA's RegisterRamReset leaves RAM in. Nothing may have
        // run yet.
        for (size_t i = 0; i < r.size; i++)
            if (r.addr[i] != 0)
                throw std::runtime_error(std::string("state region ") + r.name + " is not zero at startup");
    }
    std::vector<uint8_t> exe = read_file("/proc/self/exe");
    build_id_ = xxh64(exe.data(), exe.size());
    if (!std::getenv("FIREREDSIM_NO_FAULT_HANDLER"))
        install_fault_handlers();
    // Sound mixer mode (fireredsim_port.h): logic-only (default) or full PCM synthesis.
    if (const char *mix = std::getenv("FIREREDSIM_SOUND_MIX")) {
        std::string m = mix;
        if (m == "full")
            Port_SetSoundMix(PORT_SOUND_MIX_FULL);
        else if (m == "logic" || m.empty())
            Port_SetSoundMix(PORT_SOUND_MIX_LOGIC);
        else
            throw std::runtime_error("FIREREDSIM_SOUND_MIX must be 'logic' or 'full', not '" + m + "'");
    }
}

Simulator::Slot &Simulator::slot(int id)
{
    if (id < 0 || id >= static_cast<int>(slots_.size()) || !slots_[id]->in_use)
        throw std::runtime_error("bad environment id " + std::to_string(id));
    return *slots_[id];
}

int Simulator::num_envs() const
{
    int n = 0;
    for (const auto &s : slots_)
        n += s->in_use;
    return n;
}

int Simulator::new_slot()
{
    for (size_t i = 0; i < slots_.size(); i++)
        if (!slots_[i]->in_use)
            return static_cast<int>(i);
    slots_.push_back(std::make_unique<Slot>());
    return static_cast<int>(slots_.size() - 1);
}

void Simulator::save_live(std::vector<uint8_t> &out) const
{
    out.resize(state_size_);
    uint8_t *p = out.data();
    for (const MemRegion &r : regions_) {
        std::memcpy(p, r.addr, r.size);
        p += r.size;
    }
}

void Simulator::load_live(const uint8_t *in)
{
    for (const MemRegion &r : regions_) {
        std::memcpy(r.addr, in, r.size);
        in += r.size;
    }
}

void Simulator::zero_live()
{
    for (const MemRegion &r : regions_)
        std::memset(r.addr, 0, r.size);
}

uint64_t Simulator::hash_live() const
{
    Xxh64Stream h;
    for (const MemRegion &r : regions_)
        h.update(r.addr, r.size);
    return h.digest();
}

void Simulator::park_active()
{
    if (active_ >= 0 && active_ < static_cast<int>(slots_.size()) && slots_[active_]->in_use)
        save_live(slots_[active_]->state);
    active_ = -1;
}

void Simulator::activate(int id)
{
    if (id == active_)
        return;
    Slot &next = slot(id);
    park_active();
    load_live(next.state.data());
    active_ = id;
}

static void set_timer1_values(const std::vector<uint16_t> &values, int next)
{
    if (values.empty() || values.size() > PORT_TIMER1_VALUES_MAX)
        throw std::runtime_error("timer 1 values: 1.." + std::to_string(PORT_TIMER1_VALUES_MAX) + " values");
    Port_SetTimer1Values(values.data(), static_cast<int>(values.size()), next);
}

// Power-on of the active slot: template restore + AgbMain_Init.
void Simulator::power_on(Slot &s)
{
    if (!s.tmpl) {
        // Template: the pristine state with the hardware power-on state, the
        // initial flash contents and the timer 1 values (ARCHITECTURE.md
        // section 3, "Reset semantics").
        if (s.cfg.flash && s.cfg.flash->size() != PORT_FLASH_SIZE)
            throw std::runtime_error("flash image must be 128 KiB");
        zero_live();
        Port_PowerOn(s.cfg.flash ? s.cfg.flash->data() : nullptr);
        set_timer1_values(s.cfg.timer1_values, 0);
        auto t = std::make_shared<std::vector<uint8_t>>();
        save_live(*t);
        s.tmpl = t;
    } else {
        load_live(s.tmpl->data());
    }
    uint32_t r = Port_BootInit();
    if (r & PORT_FRAME_SOFT_RESET)
        throw std::runtime_error("SoftReset during AgbMain_Init");
    s.host.vblank_pending = (r & PORT_FRAME_VBLANK_PENDING) ? 1 : 0;
    s.host.soft_reset = 0;
}

int Simulator::create_env(const EnvConfig &cfg)
{
    if (cfg.timer1_values.empty() || cfg.timer1_values.size() > PORT_TIMER1_VALUES_MAX)
        throw std::runtime_error("EnvConfig::timer1_values: 1.." + std::to_string(PORT_TIMER1_VALUES_MAX) +
                                 " values");
    park_active();
    int id = new_slot();
    Slot &s = *slots_[id];
    s.in_use = true;
    s.cfg = cfg;
    s.host = EnvHost();
    s.tmpl.reset();
    s.state.clear();
    active_ = id;
    power_on(s);
    return id;
}

void Simulator::destroy_env(int id)
{
    Slot &s = slot(id);
    s.in_use = false;
    s.state.clear();
    s.state.shrink_to_fit();
    s.tmpl.reset();
    if (active_ == id)
        active_ = -1;
}

const uint8_t *Simulator::parked_state(int id)
{
    Slot &s = slot(id);
    if (id == active_)
        return nullptr;
    return s.state.data();
}

EnvHost &Simulator::host(int id)
{
    return slot(id).host;
}

uint32_t Simulator::run_frame(int id)
{
    activate(id);
    Slot &s = slot(id);
    uint32_t r = Port_RunFrame(s.host.vblank_pending, s.host.buttons);
    s.host.frame++;
    s.host.input_history_len++;
    s.host.soft_reset = 0;
    if (r & PORT_FRAME_SOFT_RESET) {
        // The game called SoftReset: power on again keeping the flash chip
        // (saves survive a reset), as the BIOS restarting the ROM does. The
        // timer 1 sequence continues where it was.
        std::vector<uint8_t> flash(PORT_FLASH_SIZE);
        Port_ReadFlashImage(flash.data());
        uint16_t values[PORT_TIMER1_VALUES_MAX];
        int next = 0;
        int count = Port_GetTimer1Values(values, &next);
        zero_live();
        Port_PowerOn(flash.data());
        Port_SetTimer1Values(values, count, next);
        uint32_t b = Port_BootInit();
        s.host.vblank_pending = (b & PORT_FRAME_VBLANK_PENDING) ? 1 : 0;
        s.host.soft_resets++;
        s.host.soft_reset = 1;
        return r;
    }
    s.host.vblank_pending = (r & PORT_FRAME_VBLANK_PENDING) ? 1 : 0;
    return r;
}

void Simulator::step(int id, uint16_t buttons, int n_frames, int hold_frames)
{
    if (hold_frames < 0)
        hold_frames = n_frames;
    for (int i = 0; i < n_frames; i++) {
        host(id).buttons = i < hold_frames ? buttons : 0;
        run_frame(id);
    }
}

void Simulator::reset(int id)
{
    activate(id);
    Slot &s = slot(id);
    s.host = EnvHost();
    power_on(s);
}

void Simulator::state_copy(int id, std::vector<uint8_t> &out)
{
    Slot &s = slot(id);
    if (id == active_)
        save_live(out);
    else
        out = s.state;
}

uint64_t Simulator::hash_state(int id)
{
    Slot &s = slot(id);
    if (id == active_)
        return hash_live();
    return xxh64(s.state.data(), s.state.size());
}

std::vector<uint8_t> Simulator::snapshot(int id)
{
    Slot &s = slot(id);
    std::vector<uint8_t> state;
    state_copy(id, state);
    size_t table = regions_.size() * sizeof(SnapshotRegion);
    std::vector<uint8_t> blob(sizeof(SnapshotHeader) + table + state_size_);
    SnapshotHeader h{};
    std::memcpy(h.magic, "PNSNAP\r\n", 8);
    h.version = 1;
    h.header_size = sizeof(SnapshotHeader);
    h.build_id = build_id_;
    h.region_count = static_cast<uint32_t>(regions_.size());
    h.state_size = static_cast<uint32_t>(state_size_);
    h.state_hash = xxh64(state.data(), state.size());
    h.host = s.host;
    uint8_t *p = blob.data();
    std::memcpy(p, &h, sizeof h);
    p += sizeof h;
    for (const MemRegion &r : regions_) {
        SnapshotRegion d{static_cast<uint32_t>(reinterpret_cast<uintptr_t>(r.addr)), static_cast<uint32_t>(r.size)};
        std::memcpy(p, &d, sizeof d);
        p += sizeof d;
    }
    std::memcpy(p, state.data(), state.size());
    return blob;
}

void Simulator::restore(int id, const std::vector<uint8_t> &blob)
{
    Slot &s = slot(id);
    SnapshotHeader h;
    if (blob.size() < sizeof h)
        throw std::runtime_error("snapshot too small");
    std::memcpy(&h, blob.data(), sizeof h);
    if (std::memcmp(h.magic, "PNSNAP\r\n", 8) != 0)
        throw std::runtime_error("not a fireredsim snapshot");
    if (h.version != 1 || h.header_size != sizeof h)
        throw std::runtime_error("unsupported snapshot version");
    if (h.build_id != build_id_)
        throw std::runtime_error("snapshot was taken by a different executable (build id mismatch)");
    size_t table = regions_.size() * sizeof(SnapshotRegion);
    if (h.region_count != regions_.size() || h.state_size != state_size_ ||
        blob.size() != sizeof h + table + state_size_)
        throw std::runtime_error("snapshot regions do not match this executable");
    const uint8_t *p = blob.data() + sizeof h;
    for (const MemRegion &r : regions_) {
        SnapshotRegion d;
        std::memcpy(&d, p, sizeof d);
        p += sizeof d;
        if (d.addr != static_cast<uint32_t>(reinterpret_cast<uintptr_t>(r.addr)) || d.size != r.size)
            throw std::runtime_error("snapshot regions do not match this executable");
    }
    if (xxh64(p, state_size_) != h.state_hash)
        throw std::runtime_error("snapshot state hash mismatch (corrupted file)");
    s.host = h.host;
    if (id == active_)
        load_live(p);
    else
        s.state.assign(p, p + state_size_);
}

int Simulator::clone(int id)
{
    std::vector<uint8_t> blob = snapshot(id);
    EnvConfig cfg = slot(id).cfg;
    auto tmpl = slot(id).tmpl;
    int nid = new_slot();
    Slot &s = *slots_[nid];
    s.in_use = true;
    s.cfg = cfg;
    s.tmpl = tmpl;
    s.state.assign(state_size_, 0);
    restore(nid, blob);
    return nid;
}

void Simulator::save_snapshot(int id, const std::string &path)
{
    std::vector<uint8_t> blob = snapshot(id);
    write_file(path, blob.data(), blob.size());
}

void Simulator::load_snapshot(int id, const std::string &path)
{
    restore(id, read_file(path));
}

void Simulator::probe(int id, PortProbe *out)
{
    activate(id);
    Port_Probe(out);
}

std::vector<uint8_t> Simulator::flash_image(int id)
{
    activate(id);
    std::vector<uint8_t> out(PORT_FLASH_SIZE);
    Port_ReadFlashImage(out.data());
    return out;
}

std::vector<uint16_t> Simulator::timer1_values(int id, int *next)
{
    activate(id);
    uint16_t values[PORT_TIMER1_VALUES_MAX];
    int n = Port_GetTimer1Values(values, next);
    return std::vector<uint16_t>(values, values + n);
}

// ---- rng_reseed (docs/RNG_RESEED.md) ------------------------------------------------------

uint64_t rng_stream_next(uint64_t &state)
{
    // splitmix64 (Steele, Lea, Flood 2014), as python/fireredsim/seeding.py.
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

namespace {

struct RngAddrs {
    uint32_t value, wild;
};

// Both words are game globals at their retail addresses (the mirror layout puts every game RAM
// variable there; tools/mirror/check_addresses.py checks it at link time). The wild-encounter
// state is a static of wild_encounter.c, hence the symbol table (first member: u32 rngState).
const RngAddrs &rng_addrs()
{
    static const RngAddrs a = [] {
        auto get = [](const char *name, uint32_t retail) {
            auto v = self_symbols().addr_of(name);
            if (!v)
                throw std::runtime_error(std::string("rng_reseed: symbol ") + name + " not found");
            if (*v != retail)
                throw std::runtime_error(std::string("rng_reseed: ") + name + " is not at its retail address");
            return *v;
        };
        return RngAddrs{get("gRngValue", 0x03005000u), get("sWildEncounterData", 0x020386D0u)};
    }();
    return a;
}

} // namespace

uint32_t rng_value_addr()
{
    return rng_addrs().value;
}

uint32_t rng_wild_addr()
{
    return rng_addrs().wild;
}

RngState live_rng_state()
{
    const RngAddrs &a = rng_addrs();
    RngState s;
    std::memcpy(&s.rng, reinterpret_cast<const void *>(static_cast<uintptr_t>(a.value)), 4);
    std::memcpy(&s.wild, reinterpret_cast<const void *>(static_cast<uintptr_t>(a.wild)), 4);
    return s;
}

void set_live_rng_state(const RngState &s)
{
    const RngAddrs &a = rng_addrs();
    std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(a.value)), &s.rng, 4);
    std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(a.wild)), &s.wild, 4);
}

RngState Simulator::rng_state(int id)
{
    activate(id);
    return live_rng_state();
}

void Simulator::set_rng_state(int id, const RngState &s)
{
    activate(id);
    set_live_rng_state(s);
}

RngState Simulator::reseed_rng(int id)
{
    activate(id);
    RngState s = rng_state_from(rng_stream_next(slot(id).host.rng_stream));
    set_live_rng_state(s);
    return s;
}

} // namespace pn
