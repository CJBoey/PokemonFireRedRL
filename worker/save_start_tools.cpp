// fireredsim-worker --save-start / --replay (worker_loop.h): the save start of a .sav as a
// standalone run, and fireredsim-cli's replay path inside the worker executable, so that a
// save start can be reproduced outside Python and its state hash compared with the
// worker's (state hashes are only comparable within one executable).
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool.h"
#include "input_script.h"
#include "pn_ipc.h"
#include "sim.h"
#include "worker_loop.h"
#include "xxhash64.h"

namespace pn {

namespace {

std::string hex16(uint64_t v)
{
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

std::string json_str(const std::string &s)
{
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", c);
            o += b;
        } else {
            o += c;
        }
    }
    return o + "\"";
}

// The driver's per-frame buttons as an input script (tests/inputs/FORMAT.md).
std::string buttons_script(const SaveStartInfo &info, const std::string &sav)
{
    std::string seeds;
    for (uint16_t v : info.timer1) {
        char b[16];
        std::snprintf(b, sizeof b, "%s0x%04x", seeds.empty() ? "" : ",", v);
        seeds += b;
    }
    uint64_t n = info.buttons.size();
    std::string t =
        "# fireredsim save start (worker/envpool.h, PN_TEMPLATE_SAVE): power-on with the save\n"
        "# " + sav + "\n"
        "# (flash xxh64 " + hex16(info.save_hash) + "), timer-1 values " + seeds + ".\n"
        "# Title START taps, main menu CONTINUE (A), Quest Log recap ended with B; the player is\n"
        "# controllable in the overworld (map " + std::to_string(info.map_group) + "." +
        std::to_string(info.map_num) + " at (" + std::to_string(info.x) + "," + std::to_string(info.y) +
        ")) at the boundary after frame " + std::to_string(n == 0 ? 0 : n - 1) + ". Replay:\n"
        "#   fireredsim-cli --sav SAVE --seed " + seeds + " --input THIS --frames " + std::to_string(n) + "\n"
        "#! frames " + std::to_string(n) + "\n";
    InputScript s;
    for (uint64_t f = 0; f < n;) {
        if (!info.buttons[f]) {
            f++;
            continue;
        }
        uint64_t g = f;
        while (g < n && info.buttons[g] == info.buttons[f])
            g++;
        s.add(f, g, info.buttons[f]);
        f = g;
    }
    return t + s.to_text();
}

std::vector<uint8_t> load_sav(const std::string &path)
{
    std::vector<uint8_t> flash = read_file(path);
    if (flash.size() < PORT_FLASH_SIZE)
        throw std::runtime_error(path + ": a FireRed save is 128 KiB of flash");
    flash.resize(PORT_FLASH_SIZE); // as fireredsim-cli --sav
    return flash;
}

} // namespace

int save_start_tool_main(int argc, char **argv)
{
    try {
        std::string sav, script_out;
        std::vector<uint16_t> seeds = {0};
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--save-start") sav = next();
            else if (a == "--seed") seeds = parse_u16_list(next());
            else if (a == "--script-out") script_out = next();
            else if (a == "--quiet-faults") Port_SetFaultReport(nullptr, 0);
            else throw std::runtime_error("unknown option " + a);
        }
        if (seeds.empty() || seeds.size() > PN_TIMER1_MAX)
            throw std::runtime_error("--seed: 1.." + std::to_string(PN_TIMER1_MAX) + " values");
        EnvPool pool(1);
        OpResult r = pool.reset(0, PN_TEMPLATE_SAVE, seeds.data(), static_cast<int>(seeds.size()), 0, sav.c_str());
        if (r.status & PN_ST_ERROR) {
            std::fprintf(stderr, "fireredsim-worker: %s\n", r.message.c_str());
            return 1;
        }
        const SaveStartInfo *info = pool.save_start_info(0);
        if (!info)
            throw std::runtime_error("no save start information");
        if (!script_out.empty()) {
            std::string text = buttons_script(*info, sav);
            write_file(script_out, text.data(), text.size());
        }
        std::string t1;
        for (uint16_t v : info->timer1)
            t1 += (t1.empty() ? "" : ", ") + std::to_string(v);
        std::printf("{\"save\": %s, \"save_hash\": \"%s\", \"timer1\": [%s], \"frames\": %zu, \"state_hash\": \"%s\", "
                    "\"build_id\": \"%s\", \"map\": [%d, %d], \"pos\": [%d, %d]}\n",
                    json_str(sav).c_str(), hex16(info->save_hash).c_str(), t1.c_str(), info->buttons.size(),
                    hex16(pool.hash(0)).c_str(), hex16(pool.build_id()).c_str(), info->map_group, info->map_num,
                    info->x, info->y);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "fireredsim-worker: %s\n", e.what());
        return 2;
    }
}

int replay_tool_main(int argc, char **argv)
{
    try {
        std::string sav, input, dump_state;
        std::vector<uint16_t> seeds = {0};
        uint64_t frames = 0;
        // --rng-reseed STREAM,FIRST,EVERY (docs/RNG_RESEED.md): the env's host entropy stream starts
        // at STREAM; the game's PRNG state is reseeded before frame FIRST and every EVERY frames after
        // (a Python episode: STREAM = rng_stream_seed(episode seed), FIRST = the start's hand-over
        // frame, EVERY = frameskip).
        bool reseed = false;
        uint64_t reseed_stream = 0, reseed_first = 0, reseed_every = 1;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--replay") continue;
            else if (a == "--sav") sav = next();
            else if (a == "--seed") seeds = parse_u16_list(next());
            else if (a == "--input") input = next();
            else if (a == "--frames") frames = std::stoull(next(), nullptr, 0);
            else if (a == "--dump-state") dump_state = next();
            else if (a == "--quiet-faults") Port_SetFaultReport(nullptr, 0);
            else if (a == "--rng-reseed") {
                std::string v = next();
                size_t c1 = v.find(','), c2 = c1 == std::string::npos ? c1 : v.find(',', c1 + 1);
                if (c2 == std::string::npos)
                    throw std::runtime_error("--rng-reseed STREAM,FIRST,EVERY");
                reseed_stream = std::stoull(v.substr(0, c1), nullptr, 0);
                reseed_first = std::stoull(v.substr(c1 + 1, c2 - c1 - 1), nullptr, 0);
                reseed_every = std::stoull(v.substr(c2 + 1), nullptr, 0);
                if (reseed_every == 0)
                    throw std::runtime_error("--rng-reseed: EVERY must be at least 1");
                reseed = true;
            }
            else throw std::runtime_error("unknown option " + a);
        }
        InputScript script;
        uint64_t directive_frames = 0;
        if (!input.empty()) {
            std::vector<uint8_t> raw = read_file(input);
            std::string text(raw.begin(), raw.end());
            script = InputScript::parse(text, input);
            size_t pos = text.find("#! frames ");
            if (pos != std::string::npos)
                directive_frames = std::stoull(text.substr(pos + 10), nullptr, 0);
        }
        if (frames == 0)
            frames = directive_frames ? directive_frames : script.end_frame();
        Simulator &sim = Simulator::get();
        EnvConfig cfg;
        cfg.timer1_values = seeds;
        if (!sav.empty())
            cfg.flash = std::make_shared<std::vector<uint8_t>>(load_sav(sav));
        int env = sim.create_env(cfg);
        sim.host(env).rng_stream = reseed_stream;
        for (uint64_t f = 0; f < frames; f++) {
            if (reseed && f >= reseed_first && (f - reseed_first) % reseed_every == 0)
                sim.reseed_rng(env);
            sim.host(env).buttons = script.buttons_at(f);
            uint32_t r = sim.run_frame(env);
            if (r & PORT_FRAME_TERMINAL) {
                PortTerminal t;
                sim.activate(env);
                Port_GetTerminal(&t);
                std::printf("frame %llu: environment ended: %s: %s\n", static_cast<unsigned long long>(f),
                            Port_TerminalKindName(t.kind), t.message);
                return 3;
            }
        }
        if (!dump_state.empty()) {
            std::vector<uint8_t> state;
            sim.state_copy(env, state);
            write_file(dump_state, state.data(), state.size());
        }
        std::printf("done: %llu frames, final hash %s\n", static_cast<unsigned long long>(sim.host(env).frame),
                    hex16(sim.hash_state(env)).c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "fireredsim-worker: %s\n", e.what());
        return 2;
    }
}

} // namespace pn
