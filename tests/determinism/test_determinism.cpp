// fireredsim: determinism tests (ARCHITECTURE.md section 5), run by ctest.
//
//   test_determinism replay-twice     SCRIPT  same inputs twice -> identical region hash after every frame
//   test_determinism snapshot-restore SCRIPT  snapshot at F, run to F+1000, restore, run again -> identical
//   test_determinism interleaved      SCRIPT  two envs stepped interleaved with different inputs
//                                             == each stepped alone
//   test_determinism reset            SCRIPT  reset returns to the power-on (post-init) hash
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "input_script.h"
#include "sim.h"

using namespace pn;

namespace {

int failures = 0;

#define CHECK(cond, ...)                          \
    do {                                          \
        if (!(cond)) {                            \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);             \
            std::printf("\n");                    \
            failures++;                           \
        }                                         \
    } while (0)

std::vector<uint64_t> run_hashes(Simulator &sim, int env, const InputScript &s, uint64_t from, uint64_t to)
{
    std::vector<uint64_t> h;
    for (uint64_t f = from; f < to; f++) {
        sim.host(env).buttons = s.buttons_at(f);
        sim.run_frame(env);
        h.push_back(sim.hash_state(env));
    }
    return h;
}

void replay_twice(const InputScript &s)
{
    Simulator &sim = Simulator::get();
    uint64_t n = s.end_frame() + 60;
    int a = sim.create_env();
    std::vector<uint64_t> ha = run_hashes(sim, a, s, 0, n);
    sim.destroy_env(a);
    int b = sim.create_env();
    std::vector<uint64_t> hb = run_hashes(sim, b, s, 0, n);
    size_t first_diff = n;
    for (size_t i = 0; i < n; i++)
        if (ha[i] != hb[i]) { first_diff = i; break; }
    CHECK(first_diff == n, "replays diverge at frame %zu", first_diff);
    std::printf("replay-twice: %llu frames compared, final hash %016llx\n", (unsigned long long)n,
                (unsigned long long)hb.back());
}

void snapshot_restore(const InputScript &s)
{
    Simulator &sim = Simulator::get();
    uint64_t total = s.end_frame();
    uint64_t snap_points[] = {total / 3, (2 * total) / 3, total > 200 ? total - 200 : 0};
    int env = sim.create_env();
    uint64_t done = 0;
    for (uint64_t F : snap_points) {
        // run to F
        run_hashes(sim, env, s, done, F);
        std::vector<uint8_t> snap = sim.snapshot(env);
        uint64_t h0 = sim.hash_state(env);
        std::vector<uint64_t> first = run_hashes(sim, env, s, F, F + 1000);
        sim.restore(env, snap);
        CHECK(sim.hash_state(env) == h0, "restore at %llu does not give the snapshot hash", (unsigned long long)F);
        CHECK(sim.host(env).frame == F, "restored frame counter %llu != %llu",
              (unsigned long long)sim.host(env).frame, (unsigned long long)F);
        std::vector<uint64_t> second = run_hashes(sim, env, s, F, F + 1000);
        size_t diff = first.size();
        for (size_t i = 0; i < first.size(); i++)
            if (first[i] != second[i]) { diff = i; break; }
        CHECK(diff == first.size(), "continuation after restore at %llu diverges at +%zu", (unsigned long long)F, diff);
        // also through the file format and a clone
        int c = sim.clone(env);
        sim.restore(c, snap);
        std::vector<uint64_t> third = run_hashes(sim, c, s, F, F + 1000);
        CHECK(third == first, "clone continuation after %llu diverges", (unsigned long long)F);
        sim.destroy_env(c);
        sim.restore(env, snap);
        done = F;
        std::printf("snapshot-restore: F=%llu ok (+1000 frames, hash %016llx)\n", (unsigned long long)F,
                    (unsigned long long)first.back());
    }
}

void interleaved(const InputScript &s)
{
    Simulator &sim = Simulator::get();
    uint64_t n = s.end_frame() + 30;
    // Script B: the same presses delayed by 37 frames plus extra B presses.
    InputScript sb;
    for (const auto &r : s.ranges())
        sb.add(r.start + 37, r.end + 37, r.buttons);
    InputScript sb2;
    uint64_t last = 0;
    for (const auto &r : sb.ranges()) {
        if (r.start >= last + 3 && (r.start % 5) == 0)
            sb2.add(r.start - 2, r.start - 1, BTN_B);
        sb2.add(r.start, r.end, r.buttons);
        last = r.end;
    }
    EnvConfig cb;
    cb.timer1_values = {0xBEEF, 0x1234};

    int a = sim.create_env();
    std::vector<uint64_t> alone_a = run_hashes(sim, a, s, 0, n);
    sim.destroy_env(a);
    int b = sim.create_env(cb);
    std::vector<uint64_t> alone_b = run_hashes(sim, b, sb2, 0, n);
    sim.destroy_env(b);

    a = sim.create_env();
    b = sim.create_env(cb);
    std::vector<uint64_t> ia, ib;
    for (uint64_t f = 0; f < n; f++) {
        sim.host(a).buttons = s.buttons_at(f);
        sim.run_frame(a);
        ia.push_back(sim.hash_state(a));
        // b runs two frames when f is odd and none when f % 4 == 0, to vary the interleaving
        if (f % 4 != 0 && ib.size() < n) {
            uint64_t fb = ib.size();
            sim.host(b).buttons = sb2.buttons_at(fb);
            sim.run_frame(b);
            ib.push_back(sim.hash_state(b));
        }
        if (f % 2 == 1 && ib.size() < n) {
            uint64_t fb = ib.size();
            sim.host(b).buttons = sb2.buttons_at(fb);
            sim.run_frame(b);
            ib.push_back(sim.hash_state(b));
        }
    }
    while (ib.size() < n) {
        uint64_t fb = ib.size();
        sim.host(b).buttons = sb2.buttons_at(fb);
        sim.run_frame(b);
        ib.push_back(sim.hash_state(b));
    }
    CHECK(ia == alone_a, "env A interleaved differs from A alone");
    CHECK(ib == alone_b, "env B interleaved differs from B alone");
    CHECK(alone_a.back() != alone_b.back(), "the two scripts should lead to different states");
    std::printf("interleaved: %llu frames each, A %016llx B %016llx\n", (unsigned long long)n,
                (unsigned long long)ia.back(), (unsigned long long)ib.back());
}

void reset_test(const InputScript &s)
{
    Simulator &sim = Simulator::get();
    int env = sim.create_env();
    uint64_t h_boot = sim.hash_state(env);
    run_hashes(sim, env, s, 0, s.end_frame() / 2);
    CHECK(sim.hash_state(env) != h_boot, "state did not change");
    sim.reset(env);
    CHECK(sim.hash_state(env) == h_boot, "reset does not return to the power-on state");
    CHECK(sim.host(env).frame == 0, "frame counter not reset");
    std::vector<uint64_t> again = run_hashes(sim, env, s, 0, 300);
    sim.reset(env);
    std::vector<uint64_t> again2 = run_hashes(sim, env, s, 0, 300);
    CHECK(again == again2, "runs after reset differ");
    std::printf("reset: power-on hash %016llx ok\n", (unsigned long long)h_boot);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: test_determinism {replay-twice|snapshot-restore|interleaved|reset} SCRIPT\n");
        return 2;
    }
    std::string mode = argv[1];
    InputScript s = InputScript::load(argv[2]);
    if (mode == "replay-twice") replay_twice(s);
    else if (mode == "snapshot-restore") snapshot_restore(s);
    else if (mode == "interleaved") interleaved(s);
    else if (mode == "reset") reset_test(s);
    else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2; }
    std::printf("%s: %s\n", mode.c_str(), failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
