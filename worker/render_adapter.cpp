// fireredsim worker: renderer front end for the live state (see render_adapter.h).
#include "render_adapter.h"

#include <link.h>
#include <setjmp.h>
#include <signal.h>

#include <cstring>
#include <stdexcept>

#include "pn_ipc.h"
#include "pn_render.h"
#include "fireredsim_port.h"
#include "symbols.h"

namespace pn {

namespace {

// ---- fault guard -----------------------------------------------------------------------------

const int kGuardSignals[4] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL};
struct sigaction g_old[4];
bool g_installed = false;
sigjmp_buf *volatile g_jmp = nullptr;

void guard_handler(int sig, siginfo_t *si, void *ctx)
{
    if (g_jmp) {
        sigjmp_buf *j = g_jmp;
        g_jmp = nullptr;
        siglongjmp(*j, sig);
    }
    for (int k = 0; k < 4; k++) {
        if (kGuardSignals[k] != sig)
            continue;
        struct sigaction *old = &g_old[k];
        if ((old->sa_flags & SA_SIGINFO) && old->sa_sigaction) {
            old->sa_sigaction(sig, si, ctx);
            return;
        }
        if (!(old->sa_flags & SA_SIGINFO) && old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN) {
            old->sa_handler(sig);
            return;
        }
    }
    // Default action: the instruction faults again and the process ends.
    signal(sig, SIG_DFL);
}

// ---- address map ---------------------------------------------------------------------------

int collect_ro_segments(struct dl_phdr_info *info, size_t, void *data)
{
    auto *out = static_cast<std::vector<PnAddrRange> *>(data);
    if (info->dlpi_name && info->dlpi_name[0])
        return 0; // shared objects: not the game
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) &ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_LOAD || (ph.p_flags & PF_W) || ph.p_memsz == 0)
            continue;
        uintptr_t start = info->dlpi_addr + ph.p_vaddr;
        if (start > 0xFFFFFFFFu || ph.p_memsz > 0xFFFFFFFFu - start)
            continue;
        out->push_back({static_cast<uint32_t>(start), static_cast<uint32_t>(ph.p_memsz),
                        static_cast<uint32_t>(ph.p_memsz), reinterpret_cast<const uint8_t *>(start)});
    }
    return 0;
}

uint16_t rd16(const uint8_t *p, uint32_t off)
{
    return static_cast<uint16_t>(p[off] | (p[off + 1] << 8));
}

void wr16(uint8_t *p, uint32_t off, uint16_t v)
{
    p[off] = static_cast<uint8_t>(v);
    p[off + 1] = static_cast<uint8_t>(v >> 8);
}

bool hblank_irq_enabled(const uint8_t *io)
{
    return (rd16(io, 0x208) & 1) && (rd16(io, 0x200) & 0x0002) && (rd16(io, 0x004) & 0x0010);
}

struct HookCtx {
    void (*callback)(void) = nullptr;
    uint8_t *io = nullptr;
};

// Before line `line`: the HBlank IRQ of the line before (line 227's for line 0) runs
// gMain.hblankCallback with VCOUNT and the DISPSTAT status bits of that line, as HBlankIntr does
// (identical to fireredsim-render --hblank).
void hblank_hook(void *ctx, int line)
{
    HookCtx *h = static_cast<HookCtx *>(ctx);
    unsigned vcount = line == 0 ? 227u : static_cast<unsigned>(line - 1);
    uint16_t dispstat = rd16(h->io, 0x004) & ~7u;
    dispstat |= 2; // in HBlank
    if (vcount >= 160 && vcount < 227)
        dispstat |= 1;
    if ((dispstat >> 8) == vcount)
        dispstat |= 4;
    wr16(h->io, 0x004, dispstat);
    wr16(h->io, 0x006, static_cast<uint16_t>(vcount));
    h->callback();
}

uint32_t read_u32(uint32_t addr)
{
    uint32_t v;
    std::memcpy(&v, reinterpret_cast<const void *>(static_cast<uintptr_t>(addr)), 4);
    return v;
}

} // namespace

void install_render_guard()
{
    if (g_installed)
        return;
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = guard_handler;
    // On the alternate signal stack, SIGVTALRM blocked: the same as the platform's fault handler
    // (platform/memfault.c), which this handler may call.
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, SIGVTALRM);
    for (int k = 0; k < 4; k++)
        sigaction(kGuardSignals[k], &sa, &g_old[k]);
    g_installed = true;
}

LiveRenderer::LiveRenderer(const Simulator &sim)
{
    std::vector<PnAddrRange> map;
    for (const MemRegion &r : sim.regions()) {
        uintptr_t a = reinterpret_cast<uintptr_t>(r.addr);
        map.push_back({static_cast<uint32_t>(a), static_cast<uint32_t>(r.size), static_cast<uint32_t>(r.size), r.addr});
    }
    dl_iterate_phdr(collect_ro_segments, &map);
    map_count_ = map.size();
    map_storage_.resize(map.size() * sizeof(PnAddrRange));
    std::memcpy(map_storage_.data(), map.data(), map_storage_.size());
    auto g = self_symbols().addr_of("gMain");
    if (!g)
        throw std::runtime_error("symbol gMain not found in this executable");
    gmain_ = *g;
}

bool LiveRenderer::hblank_hook_needed() const
{
    uint32_t cb = read_u32(gmain_ + 0x10); // gMain.hblankCallback
    return cb != 0 && hblank_irq_enabled(Port_IoRegs());
}

bool pixel_render_opts(uint32_t code, PnRenderOpts *opts)
{
    if (pn_pixel_bytes(code) == 0)
        return false;
    if (code == PN_PIX_RGB8) {
        *opts = PnRenderOpts{PN_FMT_RGB8, 1, 0, PN_ISA_AUTO};
    } else {
        *opts = PnRenderOpts{PN_FMT_GRAY8, static_cast<int>(code & PN_PIX_SCALE_MASK),
                             (code & PN_PIX_GRAY_SUB) ? 1 : 0, PN_ISA_AUTO};
    }
    return true;
}

bool LiveRenderer::render(uint8_t *out, bool hook, const PnRenderOpts *opts)
{
    // (a copy whose address is taken: nothing here lives in a register across sigsetjmp)
    const PnRenderOpts ro = opts ? *opts : PnRenderOpts{PN_FMT_RGB8, 1, 0, PN_ISA_AUTO};
    PnAddrMap map{reinterpret_cast<const PnAddrRange *>(map_storage_.data()), map_count_};
    PnVideoMem mem{};
    mem.io = Port_IoRegs();
    mem.pltt = Port_Palette();
    mem.vram = Port_Vram();
    mem.oam = Port_Oam();
    mem.resolve = pn_resolve_addr_map;
    mem.ctx = &map;
    if (!hook) {
        if (pn_render_ex(&mem, nullptr, &ro, out, nullptr) != 0)
            throw std::runtime_error("pn_render_ex: invalid render options");
        return true;
    }
    HookCtx h;
    h.callback = reinterpret_cast<void (*)(void)>(static_cast<uintptr_t>(read_u32(gmain_ + 0x10)));
    h.io = const_cast<uint8_t *>(Port_IoRegs());
    PnRenderHooks hooks{hblank_hook, &h};
    install_render_guard();
    sigjmp_buf jb;
    if (sigsetjmp(jb, 1) == 0) {
        g_jmp = &jb;
        int rc = pn_render_ex(&mem, &hooks, &ro, out, nullptr);
        g_jmp = nullptr;
        if (rc != 0)
            throw std::runtime_error("pn_render_ex: invalid render options");
        return true;
    }
    g_jmp = nullptr;
    return false;
}

} // namespace pn
