// fireredsim worker: renders the active environment with render/ (pn_render),
// the way fireredsim-render does (render/README.md "Native integration"):
// views = the live I/O, palette, VRAM and OAM regions; DMA addresses resolved
// through the state regions and the executable's read-only segments; optional
// HBlank IRQ hook running gMain.hblankCallback before every line.
#pragma once

#include <cstdint>
#include <vector>

#include "pn_render.h"
#include "sim.h"

namespace pn {

class LiveRenderer {
public:
    explicit LiveRenderer(const Simulator &sim);
    // True if the HBlank IRQ is enabled and gMain.hblankCallback is set: the
    // frame needs the hook to be exact (the hook writes I/O registers / OAM of
    // the live state: the caller saves and restores the state around it).
    bool hblank_hook_needed() const;
    // Renders the live state into `out` (RGB8 [160][240][3] by default, or the
    // output of `opts`: pn_render_ex). With hook, runs the game's HBlank
    // callback per line; returns false if that callback faulted (the caller
    // renders again without the hook and must restore the state).
    bool render(uint8_t *out, bool hook, const PnRenderOpts *opts = nullptr);

private:
    std::vector<uint8_t> map_storage_; // PnAddrRange[]
    size_t map_count_ = 0;
    uint32_t gmain_ = 0;
};

// PnRenderOpts of a PN_PIX_* pixel format code (pn_ipc.h); false if the code
// is not one the worker renders (pn_pixel_bytes(code) == 0).
bool pixel_render_opts(uint32_t code, PnRenderOpts *opts);

// Installs (once per process) a fault handler that turns a fault inside a
// guarded host call into a longjmp; other faults go to the previous handler.
void install_render_guard();

} // namespace pn
