// fireredsim: input scripts (tests/inputs/FORMAT.md).
//
//   <start_frame> <end_frame_exclusive> <BUTTONS>     BUTTONS = NONE | A+B+...
//
// Frames are logical frames (frame k = the main-loop body that starts with the
// k-th ReadKeys after power-on, k = 0 first), identical on the native
// simulator and on the mGBA oracle.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pn {

enum Button : uint16_t {
    BTN_A = 0x001, BTN_B = 0x002, BTN_SELECT = 0x004, BTN_START = 0x008,
    BTN_RIGHT = 0x010, BTN_LEFT = 0x020, BTN_UP = 0x040, BTN_DOWN = 0x080,
    BTN_R = 0x100, BTN_L = 0x200,
};

struct InputRange {
    uint64_t start;
    uint64_t end; // exclusive
    uint16_t buttons;
};

class InputScript {
public:
    // Parses a script; throws std::runtime_error with file:line on errors.
    static InputScript load(const std::string &path);
    static InputScript parse(const std::string &text, const std::string &name = "<string>");

    uint16_t buttons_at(uint64_t frame) const;
    // First frame after the last range (0 for an empty script).
    uint64_t end_frame() const { return ranges_.empty() ? 0 : ranges_.back().end; }
    const std::vector<InputRange> &ranges() const { return ranges_; }

    void add(uint64_t start, uint64_t end, uint16_t buttons);
    std::string to_text() const;

private:
    std::vector<InputRange> ranges_;
};

uint16_t parse_buttons(const std::string &s); // throws on unknown names
std::string buttons_to_string(uint16_t buttons);

} // namespace pn
