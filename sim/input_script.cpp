// fireredsim: input script parser (see input_script.h, tests/inputs/FORMAT.md).
#include "input_script.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pn {

namespace {
struct ButtonName {
    const char *name;
    uint16_t bit;
};
constexpr ButtonName kButtons[] = {
    {"A", BTN_A}, {"B", BTN_B}, {"SELECT", BTN_SELECT}, {"START", BTN_START},
    {"RIGHT", BTN_RIGHT}, {"LEFT", BTN_LEFT}, {"UP", BTN_UP}, {"DOWN", BTN_DOWN},
    {"R", BTN_R}, {"L", BTN_L},
};
} // namespace

uint16_t parse_buttons(const std::string &s)
{
    if (s == "NONE")
        return 0;
    uint16_t mask = 0;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, '+')) {
        bool found = false;
        for (const auto &b : kButtons) {
            if (tok == b.name) {
                mask |= b.bit;
                found = true;
            }
        }
        if (!found)
            throw std::runtime_error("unknown button '" + tok + "'");
    }
    return mask;
}

std::string buttons_to_string(uint16_t buttons)
{
    if (buttons == 0)
        return "NONE";
    std::string out;
    for (const auto &b : kButtons) {
        if (buttons & b.bit) {
            if (!out.empty())
                out += "+";
            out += b.name;
        }
    }
    return out;
}

InputScript InputScript::parse(const std::string &text, const std::string &name)
{
    InputScript script;
    std::stringstream in(text);
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        auto hash = line.find('#');
        if (hash != std::string::npos)
            line.resize(hash);
        std::stringstream ls(line);
        std::string a, b, c, extra;
        if (!(ls >> a))
            continue;
        try {
            if (!(ls >> b >> c) || (ls >> extra))
                throw std::runtime_error("expected '<start> <end> <BUTTONS>'");
            size_t pa = 0, pb = 0;
            uint64_t start = std::stoull(a, &pa);
            uint64_t end = std::stoull(b, &pb);
            if (pa != a.size() || pb != b.size())
                throw std::runtime_error("bad frame number");
            if (end <= start)
                throw std::runtime_error("empty or reversed range");
            if (!script.ranges_.empty() && start < script.ranges_.back().end)
                throw std::runtime_error("ranges must be ascending and non-overlapping");
            script.ranges_.push_back({start, end, parse_buttons(c)});
        } catch (const std::exception &e) {
            throw std::runtime_error(name + ":" + std::to_string(lineno) + ": " + e.what());
        }
    }
    return script;
}

InputScript InputScript::load(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open input script " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str(), path);
}

uint16_t InputScript::buttons_at(uint64_t frame) const
{
    auto it = std::upper_bound(ranges_.begin(), ranges_.end(), frame,
                               [](uint64_t f, const InputRange &r) { return f < r.start; });
    if (it == ranges_.begin())
        return 0;
    --it;
    return frame < it->end ? it->buttons : 0;
}

void InputScript::add(uint64_t start, uint64_t end, uint16_t buttons)
{
    if (end <= start)
        return;
    if (!ranges_.empty() && start < ranges_.back().end)
        throw std::runtime_error("InputScript::add: overlapping range");
    if (!ranges_.empty() && ranges_.back().end == start && ranges_.back().buttons == buttons) {
        ranges_.back().end = end;
        return;
    }
    ranges_.push_back({start, end, buttons});
}

std::string InputScript::to_text() const
{
    std::string out;
    for (const auto &r : ranges_)
        out += std::to_string(r.start) + " " + std::to_string(r.end) + " " + buttons_to_string(r.buttons) + "\n";
    return out;
}

} // namespace pn
