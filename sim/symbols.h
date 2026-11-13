// fireredsim: symbol table of the running executable (ELF32 .symtab of
// /proc/self/exe), used to print gMain.callback2 and similar function pointers
// by name and to look up game globals by name. Static (local) symbols are
// included: most callbacks of the game are static functions.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pn {

struct Symbol {
    uint32_t addr;
    uint32_t size;
    bool is_func;
    std::string name;
};

class SymbolTable {
public:
    // Loads the symbols of the given ELF32 file (default: the running executable).
    bool load(const std::string &path = "/proc/self/exe");

    // Name of the symbol starting exactly at addr (functions first), or "" if none.
    const std::string &name_at(uint32_t addr) const;
    // "name" if addr is a symbol start, "name+0xN" inside a sized symbol,
    // otherwise "0x%08x".
    std::string describe(uint32_t addr) const;
    // Address of a global or static symbol by name (first match).
    std::optional<uint32_t> addr_of(const std::string &name) const;

    size_t size() const { return by_addr_.size(); }

private:
    std::vector<Symbol> by_addr_; // sorted by address, functions before objects at equal address
};

// Process-wide table for the running executable, loaded on first use.
const SymbolTable &self_symbols();

} // namespace pn
