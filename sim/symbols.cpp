// fireredsim: ELF32 symbol table reader (see symbols.h).
#include "symbols.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace pn {

namespace {

#pragma pack(push, 1)
struct Elf32Ehdr {
    unsigned char ident[16];
    uint16_t type, machine;
    uint32_t version, entry, phoff, shoff, flags;
    uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};
struct Elf32Shdr {
    uint32_t name, type, flags, addr, offset, size, link, info, addralign, entsize;
};
struct Elf32Sym {
    uint32_t name, value, size;
    unsigned char info, other;
    uint16_t shndx;
};
#pragma pack(pop)

constexpr uint32_t SHT_SYMTAB = 2;
constexpr unsigned STT_OBJECT = 1;
constexpr unsigned STT_FUNC = 2;

} // namespace

bool SymbolTable::load(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < sizeof(Elf32Ehdr))
        return false;
    Elf32Ehdr eh;
    std::memcpy(&eh, data.data(), sizeof(eh));
    if (std::memcmp(eh.ident, "\x7f" "ELF", 4) != 0 || eh.ident[4] != 1 /* ELFCLASS32 */)
        return false;
    std::vector<Elf32Shdr> sh(eh.shnum);
    for (unsigned i = 0; i < eh.shnum; i++)
        std::memcpy(&sh[i], data.data() + eh.shoff + i * eh.shentsize, sizeof(Elf32Shdr));
    by_addr_.clear();
    for (const auto &s : sh) {
        if (s.type != SHT_SYMTAB)
            continue;
        const Elf32Shdr &strtab = sh[s.link];
        const char *strs = data.data() + strtab.offset;
        size_t n = s.size / sizeof(Elf32Sym);
        for (size_t i = 0; i < n; i++) {
            Elf32Sym sym;
            std::memcpy(&sym, data.data() + s.offset + i * sizeof(Elf32Sym), sizeof(sym));
            unsigned type = sym.info & 0xF;
            if ((type != STT_FUNC && type != STT_OBJECT) || sym.shndx == 0 || sym.name == 0)
                continue;
            by_addr_.push_back(Symbol{sym.value, sym.size, type == STT_FUNC, std::string(strs + sym.name)});
        }
    }
    std::stable_sort(by_addr_.begin(), by_addr_.end(), [](const Symbol &a, const Symbol &b) {
        if (a.addr != b.addr)
            return a.addr < b.addr;
        return a.is_func && !b.is_func;
    });
    return !by_addr_.empty();
}

const std::string &SymbolTable::name_at(uint32_t addr) const
{
    static const std::string empty;
    auto it = std::lower_bound(by_addr_.begin(), by_addr_.end(), addr,
                               [](const Symbol &s, uint32_t a) { return s.addr < a; });
    if (it != by_addr_.end() && it->addr == addr)
        return it->name;
    return empty;
}

std::string SymbolTable::describe(uint32_t addr) const
{
    char buf[32];
    if (addr == 0)
        return "NULL";
    const std::string &exact = name_at(addr);
    if (!exact.empty())
        return exact;
    auto it = std::upper_bound(by_addr_.begin(), by_addr_.end(), addr,
                               [](uint32_t a, const Symbol &s) { return a < s.addr; });
    if (it != by_addr_.begin()) {
        --it;
        if (addr < it->addr + it->size) {
            std::snprintf(buf, sizeof buf, "+0x%x", addr - it->addr);
            return it->name + buf;
        }
    }
    std::snprintf(buf, sizeof buf, "0x%08x", addr);
    return buf;
}

std::optional<uint32_t> SymbolTable::addr_of(const std::string &name) const
{
    for (const auto &s : by_addr_)
        if (s.name == name)
            return s.addr;
    return std::nullopt;
}

const SymbolTable &self_symbols()
{
    static SymbolTable table = [] {
        SymbolTable t;
        t.load();
        return t;
    }();
    return table;
}

} // namespace pn
