// fireredsim observer: host-side helpers (C++, no game headers).
//
// 1. Symbol resolution with file-qualified names. sim/symbols.h keeps no STT_FILE
//    entries, so a static name defined in several files (HandleInputChooseAction x4,
//    WaitForMonSelection x3, sMenu x2, Task_TopMenuHandleInput x2, ...) cannot be told
//    apart there (docs/OBS_SOURCES.md section 0.4). GNU ld keeps every file's local
//    symbols together after its FILE entry, so "file.c:name" is resolved here as the
//    local symbol of that file (or the global of that name). A plain "name" must be
//    unique: a global, or exactly one local. Compiler clones (.part/.isra/.constprop/
//    .cold) have other names and are never matched.
// 2. Host fields of the active environment (frame counter, soft reset), which live in
//    pn::Simulator, not in game memory.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "sim.h"

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

struct Sym {
    uint32_t addr;
    bool local;
    std::string file;
};

class Table {
public:
    bool ok = false;
    std::multimap<std::string, Sym> by_name;

    Table()
    {
        std::ifstream f("/proc/self/exe", std::ios::binary);
        if (!f)
            return;
        std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (data.size() < sizeof(Elf32Ehdr))
            return;
        Elf32Ehdr eh;
        std::memcpy(&eh, data.data(), sizeof eh);
        if (std::memcmp(eh.ident, "\x7f" "ELF", 4) != 0 || eh.ident[4] != 1)
            return;
        std::vector<Elf32Shdr> sh(eh.shnum);
        for (unsigned i = 0; i < eh.shnum; i++)
            std::memcpy(&sh[i], data.data() + eh.shoff + i * eh.shentsize, sizeof(Elf32Shdr));
        for (const auto &s : sh) {
            if (s.type != 2 /* SHT_SYMTAB */)
                continue;
            const char *strs = data.data() + sh[s.link].offset;
            size_t n = s.size / sizeof(Elf32Sym);
            std::string file;
            for (size_t i = 0; i < n; i++) {
                Elf32Sym sym;
                std::memcpy(&sym, data.data() + s.offset + i * sizeof(Elf32Sym), sizeof sym);
                unsigned type = sym.info & 0xF, bind = sym.info >> 4;
                if (type == 4 /* STT_FILE */) {
                    file = sym.name ? std::string(strs + sym.name) : std::string();
                    continue;
                }
                if (type != 1 /* OBJECT */ && type != 2 /* FUNC */)
                    continue;
                if (sym.shndx == 0 || sym.name == 0 || sym.shndx >= 0xff00)
                    continue;
                Sym e;
                e.addr = sym.value;
                e.local = bind == 0;
                if (e.local)
                    e.file = file;
                by_name.emplace(std::string(strs + sym.name), e);
            }
        }
        ok = !by_name.empty();
    }

    // 0 if not found or ambiguous.
    uint32_t find(const std::string &qname, std::string *why) const
    {
        auto colon = qname.find(':');
        std::string file, name = qname;
        if (colon != std::string::npos) {
            file = qname.substr(0, colon);
            name = qname.substr(colon + 1);
        }
        auto [lo, hi] = by_name.equal_range(name);
        uint32_t global = 0, local_in_file = 0, any_local = 0;
        int globals = 0, locals_in_file = 0, locals = 0;
        for (auto it = lo; it != hi; ++it) {
            const Sym &s = it->second;
            if (!s.local) {
                if (globals == 0 || s.addr != global)
                    globals++;
                global = s.addr;
            } else {
                if (locals == 0 || s.addr != any_local)
                    locals++;
                any_local = s.addr;
                if (!file.empty() && s.file == file) {
                    if (locals_in_file == 0 || s.addr != local_in_file)
                        locals_in_file++;
                    local_in_file = s.addr;
                }
            }
        }
        if (!file.empty()) {
            if (locals_in_file == 1)
                return local_in_file;
            if (locals_in_file > 1) {
                if (why) *why = "several locals in " + file;
                return 0;
            }
            if (globals == 1)
                return global;
            if (why) *why = globals ? "ambiguous global" : "not found in " + file;
            return 0;
        }
        if (globals == 1)
            return global;
        if (globals > 1) {
            if (why) *why = "ambiguous global";
            return 0;
        }
        if (locals == 1)
            return any_local;
        if (why) *why = locals ? "ambiguous static (use file.c:name)" : "not found";
        return 0;
    }
};

const Table &table()
{
    static Table t;
    return t;
}

} // namespace

extern "C" int pn_obs_resolve(const char *const *names, uint32_t *out, int n, int verbose)
{
    const Table &t = table();
    int missing = 0;
    for (int i = 0; i < n; i++) {
        std::string why;
        out[i] = t.ok ? t.find(names[i], &why) : 0;
        if (out[i] == 0) {
            missing++;
            if (verbose)
                std::fprintf(stderr, "pnobs: cannot resolve %s (%s)\n", names[i], t.ok ? why.c_str() : "no symbol table");
        }
    }
    return missing;
}

extern "C" void pn_obs_host_info(uint64_t *frame, uint8_t *soft_reset)
{
    pn::Simulator &sim = pn::Simulator::get();
    int id = sim.active();
    if (id < 0) {
        *frame = 0;
        *soft_reset = 0;
        return;
    }
    const pn::EnvHost &h = sim.host(id);
    *frame = h.frame;
    *soft_reset = h.soft_reset;
}
