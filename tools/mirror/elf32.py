"""Minimal ELF32 (little-endian) reader: section headers and symbol tables.

Used by the M1b mirror-layout tools (gen_ld.py, check_addresses.py,
compare_ram.py) and by game/native/mirror_sections.py. Standard library only.

    secs, syms = elf32.read("build/native/src/main.o")
    secs[i].name, .type, .flags, .addr, .size, .align, .index
    syms[j].name, .value, .size, .type, .bind, .shndx, .section (name or None)
"""
import struct

SHT_SYMTAB = 2
SHT_NOBITS = 8
SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4

STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2
STT_SECTION = 3
STT_FILE = 4

STB_LOCAL = 0
STB_GLOBAL = 1
STB_WEAK = 2

SHN_UNDEF = 0
SHN_ABS = 0xFFF1
SHN_COMMON = 0xFFF2


class Section:
    __slots__ = ("index", "name", "type", "flags", "addr", "offset", "size", "link", "info",
                 "align", "entsize")

    def __init__(self, index, name, typ, flags, addr, offset, size, link, info, align, entsize):
        self.index = index
        self.name = name
        self.type = typ
        self.flags = flags
        self.addr = addr
        self.offset = offset
        self.size = size
        self.link = link
        self.info = info
        self.align = align
        self.entsize = entsize

    @property
    def writable_alloc(self):
        return bool(self.flags & SHF_ALLOC) and bool(self.flags & SHF_WRITE)

    def __repr__(self):
        return "Section(%r, size=0x%x)" % (self.name, self.size)


class Symbol:
    __slots__ = ("name", "value", "size", "type", "bind", "shndx", "section")

    def __init__(self, name, value, size, typ, bind, shndx, section):
        self.name = name
        self.value = value
        self.size = size
        self.type = typ
        self.bind = bind
        self.shndx = shndx
        self.section = section

    def __repr__(self):
        return "Symbol(%r, 0x%x, %d, %r)" % (self.name, self.value, self.size, self.section)


def read(path):
    """(sections, symbols) of an ELF32 little-endian file (object or executable)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
        raise ValueError("%s: not an ELF32 little-endian file" % path)
    (e_shoff,) = struct.unpack_from("<I", data, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    raw = []
    for i in range(e_shnum):
        raw.append(struct.unpack_from("<IIIIIIIIII", data, e_shoff + i * e_shentsize))
    shstr = raw[e_shstrndx]

    def cstr(table_off, off):
        end = data.index(b"\0", table_off + off)
        return data[table_off + off:end].decode("latin-1")

    secs = []
    for i, (name, typ, flags, addr, offset, size, link, info, align, entsize) in enumerate(raw):
        secs.append(Section(i, cstr(shstr[4], name) if i else "", typ, flags, addr, offset, size,
                            link, info, align, entsize))
    syms = []
    for s in secs:
        if s.type != SHT_SYMTAB:
            continue
        strtab = secs[s.link]
        for off in range(s.offset + 16, s.offset + s.size, 16):  # entry 0 is the null symbol
            st_name, st_value, st_size, st_info, _other, st_shndx = struct.unpack_from(
                "<IIIBBH", data, off)
            sec = secs[st_shndx].name if 0 < st_shndx < len(secs) else None
            syms.append(Symbol(cstr(strtab.offset, st_name), st_value, st_size, st_info & 0xF,
                               st_info >> 4, st_shndx, sec))
    return secs, syms


def symbols_by_section(secs, syms):
    """{section index: [data symbols sorted by value]} (no section/file symbols)."""
    out = {}
    for y in syms:
        if y.type in (STT_SECTION, STT_FILE) or not y.name or y.shndx == SHN_UNDEF \
                or y.shndx >= 0xFF00:
            continue
        out.setdefault(y.shndx, []).append(y)
    for v in out.values():
        v.sort(key=lambda y: (y.value, y.name))
    return out
