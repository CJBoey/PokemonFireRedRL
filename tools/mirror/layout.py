"""Shared definitions of the mirror memory layout (every game RAM variable at its retail ROM address).

The native executable maps the GBA's memory regions at the GBA's addresses and
puts every game RAM variable at its retail ROM address. This module knows the
region table, reads the ROM's RAM symbols (reference/rom_symbols_detailed.tsv)
and says, for each of them, which native object/symbol implements it or why it
has no native counterpart (a documented gap). Used by gen_ld.py (link time),
check_addresses.py (post-link, gate 2) and compare_ram.py (gate 4).
"""
import os
import re
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ROM_SYMBOLS_DETAILED = os.path.join(ROOT, "reference", "rom_symbols_detailed.tsv")
ROM_SYMBOLS = os.path.join(ROOT, "reference", "rom_symbols.txt")

Region = namedtuple("Region", "name addr size section lma snapshot")

# The state regions in snapshot order. `section` is the output section of the
# generated linker script; `lma` its load address (any value above the
# executable's image: it only keeps the ELF headers out of these segments, see
# gen_ld.py). The oracle's dump files are named after `name` (iwram.bin, ...).
REGIONS = [
    Region("ewram", 0x02000000, 0x40000, ".gba_ewram", 0x42000000, True),
    Region("iwram", 0x03000000, 0x8000, ".gba_iwram", 0x43000000, True),
    Region("io", 0x04000000, 0x400, ".gba_io", 0x44000000, True),
    Region("pltt", 0x05000000, 0x400, ".gba_pltt", 0x45000000, True),
    Region("vram", 0x06000000, 0x18000, ".gba_vram", 0x46000000, True),
    Region("oam", 0x07000000, 0x400, ".gba_oam", 0x47000000, True),
    Region("flash", 0x0E000000, 0x20000, ".gba_flash", 0x4E000000, True),
]
PNSTATE_ADDR = 0x10000000
PNSTATE_SECTION = ".pnstate"
PNSTATE_LMA = 0x50000000
REGION_BY_NAME = {r.name: r for r in REGIONS}

# GBA RAM: EWRAM and IWRAM hold the game's variables.
RAM_REGIONS = ("ewram", "iwram")
GBA_ROM_RANGE = (0x08000000, 0x0A000000)

# The top of IWRAM on the GBA: the stacks (user 0x03007F00, IRQ 0x03007FA0,
# supervisor 0x03007FE0, growing down towards the last variable, `end` =
# 0x03007480) and three fixed words used by the BIOS and the game.
IWRAM_END_OF_VARIABLES = 0x03007480
IWRAM_FIXED_WORDS = 0x03007FF0  # SOUND_INFO_PTR 0x03007FF0, INTR_CHECK 0x03007FF8, INTR_VECTOR 0x03007FFC

HEAP_SIZE = 0x1C000  # gHeap is a linker-script symbol without an ELF size

RomSym = namedtuple("RomSym", "name addr size nm_type section obj")

# ROM RAM symbols that are implemented by a differently named native object
# (or symbol). Keys: (ROM object, ROM name) or (ROM object, None) for all of
# the object's symbols. Values: (native object below build/native, native name
# or None = same name).
ALIASES = {
    ("linker-script", "gHeap"): ("platform/hw.o", "gHeap"),
    ("src/m4a_1.o", None): ("platform/sound.o", None),  # C port of m4a_1.s
}

# ROM RAM symbols with no native counterpart: their bytes stay reserved (zero)
# in the native EWRAM/IWRAM. Each entry: (predicate on RomSym, reason).
DOCUMENTED_GAPS = [
    (lambda s: s.obj.startswith("../../tools/agbcc/lib/libc.a"),
     "agbcc newlib data (malloc arena, impure/locale data, syscall stubs, errno): the game never "
     "calls newlib's malloc/stdio; the native build uses the host C library outside the game's "
     "memory and pret's own gHeap allocator"),
    (lambda s: s.obj.startswith("../../tools/agbcc/lib/libgcc.a"),
     "agbcc libgcc soft-float NaN constants (thenan): no floating point in game logic"),
    (lambda s: s.obj == "linker-script" and s.name == "end",
     "linker-script symbol `end` (size 0) marks the end of IWRAM variables"),
]

# Native game variables (writable sections of game objects) that have no ROM
# RAM counterpart. They go to .pnstate; each needs a reason, also listed in
# game/NATIVE_CHANGES.md. Keys: (native object below build/native, symbol).
NATIVE_ONLY = {}


def native_rel(path, root=None):
    """Object path below the object directory (game/Makefile.native OBJ_DIR):
    '<root>/src/main.o' or 'build/native/src/main.o' -> 'src/main.o'.

    `root` (default: $PN_OBJ_ROOT, exported by the makefile) names the object directory;
    an absolute path must lie below it. Without a root the path must contain
    'build/native/' (the makefile's default OBJ_DIR) or is returned unchanged."""
    p = path.replace(os.sep, "/")
    root = root or os.environ.get("PN_OBJ_ROOT")
    if root and os.path.isabs(p):
        r = os.path.abspath(root).replace(os.sep, "/").rstrip("/") + "/"
        if p.startswith(r):
            return p[len(r):]
        raise ValueError("%s is not below the object directory %s" % (path, root))
    key = "build/native/"
    i = p.rfind(key)
    return p[i + len(key):] if i >= 0 else p


def object_tag(path, root=None):
    """'<root>/src/main.o' or 'src/main.o' -> 'src_main' (section-name tag)."""
    rel = native_rel(path, root)
    if rel.endswith(".o"):
        rel = rel[:-2]
    return re.sub(r"[^A-Za-z0-9_]", "_", rel)


def rom_object_key(obj):
    """'src/main.o (common_data)' -> 'src/main.o'."""
    return re.sub(r"\s*\(common_data\)$", "", obj.strip())


def base_name(name):
    """Static locals carry a compiler-specific suffix on each side ('i.96' / 'i.0')."""
    m = re.match(r"^(.*)\.(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def region_of(addr):
    for r in REGIONS:
        if r.addr <= addr < r.addr + r.size:
            return r
    return None


def load_rom_ram_symbols(path=ROM_SYMBOLS_DETAILED):
    """Every ROM symbol inside EWRAM or IWRAM (907 for FireRed v1.0)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, addr, size, nm_type, section, obj = line.rstrip("\n").split("\t")
            addr = int(addr, 16)
            r = region_of(addr)
            if r is None or r.name not in RAM_REGIONS:
                continue
            size = int(size)
            if name == "gHeap" and obj == "linker-script":
                size = HEAP_SIZE
            out.append(RomSym(name, addr, size, nm_type, section, rom_object_key(obj)))
    return out


def gap_reason(sym):
    for pred, reason in DOCUMENTED_GAPS:
        if pred(sym):
            return reason
    return None


def native_target(sym):
    """(native object, native name or None) implementing a ROM RAM symbol, or None for a gap."""
    if gap_reason(sym) is not None:
        return None
    for key in ((sym.obj, sym.name), (sym.obj, None)):
        if key in ALIASES:
            obj, name = ALIASES[key]
            return obj, name or sym.name
    return sym.obj, sym.name
