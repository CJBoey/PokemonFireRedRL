#!/usr/bin/env python3
"""Generate the mirror-layout linker script.

usage: gen_ld.py [--rom-symbols TSV] --objs LIST [--objs LIST ...] -o mirror.ld
                 [--obj-root DIR] [--placement-out TSV] [--quiet]

LIST files name one object per line (<OBJ_DIR>/libfirered.a.objs,
libplatform.a.objs; OBJ_DIR is game/Makefile.native's object directory, one per
CMake build directory, default game/build/native; --obj-root DIR names it, default:
the directory of the first LIST). Every object was post-processed by
game/native/mirror_sections.py: each game RAM variable sits alone in a section
named .mirror.<ewram|iwram>.<object>.<variable>, native-only platform state in
.pnstate.<object>.<variable>.

For each of the ROM's RAM symbols (reference/rom_symbols_detailed.tsv, 907 in
EWRAM/IWRAM) the script finds the native object and symbol that implement it
(same object path and name; static locals by object + base name, their
numeric suffixes differ between agbcc and GCC; explicit aliases and
documented gaps in layout.py) and places the symbol's section so that the
symbol lands on the ROM address:

  .gba_ewram 0x02000000 (NOLOAD) : AT(0x42000000)
  {
      __gba_ewram_start = .;
      . = 0x00000; *(.mirror.ewram.platform_hw.gHeap)      /* gHeap ... */
      . = 0x1c000; *(.mirror.ewram.src_main.gDecompressionBuffer)
      ...
      . = 0x40000;
      __gba_ewram_end = .;
  }

Inside an output section `.` is an offset from the section start; entries are
sorted by address, so an overlap or an over-aligned input section would move
`.` backwards and fail the link. The I/O, palette, VRAM, OAM and flash regions
are empty NOLOAD sections at their GBA addresses; .pnstate (0x10000000) gets
the game stack first, then every other .pnstate.* section. The AT() load
addresses are only there so that GNU ld keeps the ELF headers in the
executable's first segment (at 0x08000000) instead of mapping them in front of
EWRAM; the kernel maps every PT_LOAD at its p_vaddr.

The script fails loudly (exit 1) on: a ROM RAM symbol without a native
counterpart that is not a documented gap, a size mismatch, an ambiguous match,
a .mirror section that would get two different addresses, a game or platform
.mirror section left unplaced (unless listed in layout.NATIVE_ONLY), a kind
mismatch (an EWRAM variable in an IWRAM section or vice versa), overlapping or
misaligned placements.
"""
import argparse
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import elf32  # noqa: E402
import layout  # noqa: E402


class MirrorSection:
    def __init__(self, obj, sec, syms):
        self.obj = obj            # native object below the object directory ('src/main.o')
        self.name = sec.name      # '.mirror.iwram.src_main.gMain'
        self.kind = sec.name.split(".")[2]
        self.size = sec.size
        self.align = max(sec.align, 1)
        self.syms = syms          # [elf32.Symbol] (value = offset in the section)
        self.addr = None          # placement
        self.placed_by = []       # ROM symbols that fixed the address


def read_objects(list_files):
    objs = []
    for lf in list_files:
        base = os.path.dirname(os.path.abspath(lf))
        # The makefile writes absolute paths. Older lists hold paths relative to
        # game/ (build/native/...); the list file itself lived in game/build/native.
        game_dir = os.path.abspath(os.path.join(base, "..", ".."))
        with open(lf) as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                if not os.path.isabs(p):
                    p = os.path.join(game_dir, p)
                objs.append(p)
    return objs


def fail(errors):
    for e in errors:
        sys.stderr.write("gen_ld.py: ERROR: %s\n" % e)
    sys.stderr.write("gen_ld.py: %d error(s); no linker script written\n" % len(errors))
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom-symbols", default=layout.ROM_SYMBOLS_DETAILED)
    ap.add_argument("--objs", action="append", required=True, help="file listing objects (one per line)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--obj-root", help="the objects' directory (OBJ_DIR); default: the directory of the first --objs file")
    ap.add_argument("--placement-out", help="write the symbol placement table (TSV)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    errors = []
    rom = layout.load_rom_ram_symbols(args.rom_symbols)

    # ---- native objects -------------------------------------------------------------
    sections = []                       # all MirrorSections
    by_obj = defaultdict(list)          # native object -> [MirrorSection]
    tags = {}
    obj_root = os.path.abspath(args.obj_root) if args.obj_root else os.path.dirname(os.path.abspath(args.objs[0]))
    for path in read_objects(args.objs):
        rel = layout.native_rel(path, obj_root)
        secs, syms = elf32.read(path)
        by_sec = elf32.symbols_by_section(secs, syms)
        for s in secs:
            if not s.name.startswith(".mirror."):
                continue
            tag = s.name.split(".")[3]
            if tag != layout.object_tag(rel):
                errors.append("%s: section %s does not carry the object's tag %s"
                              % (rel, s.name, layout.object_tag(rel)))
            if tags.setdefault(tag, rel) != rel:
                errors.append("objects %s and %s have the same section tag %s" % (tags[tag], rel, tag))
            ms = MirrorSection(rel, s, [y for y in by_sec.get(s.index, [])
                                        if y.type in (elf32.STT_OBJECT, elf32.STT_NOTYPE)])
            sections.append(ms)
            by_obj[rel].append(ms)
    names = defaultdict(list)
    for ms in sections:
        names[ms.name].append(ms)
    for n, lst in names.items():
        if len(lst) > 1:
            errors.append("section name %s is not unique (%s)" % (n, ", ".join(m.obj for m in lst)))

    # ---- match ROM symbols to native symbols ------------------------------------------
    placements = []   # (RomSym, MirrorSection, native symbol)
    gaps = []         # (RomSym, reason)
    pending_static = defaultdict(list)  # (native obj, base) -> [RomSym] with numeric suffix
    for rs in rom:
        tgt = layout.native_target(rs)
        if tgt is None:
            gaps.append((rs, layout.gap_reason(rs)))
            continue
        nobj, nname = tgt
        if nobj not in by_obj:
            errors.append("ROM %s (%s, 0x%08x): native object %s has no RAM sections"
                          % (rs.name, rs.obj, rs.addr, nobj))
            continue
        exact = [(ms, y) for ms in by_obj[nobj] for y in ms.syms if y.name == nname]
        if len(exact) == 1:
            placements.append((rs, exact[0][0], exact[0][1]))
            continue
        if len(exact) > 1:
            errors.append("ROM %s (%s): %d native symbols named %s in %s" % (rs.name, rs.obj, len(exact), nname, nobj))
            continue
        base, num = layout.base_name(nname)
        if num is None:
            errors.append("ROM %s (%s, 0x%08x, %d bytes): no native symbol %s in %s"
                          % (rs.name, rs.obj, rs.addr, rs.size, nname, nobj))
            continue
        pending_static[(nobj, base)].append(rs)
    # Static locals: same object and base name; several with one base name are
    # paired in the order of their suffixes (both compilers number them in
    # order of appearance), then sizes must agree.
    for (nobj, base), roms in pending_static.items():
        cands = []
        for ms in by_obj[nobj]:
            for y in ms.syms:
                b, n = layout.base_name(y.name)
                if b == base and n is not None:
                    cands.append((n, ms, y))
        roms.sort(key=lambda s: layout.base_name(s.name)[1])
        cands.sort(key=lambda c: c[0])
        if len(cands) != len(roms):
            # Try to pair by size when the counts differ.
            if len(roms) == 1:
                same = [c for c in cands if c[2].size == roms[0].size]
                if len(same) == 1:
                    placements.append((roms[0], same[0][1], same[0][2]))
                    continue
            errors.append("static %s in %s: ROM has %s, native has %s"
                          % (base, nobj, [s.name for s in roms], [c[2].name for c in cands]))
            continue
        for rs, (_n, ms, y) in zip(roms, cands):
            placements.append((rs, ms, y))

    # ---- section addresses ------------------------------------------------------------
    for rs, ms, y in placements:
        region = layout.region_of(rs.addr)
        if ms.kind != region.name:
            errors.append("ROM %s at 0x%08x is in %s, native %s is in a %s section (%s)"
                          % (rs.name, rs.addr, region.name, y.name, ms.kind, ms.name))
        if y.size != rs.size:
            errors.append("size mismatch: ROM %s (%s) %d bytes, native %s (%s) %d bytes"
                          % (rs.name, rs.obj, rs.size, y.name, ms.obj, y.size))
        addr = rs.addr - y.value
        if ms.addr is not None and ms.addr != addr:
            errors.append("section %s: ROM %s wants it at 0x%08x, %s at 0x%08x"
                          % (ms.name, rs.name, addr, ms.placed_by[0].name, ms.addr))
        ms.addr = addr
        ms.placed_by.append(rs)
    native_only = []
    for ms in sections:
        if ms.addr is not None:
            continue
        keys = [(ms.obj, y.name) for y in ms.syms]
        if any(k in layout.NATIVE_ONLY for k in keys):
            native_only.append(ms)
            continue
        errors.append("native RAM section %s (%s, %d bytes: %s) has no ROM counterpart "
                      "(give it one, or list it in layout.NATIVE_ONLY with a reason)"
                      % (ms.name, ms.obj, ms.size, ", ".join(y.name for y in ms.syms) or "no symbol"))

    # ---- overlaps, alignment, bounds ------------------------------------------------------
    per_region = defaultdict(list)
    for ms in sections:
        if ms.addr is None:
            continue
        r = layout.region_of(ms.addr)
        if r is None or ms.addr + ms.size > r.addr + r.size:
            errors.append("section %s at 0x%08x (+%d) leaves its region" % (ms.name, ms.addr, ms.size))
            continue
        if ms.addr % ms.align:
            errors.append("section %s at 0x%08x is not %d-aligned (native alignment > ROM's)"
                          % (ms.name, ms.addr, ms.align))
        per_region[r.name].append(ms)
    for rname, lst in per_region.items():
        lst.sort(key=lambda m: m.addr)
        for a, b in zip(lst, lst[1:]):
            if a.addr + a.size > b.addr:
                errors.append("%s: %s (0x%08x+%d) overlaps %s (0x%08x)"
                              % (rname, a.name, a.addr, a.size, b.name, b.addr))
    if errors:
        fail(errors)

    # ---- emit -------------------------------------------------------------------------
    out = []
    out.append("/* Generated by tools/mirror/gen_ld.py from %s and the native objects."
               % os.path.relpath(args.rom_symbols, layout.ROOT))
    out.append("   DO NOT EDIT. Mirror memory layout, docs/MIRROR_LAYOUT.md: every game RAM")
    out.append("   variable at its retail ROM address, the GBA's other memory regions at their")
    out.append("   addresses, native-only state in .pnstate. All sections are NOLOAD: RAM starts")
    out.append("   zeroed, as after the BIOS's RegisterRamReset. AT() keeps the ELF headers in")
    out.append("   the executable's first segment (see gen_ld.py). */")
    out.append("SECTIONS")
    out.append("{")
    for r in layout.REGIONS:
        out.append("  %s 0x%08x (NOLOAD) : AT(0x%08x)" % (r.section, r.addr, r.lma))
        out.append("  {")
        out.append("    __gba_%s_start = .;" % r.name)
        for ms in per_region.get(r.name, []):
            syms = ", ".join("%s %s 0x%08x" % (rs.name, rs.obj, rs.addr) for rs in ms.placed_by)
            out.append("    . = 0x%05x; *(%s)  /* %s */" % (ms.addr - r.addr, ms.name, syms))
        out.append("    . = 0x%05x;" % r.size)
        out.append("    __gba_%s_end = .;" % r.name)
        out.append("  }")
    out.append("  %s 0x%08x (NOLOAD) : AT(0x%08x)" % (layout.PNSTATE_SECTION, layout.PNSTATE_ADDR,
                                                    layout.PNSTATE_LMA))
    out.append("  {")
    out.append("    __pnstate_start = .;")
    out.append("    *(.pnstate.stack)  /* the game stack first: an overflow leaves mapped memory */")
    out.append("    . = ALIGN(64);")
    for ms in native_only:
        out.append("    *(%s)  /* native-only game variable, layout.NATIVE_ONLY */" % ms.name)
    out.append("    *(.pnstate .pnstate.*)")
    out.append("    . = ALIGN(0x1000);")
    out.append("    __pnstate_end = .;")
    out.append("  }")
    out.append("}")
    out.append("INSERT AFTER .bss;")
    tmp = args.output + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, args.output)

    if args.placement_out:
        tmp = args.placement_out + ".tmp"
        with open(tmp, "w") as f:
            f.write("# rom_name\trom_object\taddr\tsize\tnative_object\tnative_name\tsection\n")
            rows = [(rs.addr, rs.name, rs.obj, rs.size, ms.obj, y.name, ms.name) for rs, ms, y in placements]
            rows += [(rs.addr, rs.name, rs.obj, rs.size, "-", "-", "GAP: " + reason) for rs, reason in gaps]
            for addr, name, obj, size, nobj, nname, sec in sorted(rows):
                f.write("%s\t%s\t0x%08x\t%d\t%s\t%s\t%s\n" % (name, obj, addr, size, nobj, nname, sec))
        os.replace(tmp, args.placement_out)

    if not args.quiet:
        nsec = sum(len(v) for v in per_region.values())
        print("gen_ld.py: %d ROM RAM symbols: %d placed (%d sections), %d documented gaps, "
              "%d native-only game sections -> %s"
              % (len(rom), len(placements), nsec, len(gaps), len(native_only), args.output))


if __name__ == "__main__":
    main()
