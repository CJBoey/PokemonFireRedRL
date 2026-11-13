#!/usr/bin/env python3
"""Post-link check of the mirror memory layout.

usage: check_addresses.py [--quiet] [--rom-symbols TSV] [--list] EXECUTABLE

Reads the linked executable's section headers and symbol table (and
EXECUTABLE.map, the link map) and fails (exit 1) unless

  1. the state regions are NOBITS sections at the GBA's addresses with the
     GBA's sizes (EWRAM 0x02000000/256 KiB, IWRAM 0x03000000/32 KiB, I/O
     0x04000000, palette 0x05000000, VRAM 0x06000000, OAM 0x07000000, flash
     0x0E000000) and .pnstate is at 0x10000000;
  2. every RAM symbol of the retail ROM (reference/rom_symbols_detailed.tsv,
     907 in EWRAM/IWRAM) has a native symbol with the same name (static
     locals: the same base name, their numeric suffixes are compiler-specific;
     aliases from layout.py) at the same address with the same size, or is a
     documented gap (layout.DOCUMENTED_GAPS) whose bytes no native symbol uses;
  3. every native data symbol inside EWRAM/IWRAM is one of those;
  4. no input section of the game or platform archives that is writable lies
     outside .gba_ewram/.gba_iwram/.pnstate (host-only .host.* sections
     excepted), and no .mirror.*/.pnstate.* section was left to the linker's
     orphan placement.

The independent matching (by address in the linked executable) makes this a
check of gen_ld.py's output, not a restatement of it.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import elf32  # noqa: E402
import layout  # noqa: E402

GAME_LIBS = ("libfirered.a", "libplatform.a")
STATE_SECTIONS = {r.section for r in layout.REGIONS} | {layout.PNSTATE_SECTION}


def parse_map(mapfile):
    """[(output section, input section, size, object)] for input sections with a size."""
    out_sec = None
    pending = None
    res = []
    with open(mapfile, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^(\S+)\s+0x[0-9a-f]+\s+0x[0-9a-f]+", line)
            if m:
                out_sec = m.group(1)
                pending = None
                continue
            m = re.match(r"^(\S+)\s*$", line)
            if m and not line.startswith(" "):
                out_sec = m.group(1)
                pending = None
                continue
            m = re.match(r"^ (\S+)\s*$", line)
            if m:
                pending = m.group(1)
                continue
            m = re.match(r"^ (\S+)?\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S.*)$", line)
            if not m:
                pending = None
                continue
            in_sec = m.group(1) or pending
            pending = None
            size = int(m.group(3), 16)
            if size:
                res.append((out_sec, in_sec, size, m.group(4).strip()))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exe")
    ap.add_argument("--rom-symbols", default=layout.ROM_SYMBOLS_DETAILED)
    ap.add_argument("--quiet", action="store_true", help="one line on success")
    ap.add_argument("--list", action="store_true", help="print every ROM RAM symbol with its native match")
    args = ap.parse_args()
    errors = []

    secs, syms = elf32.read(args.exe)
    by_name = {s.name: s for s in secs}

    # 1. regions -------------------------------------------------------------------------
    for r in layout.REGIONS:
        s = by_name.get(r.section)
        if s is None:
            errors.append("section %s missing" % r.section)
        elif s.addr != r.addr or s.size != r.size or s.type != elf32.SHT_NOBITS or not s.writable_alloc:
            errors.append("section %s: 0x%08x size 0x%x type %d (want 0x%08x size 0x%x NOBITS, writable)"
                          % (r.section, s.addr, s.size, s.type, r.addr, r.size))
    pn = by_name.get(layout.PNSTATE_SECTION)
    if pn is None or pn.addr != layout.PNSTATE_ADDR or pn.type != elf32.SHT_NOBITS or pn.size == 0:
        errors.append("section %s missing or not NOBITS at 0x%08x" % (layout.PNSTATE_SECTION, layout.PNSTATE_ADDR))

    # 2./3. symbols ----------------------------------------------------------------------
    ram_secs = {s.index for s in secs if s.name in (".gba_ewram", ".gba_iwram")}
    native = [y for y in syms if y.shndx in ram_secs and y.type in (elf32.STT_OBJECT, elf32.STT_NOTYPE)
              and not y.name.startswith("__gba_")]
    at_addr = {}
    for y in native:
        at_addr.setdefault(y.value, []).append(y)
    rom = layout.load_rom_ram_symbols(args.rom_symbols)
    matched = set()
    n_ok = 0
    gaps = []
    listing = []
    for rs in rom:
        tgt = layout.native_target(rs)
        if tgt is None:
            gaps.append(rs)
            for y in native:
                if y.size and y.value < rs.addr + max(rs.size, 1) and rs.addr < y.value + y.size:
                    errors.append("documented gap %s (0x%08x, %d bytes) overlaps native %s (0x%08x, %d)"
                                  % (rs.name, rs.addr, rs.size, y.name, y.value, y.size))
            listing.append("%-36s 0x%08x %6d  GAP (%s)" % (rs.name, rs.addr, rs.size, layout.gap_reason(rs)[:60]))
            continue
        _obj, nname = tgt
        base, num = layout.base_name(nname)
        cands = [y for y in at_addr.get(rs.addr, [])
                 if y.name == nname or (num is not None and layout.base_name(y.name)[0] == base
                                        and layout.base_name(y.name)[1] is not None)]
        good = [y for y in cands if y.size == rs.size]
        if not good:
            what = ", ".join("%s (%d bytes)" % (y.name, y.size) for y in at_addr.get(rs.addr, [])) or "nothing"
            errors.append("ROM %s (%s) at 0x%08x, %d bytes: native has %s there"
                          % (rs.name, rs.obj, rs.addr, rs.size, what))
            listing.append("%-36s 0x%08x %6d  MISSING" % (rs.name, rs.addr, rs.size))
            continue
        matched.add(id(good[0]))
        n_ok += 1
        listing.append("%-36s 0x%08x %6d  %s" % (rs.name, rs.addr, rs.size, good[0].name))
    for y in native:
        if id(y) not in matched and y.size:
            errors.append("native %s at 0x%08x (%d bytes) has no ROM RAM symbol at that address"
                          % (y.name, y.value, y.size))

    # 4. link map --------------------------------------------------------------------------
    mapfile = args.exe + ".map"
    sec_flags = {s.name: s for s in secs}
    if not os.path.exists(mapfile):
        errors.append("link map %s not found" % mapfile)
    else:
        for out_sec, in_sec, size, obj in parse_map(mapfile):
            from_game = any(lib in obj for lib in GAME_LIBS)
            if in_sec and (in_sec.startswith(".mirror.") or in_sec.startswith(".pnstate")) \
                    and out_sec not in STATE_SECTIONS:
                errors.append("%s %s (%d bytes) was placed in %s (orphan?)" % (obj, in_sec, size, out_sec))
                continue
            if not from_game or out_sec in STATE_SECTIONS:
                continue
            o = sec_flags.get(out_sec)
            writable = o is not None and bool(o.flags & elf32.SHF_WRITE) and bool(o.flags & elf32.SHF_ALLOC)
            if writable and not (in_sec or "").startswith(".host"):
                errors.append("writable game/platform data outside the state regions: %s %s (%d bytes) -> %s"
                              % (obj, in_sec, size, out_sec))

    if args.list:
        print("\n".join(listing))
    if errors:
        for e in errors[:200]:
            print("check_addresses: FAIL: " + e)
        if len(errors) > 200:
            print("check_addresses: ... %d more" % (len(errors) - 200))
        sys.exit(1)
    pn_kib = (pn.size + 1023) // 1024
    if args.quiet:
        print("check_addresses: OK: %d/%d ROM RAM symbols at their ROM address and size, %d documented gaps; "
              ".pnstate %d KiB" % (n_ok, len(rom), len(gaps), pn_kib))
    else:
        print("check_addresses: OK")
        print("  %d ROM RAM symbols (EWRAM/IWRAM): %d at their ROM address with their ROM size, "
              "%d documented gaps (%s)" % (len(rom), n_ok, len(gaps), ", ".join(sorted({g.name for g in gaps}))))
        print("  %d native data symbols in EWRAM/IWRAM, all of them ROM symbols" % len(native))
        for r in layout.REGIONS:
            print("  %-6s 0x%08x..0x%08x (%d KiB)" % (r.name, r.addr, r.addr + r.size, r.size // 1024 or 1))
        print("  pnstate 0x%08x..0x%08x (%d KiB, native-only state incl. the game stack)"
              % (pn.addr, pn.addr + pn.size, pn_kib))


if __name__ == "__main__":
    main()
