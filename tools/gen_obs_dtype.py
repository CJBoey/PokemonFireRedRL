#!/usr/bin/env python3
"""Generates python/fireredsim/obs_dtype.py (NumPy structured dtype of PnObs) from
obs/observation.h, and checks it against the C compiler's layout.

    python3 tools/gen_obs_dtype.py                      write python/fireredsim/obs_dtype.py
    python3 tools/gen_obs_dtype.py --check              fail if obs_dtype.py is out of date
    python3 tools/gen_obs_dtype.py --emit-c FILE        write a C program that prints offsetof/
                                                        sizeof of every field (compiled by CMake
                                                        with the project's flags: pn_obs_layout)
    python3 tools/gen_obs_dtype.py --emit-fields-h FILE write the field table (name, offset, size,
                                                        privileged) as a C header for the tests
    python3 tools/gen_obs_dtype.py --check-layout BIN   run BIN (pn_obs_layout) and compare its
                                                        output with the header parse and with the
                                                        generated module's dtype (ctest)

The header is parsed with a deliberately small grammar (obs/observation.h keeps to
it): `typedef struct Name { <type> <name>[<N>]; ... } Name;` with fixed-width
integer types or previously defined struct typedefs, inside #pragma pack(push, 1);
`enum Name { PREFIX_A = 0, PREFIX_B, ... };`; `#define PN_... <int>`.
Field names ending in "_priv" (or under such a field) are privileged.
"""
import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "obs" / "observation.h"
CHARMAP = ROOT / "game" / "charmap.txt"
OUT = ROOT / "python" / "fireredsim" / "obs_dtype.py"

PRIM = {
    "uint8_t": ("u1", 1), "int8_t": ("i1", 1),
    "uint16_t": ("<u2", 2), "int16_t": ("<i2", 2),
    "uint32_t": ("<u4", 4), "int32_t": ("<i4", 4),
    "uint64_t": ("<u8", 8), "int64_t": ("<i8", 8),
}

# GetExtCtrlCodeLength (game/src/string_util.c): length of an 0xFC control code
# including the code byte, indexed by the code
EXT_CTRL_LENGTHS = [1, 2, 2, 2, 4, 2, 2, 1, 2, 1, 1, 3, 2, 2, 2, 1, 3, 2, 2, 2, 2, 1, 1, 1, 1]


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def parse_header(path=HEADER):
    raw = path.read_text()
    text = strip_comments(raw)
    structs = {}  # name -> list of (field, type, count)
    order = []
    for m in re.finditer(r"typedef\s+struct\s+(\w+)\s*\{(.*?)\}\s*(\w+)\s*;", text, flags=re.S):
        name, body, tname = m.group(1), m.group(2), m.group(3)
        if name != tname:
            raise SystemExit(f"struct {name}: typedef name {tname} differs")
        fields = []
        for decl in body.split(";"):
            decl = " ".join(decl.split())
            if not decl:
                continue
            fm = re.fullmatch(r"(\w+)\s+(\w+)(?:\[(\d+)\])?", decl)
            if not fm:
                raise SystemExit(f"struct {name}: cannot parse field '{decl}'")
            ftype, fname, count = fm.group(1), fm.group(2), int(fm.group(3) or 0)
            if ftype not in PRIM and ftype not in structs:
                raise SystemExit(f"struct {name}: unknown type {ftype}")
            fields.append((fname, ftype, count))
        structs[name] = fields
        order.append(name)
    enums = {}
    for m in re.finditer(r"enum\s+(\w+)\s*\{(.*?)\}\s*;", text, flags=re.S):
        items, value = [], -1
        for item in m.group(2).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                k, v = (s.strip() for s in item.split("=", 1))
                value = int(v, 0)
            else:
                k = item
                value += 1
            items.append((k, value))
        enums[m.group(1)] = items
    defines = {}
    for m in re.finditer(r"#define\s+(PN_\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", text):
        defines[m.group(1)] = int(m.group(2), 0)
    asserts = {m.group(1): int(m.group(2)) for m in
               re.finditer(r"PN_OBS_STATIC_ASSERT\(sizeof\((\w+)\)\s*==\s*(\d+)", text)}
    return structs, order, enums, defines, asserts


def layout(structs):
    """size of every struct (packed)"""
    sizes = {}

    def size_of(t):
        if t in PRIM:
            return PRIM[t][1]
        if t not in sizes:
            sizes[t] = sum(size_of(ft) * (c or 1) for _, ft, c in structs[t])
        return sizes[t]

    for name in structs:
        size_of(name)
    return sizes


def flat_fields(structs, sizes, tname, prefix="", base=0, priv=False):
    """(path, offset, size, privileged) for every primitive field or primitive array"""
    out = []
    off = base
    for fname, ftype, count in structs[tname]:
        p = priv or fname.endswith("_priv")
        if ftype in PRIM:
            n = PRIM[ftype][1] * (count or 1)
            out.append((prefix + fname, off, n, p))
            off += n
        else:
            for i in range(count or 1):
                sub = f"{prefix}{fname}[{i}]." if count else f"{prefix}{fname}."
                out += flat_fields(structs, sizes, ftype, sub, off, p)
                off += sizes[ftype]
    return out


# ---- python module ------------------------------------------------------------------

def dtype_src(structs, order, sizes, name):
    names, formats, offsets = [], [], []
    off = 0
    for fname, ftype, count in structs[name]:
        names.append(fname)
        base = PRIM[ftype][0] if ftype in PRIM else f"_DT_{ftype}"
        formats.append(f"({base}, ({count},))" if count else base)
        offsets.append(off)
        off += (PRIM[ftype][1] if ftype in PRIM else sizes[ftype]) * (count or 1)
    return names, formats, offsets


def py_format(f):
    # f is a numpy base string, "_DT_X", or "(base, (n,))"
    if f.startswith("("):
        base, shape = f[1:-1].split(", ", 1)
        base = base if base.startswith("_DT_") else repr(base)
        return f"({base}, {shape})"
    return f if f.startswith("_DT_") else repr(f)


def load_charmap():
    table = {}
    for line in CHARMAP.read_text(encoding="utf-8").splitlines():
        line = line.split("@", 1)[0].rstrip()
        m = re.fullmatch(r"\s*'(.+)'\s*=\s*([0-9A-Fa-f]{2})", line)
        if m:
            ch, b = m.group(1), int(m.group(2), 16)
            if ch.startswith("\\") and len(ch) > 1:
                continue
            table.setdefault(b, ch)
            continue
        m = re.fullmatch(r"\s*([A-Z_0-9]+)\s*=\s*([0-9A-Fa-f]{2})", line)
        if m:
            b = int(m.group(2), 16)
            if b < 0xF7:
                table.setdefault(b, "{" + m.group(1) + "}")
    return table


def render_module(structs, order, sizes, enums, defines):
    lines = [
        '"""GENERATED by tools/gen_obs_dtype.py from obs/observation.h and game/charmap.txt.',
        "",
        "Do not edit: change the header and rebuild (the build regenerates this file and",
        "ctest obs_dtype_matches_c checks it against the C compiler's layout).",
        "",
        "    import numpy as np",
        "    from fireredsim.obs_dtype import OBS_DTYPE, decode_text",
        "    obs = np.frombuffer(buf, dtype=OBS_DTYPE)[0]",
        "    obs['screen'], decode_text(obs['dialog_text'])",
        '"""',
        "import numpy as np",
        "",
        f"OBS_MAGIC = 0x{defines['PN_OBS_MAGIC']:08X}",
        f"OBS_VERSION = {defines['PN_OBS_VERSION']}",
        f"OBS_SIZE = {sizes['PnObs']}",
        "OBS_VISIBLE = 0",
        "OBS_PRIVILEGED = 1",
        "",
    ]
    for name in order:
        names, formats, offsets = dtype_src(structs, order, sizes, name)
        lines.append(f"_DT_{name} = np.dtype({{")
        lines.append(f"    'names': {names!r},")
        lines.append("    'formats': [" + ", ".join(py_format(f) for f in formats) + "],")
        lines.append(f"    'offsets': {offsets!r},")
        lines.append(f"    'itemsize': {sizes[name]},")
        lines.append("})")
        lines.append(f"{name.upper()[2:]}_DTYPE = _DT_{name}" if name.startswith("Pn") else f"{name}_DTYPE = _DT_{name}")
        lines.append("")
    lines.append("OBS_DTYPE = _DT_PnObs")
    lines.append("assert OBS_DTYPE.itemsize == OBS_SIZE")
    lines.append("")
    # enums
    for ename, items in enums.items():
        prefix = None
        for k, _ in items:
            parts = k.split("_")
            prefix = parts[:2] if prefix is None else prefix
        lines.append(f"# enum {ename}")
        for k, v in items:
            lines.append(f"{k[3:]} = {v}")
        lines.append("")

    def names_list(ename, strip):
        vals = {v: k[len(strip):] for k, v in enums[ename] if not k.endswith("_COUNT")}
        n = max(vals) + 1
        return [vals.get(i, "?") for i in range(n)]

    lines.append(f"SCREEN_NAMES = {names_list('PnScreen', 'PN_SCREEN_')!r}")
    lines.append(f"INPUT_KIND_NAMES = {names_list('PnInputKind', 'PN_INPUT_')!r}")
    lines.append(f"WAIT_CTX_NAMES = {names_list('PnWaitCtx', 'PN_WAIT_')!r}")
    lines.append("")
    keys = {k[len('PN_KEY_'):]: v for k, v in defines.items() if k.startswith("PN_KEY_") and k != "PN_KEY_DPAD"}
    lines.append(f"KEY_BITS = {keys!r}")
    flags = {k[len('PN_INPUT_FLAG_'):]: v for k, v in defines.items() if k.startswith("PN_INPUT_FLAG_")}
    lines.append(f"INPUT_FLAG_BITS = {flags!r}")
    valid = {k[len('PN_VALID_'):]: v for k, v in defines.items() if k.startswith("PN_VALID_")}
    lines.append(f"VALID_BITS = {valid!r}")
    if "PN_SKILL_SUCCESS_SHIFT" in defines:  # v5: skill_events_priv bit of the "succeeded" variant of event q
        lines.append(f"SKILL_SUCCESS_SHIFT = {defines['PN_SKILL_SUCCESS_SHIFT']}")
    lines.append("")
    lines.append("")
    lines.append("def keys_to_names(mask):")
    lines.append('    """GBA key mask -> list of key names"""')
    lines.append("    return [k for k, v in KEY_BITS.items() if mask & v]")
    lines.append("")
    top_priv = [f for f, _, _ in structs["PnObs"] if f.endswith("_priv")]
    lines.append("# top-level fields that are entirely privileged (zero in visible mode); nested")
    lines.append("# *_priv members of party/battlers/objects are covered by privileged_mask()")
    lines.append(f"PRIVILEGED_FIELDS = {top_priv!r}")
    lines.append("")
    lines.append("")
    lines.append("def privileged_mask():")
    lines.append('    """uint8 array of OBS_SIZE bytes: 1 where the byte belongs to a privileged field"""')
    lines.append("    m = np.zeros(OBS_SIZE, dtype=np.uint8)")
    lines.append("    for off, size in _PRIV_RANGES:")
    lines.append("        m[off:off + size] = 1")
    lines.append("    return m")
    lines.append("")
    ranges = []
    for p, off, size, pv in flat_fields(structs, sizes, "PnObs"):
        if pv:
            if ranges and ranges[-1][0] + ranges[-1][1] == off:
                ranges[-1][1] += size
            else:
                ranges.append([off, size])
    lines.append(f"_PRIV_RANGES = {[tuple(r) for r in ranges]!r}")
    lines.append("")
    # charmap
    cm = load_charmap()
    lines.append("# game/charmap.txt (single-byte entries; Latin before Japanese)")
    lines.append("CHARMAP = {")
    for b in sorted(cm):
        lines.append(f"    0x{b:02X}: {cm[b]!r},")
    lines.append("}")
    lines.append(f"_EXT_CTRL_LENGTHS = {EXT_CTRL_LENGTHS!r}")
    lines.append("")
    lines.append(DECODER)
    return "\n".join(lines) + "\n"


DECODER = '''
def decode_text(data, control_codes=False):
    """Decodes game-encoded bytes (dialog_text, nickname, player_name, ...) to str.
    0xFF ends the string; 0xFE is a newline, 0xFA (scroll) and 0xFB (new page) become
    newlines too; 0xFD xx placeholders and 0xFC control codes are dropped unless
    control_codes=True (then rendered as {PLACEHOLDER_xx} / {CTRL_xx})."""
    data = bytes(np.asarray(data, dtype=np.uint8).tobytes()) if not isinstance(data, (bytes, bytearray)) else bytes(data)
    out = []
    i = 0
    while i < len(data):
        c = data[i]
        if c == 0xFF:
            break
        if c == 0xFE:
            out.append("\\n")
        elif c == 0xFA:
            out.append("\\n")
        elif c == 0xFB:
            out.append("\\n\\n")
        elif c == 0xFD:
            if control_codes and i + 1 < len(data):
                out.append("{PLACEHOLDER_%02X}" % data[i + 1])
            i += 2
            continue
        elif c == 0xFC:
            code = data[i + 1] if i + 1 < len(data) else 0
            n = _EXT_CTRL_LENGTHS[code] if code < len(_EXT_CTRL_LENGTHS) else 1
            if control_codes:
                out.append("{CTRL_%02X}" % code)
            i += 1 + max(n, 1)
            continue
        elif c in (0xF8, 0xF9):
            i += 2
            continue
        else:
            out.append(CHARMAP.get(c, "?"))
        i += 1
    return "".join(out)
'''


# ---- C layout program -----------------------------------------------------------------

def c_member(path):
    # "party[0].species" -> member designator usable in offsetof (GCC extension)
    return path


def emit_fields_h(structs, sizes):
    flat = flat_fields(structs, sizes, "PnObs")
    lines = [
        "/* GENERATED by tools/gen_obs_dtype.py from obs/observation.h. Do not edit. */",
        "#pragma once",
        "#include <stddef.h>",
        "#include <stdint.h>",
        '#include "observation.h"',
        "typedef struct PnObsFieldInfo { const char *name; uint32_t offset; uint32_t size; uint8_t priv; } PnObsFieldInfo;",
        "static const PnObsFieldInfo pn_obs_fields[] = {",
    ]
    for p, off, size, pv in flat:
        lines.append(f'    {{"{p}", (uint32_t)offsetof(PnObs, {c_member(p)}), (uint32_t)sizeof(((PnObs *)0)->{p}), {int(pv)}}},')
    lines.append("};")
    lines.append(f"#define PN_OBS_NUM_FIELDS {len(flat)}")
    return "\n".join(lines) + "\n"


def emit_c(structs, sizes):
    lines = [
        "/* GENERATED by tools/gen_obs_dtype.py: prints the C layout of PnObs. */",
        "#include <stdio.h>",
        '#include "pn_obs_fields.h"',
        "int main(void)",
        "{",
        "    int i;",
    ]
    for name in structs:
        lines.append(f'    printf("sizeof {name} %u\\n", (unsigned)sizeof({name}));')
    lines.append("    for (i = 0; i < PN_OBS_NUM_FIELDS; i++)")
    lines.append('        printf("field %s %u %u %u\\n", pn_obs_fields[i].name, pn_obs_fields[i].offset, pn_obs_fields[i].size, pn_obs_fields[i].priv);')
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def check_layout(binary, structs, sizes):
    out = subprocess.run([binary], check=True, capture_output=True, text=True).stdout
    errors = []
    c_sizes, c_fields = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "sizeof":
            c_sizes[parts[1]] = int(parts[2])
        elif parts[0] == "field":
            c_fields[parts[1]] = (int(parts[2]), int(parts[3]), int(parts[4]))
    for name, size in sizes.items():
        if c_sizes.get(name) != size:
            errors.append(f"sizeof {name}: C {c_sizes.get(name)} vs parse {size}")
    flat = flat_fields(structs, sizes, "PnObs")
    for p, off, size, pv in flat:
        if c_fields.get(p) != (off, size, int(pv)):
            errors.append(f"field {p}: C {c_fields.get(p)} vs parse {(off, size, int(pv))}")
    # the generated module
    spec = importlib.util.spec_from_file_location("obs_dtype", OUT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dt = mod.OBS_DTYPE
    if dt.itemsize != c_sizes.get("PnObs"):
        errors.append(f"OBS_DTYPE.itemsize {dt.itemsize} != sizeof(PnObs) {c_sizes.get('PnObs')}")

    def walk(dtype, prefix, base):
        for fname in dtype.names:
            sub, off = dtype.fields[fname][:2]
            path = prefix + fname
            if sub.subdtype is not None:
                inner, shape = sub.subdtype
                if inner.names:
                    for i in range(shape[0]):
                        yield from walk(inner, f"{path}[{i}].", base + off + i * inner.itemsize)
                    continue
                yield path, base + off, sub.itemsize
            elif sub.names:
                yield from walk(sub, path + ".", base + off)
            else:
                yield path, base + off, sub.itemsize

    np_fields = {p: (o, s) for p, o, s in walk(dt, "", 0)}
    for p, (off, size, _) in c_fields.items():
        if np_fields.get(p) != (off, size):
            errors.append(f"numpy field {p}: {np_fields.get(p)} vs C {(off, size)}")
    if set(np_fields) != set(c_fields):
        errors.append(f"field sets differ: {sorted(set(np_fields) ^ set(c_fields))[:10]}")
    priv = mod.privileged_mask()
    for p, (off, size, pv) in c_fields.items():
        if any(priv[off:off + size] != pv):
            errors.append(f"privileged mask wrong for {p}")
    if errors:
        print("\n".join(errors[:50]))
        raise SystemExit(f"obs dtype check FAILED: {len(errors)} errors")
    print(f"obs dtype check OK: sizeof(PnObs) = {c_sizes['PnObs']} = OBS_DTYPE.itemsize, "
          f"{len(c_fields)} fields agree (offset, size, privileged), version {mod.OBS_VERSION}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit-c")
    ap.add_argument("--emit-fields-h")
    ap.add_argument("--check-layout")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    structs, order, enums, defines, asserts = parse_header()
    sizes = layout(structs)
    for name, size in asserts.items():
        if sizes.get(name) != size:
            raise SystemExit(f"header static_assert sizeof({name}) == {size} but the fields add up to {sizes.get(name)}")
    if a.emit_c:
        Path(a.emit_c).write_text(emit_c(structs, sizes))
        Path(a.emit_c).with_name("pn_obs_fields.h").write_text(emit_fields_h(structs, sizes))
        return
    if a.emit_fields_h:
        Path(a.emit_fields_h).write_text(emit_fields_h(structs, sizes))
        return
    if a.check_layout:
        check_layout(a.check_layout, structs, sizes)
        return
    text = render_module(structs, order, sizes, enums, defines)
    out = Path(a.out)
    if a.check:
        if not out.exists() or out.read_text() != text:
            raise SystemExit(f"{out} is out of date: run tools/gen_obs_dtype.py")
        print(f"{out} up to date")
        return
    if not out.exists() or out.read_text() != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out} (OBS_SIZE {sizes['PnObs']})")


if __name__ == "__main__":
    main()
