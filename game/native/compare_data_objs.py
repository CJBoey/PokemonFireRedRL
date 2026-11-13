#!/usr/bin/env python3
"""fireredsim: verify the ARM->i386 conversion of the data assembly.

For every data object of the native build (data/*.s, the generated songs) this
assembles the same source with pret's ARM pipeline (arm-none-eabi-as) and
compares, section by section:
  * the raw section bytes (both targets use REL relocations, so the addends are
    stored in place and the bytes must be identical),
  * the relocations (offset, size, symbol),
  * the defined symbols (name, section, value).
Section names differ only where as_x86.py renames them on purpose
(script_data -> .rodata.script_data) and where mirror_sections.py renamed the
writable section for the mirror layout (.bss -> .mirror.iwram.<object>.<first variable>).

usage (from game/):  python3 native/compare_data_objs.py [-j N] [files...]
"""
import concurrent.futures
import glob
import os
import re
import subprocess
import sys
import tempfile

ARM_AS = 'arm-none-eabi-as'
ASFLAGS_ARM = ['-mcpu=arm7tdmi', '--defsym', 'FIRERED=1', '--defsym', 'REVISION=0', '--defsym', 'ENGLISH=1',
               '--defsym', 'MODERN=1']
PREPROC = 'tools/preproc/preproc'
SECTION_MAP = {'script_data': '.rodata.script_data'}


def native_section_name(name, n_secs):
    """Name of the native counterpart of ARM section `name`."""
    if name == '.bss':
        mirror = [n for n in n_secs if n.startswith('.mirror.iwram.')]
        if len(mirror) == 1:
            return mirror[0]
    return SECTION_MAP.get(name, name)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, **kw).stdout


def read_sections(obj, readelf):
    out = sh('%s -SW %s' % (readelf, obj)).decode()
    secs = {}
    for line in out.splitlines():
        m = re.match(r'^\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)', line)
        if m:
            secs[m.group(2)] = (int(m.group(1)), m.group(3), int(m.group(4), 16), int(m.group(5), 16))
    return secs


def section_bytes(obj, name, typ, off, size):
    if typ == 'NOBITS':
        return b'\0' * size
    with open(obj, 'rb') as f:
        f.seek(off)
        return f.read(size)


def relocs(obj, readelf):
    out = sh('%s -rW %s' % (readelf, obj)).decode()
    res = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"^Relocation section '\.rel(\S+)'", line)
        if m:
            cur = m.group(1)
            res[cur] = []
            continue
        m = re.match(r'^([0-9a-f]+)\s+[0-9a-f]+\s+(\S+)\s+[0-9a-f]+\s+(\S+)', line)
        if m and cur is not None:
            typ = m.group(2)
            size = {'R_ARM_ABS32': 4, 'R_386_32': 4, 'R_ARM_ABS16': 2, 'R_386_16': 2,
                    'R_ARM_ABS8': 1, 'R_386_8': 1}.get(typ, typ)
            res[cur].append((int(m.group(1), 16), size, m.group(3)))
    return res


def symbols(obj, nm):
    out = sh('%s %s' % (nm, obj)).decode()
    res = set()
    for line in out.splitlines():
        p = line.split()
        if len(p) == 3 and p[1].upper() in 'DRBT' and not p[2].startswith('$'):
            res.add((p[2], int(p[0], 16)))
    return res


def arm_assemble(src, out):
    if src.startswith('data/'):
        cmd = ('%s %s charmap.txt | arm-none-eabi-cpp -I include - | %s -ie %s charmap.txt | %s %s -o %s'
               % (PREPROC, src, PREPROC, src, ARM_AS, ' '.join(ASFLAGS_ARM), out))
        if src in ('data/maps.s', 'data/map_events.s'):
            cmd = cmd.replace('arm-none-eabi-cpp -I include -', 'arm-none-eabi-cpp -I include -nostdinc -undef -Wno-unicode -')
    else:
        cmd = '%s %s -I sound -o %s %s' % (ARM_AS, ' '.join(ASFLAGS_ARM), out, src)
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def compare(src, native_obj):
    with tempfile.TemporaryDirectory() as td:
        arm_obj = os.path.join(td, 'arm.o')
        try:
            arm_assemble(src, arm_obj)
        except subprocess.CalledProcessError as e:
            return src, ['ARM assembly failed: ' + e.stderr.decode()[-400:]]
        problems = []
        a_secs = read_sections(arm_obj, 'arm-none-eabi-readelf')
        n_secs = read_sections(native_obj, 'readelf')
        a_rel = relocs(arm_obj, 'arm-none-eabi-readelf')
        n_rel = relocs(native_obj, 'readelf')
        for name, (idx, typ, off, size) in a_secs.items():
            if name in ('', '.text', '.data', '.ARM.attributes', '.symtab', '.strtab', '.shstrtab') or name.startswith('.rel'):
                if name in ('.text', '.data') and size != 0:
                    problems.append('ARM object has non-empty %s' % name)
                continue
            nname = native_section_name(name, n_secs)
            if nname not in n_secs:
                problems.append('section %s missing in native object' % name)
                continue
            _, ntyp, noff, nsize = n_secs[nname]
            if nsize != size:
                problems.append('section %s size %d (ARM) vs %d (native)' % (name, size, nsize))
                continue
            ab = section_bytes(arm_obj, name, typ, off, size)
            nb = section_bytes(native_obj, nname, ntyp, noff, nsize)
            if ab != nb:
                first = next(i for i in range(len(ab)) if ab[i] != nb[i])
                problems.append('section %s bytes differ (first at 0x%x)' % (name, first))
            ar = sorted((o, sz, native_section_name(sym, n_secs)) for o, sz, sym in a_rel.get(name, []))
            nr = sorted(n_rel.get(nname, []))
            if ar != nr:
                problems.append('section %s relocations differ (%d ARM vs %d native)' % (name, len(ar), len(nr)))
        for name, (idx, typ, off, size) in n_secs.items():
            if size and name not in ('.text', '.data', '.bss', '.note.GNU-stack', '.symtab', '.strtab', '.shstrtab') \
                    and not name.startswith('.rel') and name not in SECTION_MAP.values() and name not in a_secs \
                    and not name.startswith('.mirror.iwram.'):
                problems.append('native object has extra section %s' % name)
        a_sym = symbols(arm_obj, 'arm-none-eabi-nm')
        n_sym = symbols(native_obj, 'nm')
        if a_sym != n_sym:
            only_a = sorted(a_sym - n_sym)[:5]
            only_n = sorted(n_sym - a_sym)[:5]
            problems.append('symbols differ: only ARM %s only native %s' % (only_a, only_n))
        return src, problems


def main():
    args = sys.argv[1:]
    jobs = os.cpu_count() or 4
    if args[:1] == ['-j']:
        jobs = int(args[1])
        args = args[2:]
    if args:
        srcs = args
    else:
        srcs = sorted(glob.glob('data/*.s')) + sorted(glob.glob('sound/songs/midi/*.s'))
    pairs = []
    for s in srcs:
        obj = os.path.join(os.environ.get('PN_OBJ_ROOT', 'build/native'), os.path.splitext(s)[0] + '.o')
        if not os.path.exists(obj):
            print('missing native object for %s' % s)
            continue
        pairs.append((s, obj))
    bad = 0
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        for src, problems in ex.map(lambda p: compare(*p), pairs):
            if problems:
                bad += 1
                print('MISMATCH %s' % src)
                for p in problems:
                    print('    ' + p)
    print('%d files compared, %d mismatching' % (len(pairs), bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
