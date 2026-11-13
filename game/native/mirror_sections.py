#!/usr/bin/env python3
"""fireredsim: prepare an object's writable sections for the mirror memory layout.

usage: mirror_sections.py {game|platform} IN.o OUT.o

The native executable puts every game RAM variable at the address it has in the
retail ROM (docs/MIRROR_LAYOUT.md): EWRAM at 0x02000000, IWRAM at 0x03000000.
Objects are compiled with -fdata-sections and -DPN_MIRROR_SECTIONS (each
EWRAM_DATA/IWRAM_DATA/COMMON_DATA variable gets a section of its own, see
include/gba/defines.h), so every RAM variable is alone in its input section.
This script gives those sections globally unique names that tools/mirror/gen_ld.py
places one by one:

game objects (game/src/*.c, game/data/*.s, songs):
  ewram_data[.N]                     -> .mirror.ewram.<obj>.<var>   (EWRAM on the GBA)
  .bss[.*], common_data[.N],
  iwram_data[.N], .bss.code          -> .mirror.iwram.<obj>.<var>   (IWRAM on the GBA)
  .data[.*]                          -> .rodata.romdata[.*]         read-only: the ROM
                                         linker script places .data in ROM, so these
                                         are constants on the GBA as well
platform objects (platform/*.c, hardware replacements):
  ewram_data[.N] / iwram_data[.N]    -> .mirror.{ewram,iwram}.<obj>.<var>  (platform
                                         code that ports ROM RAM variables: gHeap,
                                         the m4a_1.s decoding buffer)
  .host.*                            -> unchanged (HOST_STATE: host-only bookkeeping)
  .pnstate.stack                     -> unchanged (the game stack, platform/frame.c)
  every other writable               -> .pnstate.<obj>.<var>        (native-only state;
                                         must be zero-initialised: the section is NOLOAD)

<obj> is the object's path below the object directory (game/Makefile.native OBJ_DIR,
exported to this script as $PN_OBJ_ROOT; default build/native) with '/' -> '_' and
without '.o'
(src_main, data_sound_data, platform_hw); <var> is the symbol at the lowest
offset of the section (the only one for C objects). Empty sections are left
alone. Any other writable allocated section is an error, as is a COMMON symbol.

Objects built with -fprofile-generate (PGO, tools/pgo_build.sh) also hold the
profiler's own data: .data..LPBX*, .data.__gcov_*, .bss.__gcov*.* and the
constructor's .init_array and destructor's .fini_array. Those are host state (libgcov writes the .gcda files
at exit) and are left unchanged, i.e. ordinary .data/.bss outside the game's
memory and outside the snapshotted state.

Both kinds of objects also get their references to the C library's memory and
string functions redirected to the platform's versions (memcpy -> Port_memcpy,
see platform/libc_port.c): game code runs on a stack in .pnstate and must not
execute host C library code during a frame (determinism), and the retail ROM's
newlib semantics apply. An object that still references one of the functions in
FORBIDDEN after renaming is an error.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'mirror'))
import elf32  # noqa: E402
from layout import object_tag  # noqa: E402


# Host C library functions redirected to platform/libc_port.c.
REDIRECT = ['memcpy', 'memmove', 'memset', 'memcmp', 'strcmp', 'strcpy', 'strlen']

# Host functions that must never be called from game or platform code (they
# would run on the game stack, use host state or be nondeterministic). The
# platform's fatal-error path (Port_Fatal: vsnprintf, fprintf, abort) is
# allowed: the process or the environment ends there.
FORBIDDEN = set(REDIRECT) | {
    'strncpy', 'strncmp', 'strcat', 'strncat', 'strchr', 'strrchr', 'memchr', 'strdup',
    '__memcpy_chk', '__memmove_chk', '__memset_chk', '__strcpy_chk', '__strcat_chk',
    'setjmp', '_setjmp', '__sigsetjmp', 'longjmp', '_longjmp', 'siglongjmp', '__longjmp_chk',
    '__stack_chk_fail', '__stack_chk_fail_local',
    'malloc', 'calloc', 'realloc', 'free',
    'rand', 'srand', 'random', 'srandom', 'rand_r',
    'time', 'clock', 'gettimeofday', 'clock_gettime', 'getenv',
}


def ram_kind(name, mode):
    """'ewram'/'iwram' for a section that holds GBA RAM variables, else None."""
    if name == 'ewram_data' or name.startswith('ewram_data.'):
        return 'ewram'
    if name == 'iwram_data' or name.startswith('iwram_data.'):
        return 'iwram'
    if mode == 'game' and (name == '.bss' or name.startswith('.bss.') or name == 'common_data'
                           or name.startswith('common_data.')):
        return 'iwram'
    return None


def is_profile_section(name):
    """Data of gcc's -fprofile-generate instrumentation (never game or platform state)."""
    return name.startswith(('.data..LPBX', '.data.__gcov_', '.bss.__gcov'))


def section_var(sec, by_sec):
    syms = [y for y in by_sec.get(sec.index, []) if y.type in (elf32.STT_OBJECT, elf32.STT_NOTYPE)]
    if not syms:
        return 'anon' + re.sub(r'[^A-Za-z0-9_.]', '_', sec.name)
    return re.sub(r'[^A-Za-z0-9_.]', '_', syms[0].name)


def nonzero_contents(path, name):
    out = subprocess.run(['objcopy', '-O', 'binary', '--only-section=' + name, path, '/dev/stdout'],
                         capture_output=True, check=True).stdout
    return any(out)


def undefined_symbols(path):
    out = subprocess.run(['nm', '-u', path], capture_output=True, text=True, check=True).stdout
    return {l.split()[-1] for l in out.splitlines() if l.strip()}


def err(src, msg):
    sys.stderr.write('mirror_sections.py: %s: %s\n' % (src, msg))
    return 1


def process(mode, src, dst):
    tag = object_tag(os.path.abspath(dst))
    secs, syms = elf32.read(src)
    by_sec = elf32.symbols_by_section(secs, syms)
    commons = [y.name for y in syms if y.shndx == elf32.SHN_COMMON]
    if commons:
        return err(src, 'COMMON symbols (build with -fno-common): ' + ' '.join(commons))
    args = []
    for name in REDIRECT:
        args += ['--redefine-sym', '%s=Port_%s' % (name, name)]
    used = set()
    profiled = any(is_profile_section(s.name) for s in secs)
    for s in secs:
        if not s.writable_alloc:
            continue
        name = s.name
        if profiled and (is_profile_section(name) or name.startswith(('.init_array', '.fini_array'))):
            continue
        if mode == 'platform' and (name.startswith('.host') or name == '.pnstate.stack'):
            continue
        if mode == 'game' and (name == '.data' or name.startswith('.data.')):
            if s.size:
                args += ['--rename-section',
                         '%s=.rodata.romdata%s,alloc,load,readonly,data,contents' % (name, name[5:])]
            continue
        if s.size == 0:
            continue
        kind = ram_kind(name, mode)
        if kind:
            new = '.mirror.%s.%s.%s' % (kind, tag, section_var(s, by_sec))
        elif mode == 'platform':
            new = '.pnstate.%s.%s' % (tag, section_var(s, by_sec))
        else:
            return err(src, 'unexpected writable section %s' % name)
        if new in used:
            return err(src, 'two writable sections would both be named %s' % new)
        used.add(new)
        if s.type != elf32.SHT_NOBITS and nonzero_contents(src, name):
            if mode == 'platform':
                return err(src, 'platform section %s has non-zero initial data; RAM is NOLOAD '
                                '(initialise state in Port_PowerOn instead)' % name)
            # Game RAM initialisers are dropped on the GBA too (ewram/iwram are NOLOAD
            # in the ROM linker script and RegisterRamReset clears them).
        args += ['--rename-section', '%s=%s' % (name, new)]
    tmp = dst + '.tmp'
    subprocess.run(['objcopy'] + args + [src, tmp], check=True)
    bad = sorted(undefined_symbols(tmp) & FORBIDDEN)
    if bad:
        os.unlink(tmp)
        return err(src, 'calls host C library functions that must not run inside a frame: %s '
                        '(use the Port_* versions in platform/libc_port.c)' % ' '.join(bad))
    os.replace(tmp, dst)
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ('game', 'platform'):
        sys.stderr.write(__doc__)
        sys.exit(2)
    sys.exit(process(sys.argv[1], sys.argv[2], sys.argv[3]))
