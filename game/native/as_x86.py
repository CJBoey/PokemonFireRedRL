#!/usr/bin/env python3
"""fireredsim: assemble pret's ARM-flavoured GAS data sources for i386 ELF.

Used as $(AS) by game/Makefile.native. Accepts the same command line the pret
rules give to arm-none-eabi-as (``[flags] [-I dir] -o out.o [file.s]``, source on
stdin when no file is given), rewrites the ARM GAS dialect into i386 GAS, runs
``as --32`` and finally moves every writable section of the object into the
mirror layout (see game/native/mirror_sections.py).

Only data is assembled this way (data/*.s, sound/songs/midi/*.s); the ARM code
files (crt0.s, libagbsyscall.s, m4a_1.s, ...) are replaced by C in platform/.

Dialect differences handled (each verified by native/compare_data_objs.py, which
assembles every data file with both toolchains and compares section bytes and
relocations):
  * '@' starts a comment on ARM (but '\\@' is the macro counter);
  * '.align N' is a power of two on ARM, a byte count on i386 -> '.p2align N';
  * '.word' is 32-bit on ARM, 16-bit on i386 -> '.4byte'; '.hword' -> '.2byte';
  * '%progbits' -> '@progbits' (etc.);
  * section 'script_data' ("aw" in ROM) -> read-only '.rodata.script_data';
  * i386 GAS splits macro arguments such as '(a + b) - c' or '- 1' into two
    arguments; every macro argument that is an expression is wrapped in parens;
  * '.include' is expanded textually (needed for files assembled directly).
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    sys.stderr.write("as_x86.py: " + msg + "\n")
    sys.exit(1)


def strip_comment(line):
    """Remove an ARM '@' comment (outside strings, not '\\@')."""
    if '@' not in line:
        return line
    out = []
    in_str = False
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if in_str:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
                out.append(c)
            elif c == '\\' and i + 1 < n and line[i + 1] == '@':
                out.append('\\@')
                i += 2
                continue
            elif c == '@':
                break
            else:
                out.append(c)
        i += 1
    return ''.join(out)


def split_top_level(text):
    """Split macro arguments on commas at paren depth 0, outside strings."""
    out, cur, depth, in_str = [], [], 0, False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            cur.append(c)
            if c == '\\' and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            cur.append(c)
        elif c == '(':
            depth += 1
            cur.append(c)
        elif c == ')':
            depth -= 1
            cur.append(c)
        elif c == ',' and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append(''.join(cur))
    return out


_SIMPLE_ARG = re.compile(r'^(\\?[A-Za-z_.$][A-Za-z0-9_.$]*|\\[A-Za-z_][A-Za-z0-9_]*(\\\(\))?|[0-9][0-9A-Za-z_]*|0[xX][0-9A-Fa-f]+)$')


_OPERAND_END = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.$)"')
_OPERAND_START = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.$("\\')


def split_blank_separated(arg):
    """ARM GAS also separates macro arguments by blanks: 'setbyte (x + 1) \\v | 2'
    passes '(x + 1)' and '\\v | 2'. A blank at paren depth 0 that sits between the
    end of one operand and the start of another separates two arguments; blanks
    next to an operator are part of the expression."""
    parts, cur, depth, in_str = [], [], 0, False
    i, n = 0, len(arg)
    while i < n:
        c = arg[i]
        if in_str:
            cur.append(c)
            if c == '\\' and i + 1 < n:
                cur.append(arg[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0 and c in ' \t':
            j = i
            while j < n and arg[j] in ' \t':
                j += 1
            prev = ''.join(cur).rstrip()
            if prev and j < n and prev[-1] in _OPERAND_END and arg[j] in _OPERAND_START:
                parts.append(''.join(cur))
                cur = []
                i = j
                continue
        cur.append(c)
        i += 1
    parts.append(''.join(cur))
    return parts


def fully_parenthesized(s):
    if not (s.startswith('(') and s.endswith(')')):
        return False
    depth = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return True


def wrap_arg(a):
    s = a.strip()
    if not s:
        return a
    # keyword argument name=value
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.*)$', s)
    if m:
        return m.group(1) + '=' + wrap_arg(m.group(2)).strip()
    if s.startswith('"') or _SIMPLE_ARG.match(s) or fully_parenthesized(s):
        return s
    if '\\' in s and not re.search(r'[\s+\-*/%<>&|^~!()]', s):
        return s  # token pasting such as GROUP_\map_name
    return '(' + s + ')'


_RE_MACRO_DEF = re.compile(r'^\s*\.macro\s+([A-Za-z_.$][A-Za-z0-9_.$]*)')
_RE_CALL = re.compile(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s+)(.*\S)\s*$')
_RE_LABELS = re.compile(r'^(\s*(?:[A-Za-z_.$][A-Za-z0-9_.$]*:+\s*)+)(.*)$')


def convert_statement(stmt, macros):
    s = stmt
    # directives
    m = re.match(r'^(\s*)\.(\w+)\b(.*)$', s)
    if m:
        indent, d, rest = m.groups()
        dl = d.lower()
        if dl == 'align':
            return indent + '.p2align' + rest
        if dl == 'word':
            return indent + '.4byte' + rest
        if dl == 'hword':
            return indent + '.2byte' + rest
        if dl in ('thumb', 'arm', 'thumb_func', 'force_thumb', 'syntax', 'code', 'cpu', 'fpu', 'arch', 'pool', 'ltorg'):
            return ''
        if dl == 'section':
            r = rest.replace('%progbits', '@progbits').replace('%nobits', '@nobits')
            name = r.strip().split(',')[0].strip()
            if name == 'script_data':
                return indent + '.section .rodata.script_data, "a", @progbits'
            return indent + '.section' + r
        if dl == 'type':
            return indent + '.type' + rest.replace('%function', '@function').replace('%object', '@object')
        return s
    # macro invocation (possibly preceded by labels)
    lm = _RE_LABELS.match(s)
    prefix = ''
    body = s
    if lm and lm.group(2):
        prefix, body = lm.group(1), lm.group(2)
    cm = _RE_CALL.match(body)
    if cm and cm.group(2) in macros:
        indent, name, sep, rest = cm.groups()
        args = []
        for a in split_top_level(rest):
            args += split_blank_separated(a.strip())
        new = [wrap_arg(a) for a in args]
        return prefix + indent + name + sep + ', '.join(x.strip() for x in new)
    return s


def split_statements(line):
    """Split on ';' outside strings (GAS statement separator on both targets)."""
    if ';' not in line:
        return [line]
    out, cur, in_str = [], [], False
    i = 0
    while i < len(line):
        c = line[i]
        if in_str:
            cur.append(c)
            if c == '\\' and i + 1 < len(line):
                cur.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            cur.append(c)
        elif c == ';':
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append(''.join(cur))
    return out


def find_include(path, incdirs, curdir):
    cands = [path] if os.path.isabs(path) else [os.path.join(d, path) for d in ['.', curdir] + incdirs]
    for c in cands:
        if os.path.isfile(c):
            return c
    die("cannot find include '%s'" % path)


_RE_INCLUDE = re.compile(r'^\s*\.include\s+"([^"]+)"\s*$')


def expand(lines, incdirs, curdir, depth=0):
    if depth > 32:
        die("include nesting too deep")
    for line in lines:
        m = _RE_INCLUDE.match(strip_comment(line))
        if m:
            p = find_include(m.group(1), incdirs, curdir)
            with open(p, encoding='latin-1') as f:
                sub = f.read().split('\n')
            yield '# 1 "%s"' % p
            for x in expand(sub, incdirs, os.path.dirname(p), depth + 1):
                yield x
        else:
            yield line


def convert(text, incdirs):
    macros = set()
    out = []
    for raw in expand(text.split('\n'), incdirs, '.'):
        if raw.startswith('#'):
            out.append(raw)  # cpp line markers / comments
            continue
        line = strip_comment(raw)
        md = _RE_MACRO_DEF.match(line)
        if md:
            macros.add(md.group(1))
            out.append(line)
            continue
        out.append(';'.join(convert_statement(st, macros) for st in split_statements(line)))
    return '\n'.join(out) + '\n'


def main(argv):
    asflags = []
    incdirs = []
    out = None
    src = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '-o':
            out = argv[i + 1]
            i += 2
            continue
        if a == '-I':
            incdirs.append(argv[i + 1])
            asflags += [a, argv[i + 1]]
            i += 2
            continue
        if a.startswith('-I'):
            incdirs.append(a[2:])
            asflags.append(a)
        elif a == '--defsym':
            asflags += [a, argv[i + 1]]
            i += 2
            continue
        elif a.startswith('-mcpu') or a.startswith('-march') or a in ('-mthumb', '-mthumb-interwork'):
            pass  # ARM-only flags
        elif a.startswith('-') and a != '-':
            asflags.append(a)
        elif a == '-':
            src = None
        else:
            if src is not None:
                die("more than one input file")
            src = a
        i += 1
    if out is None:
        die("no -o given")
    if src is None:
        text = sys.stdin.buffer.read().decode('latin-1')
    else:
        with open(src, encoding='latin-1') as f:
            text = f.read()
    conv = convert(text, incdirs)
    if os.environ.get('AS_X86_KEEP'):
        with open(out + '.x86.s', 'w', encoding='latin-1') as f:
            f.write(conv)
    tmp = out + '.raw.o'
    p = subprocess.run(['as', '--32'] + asflags + ['-o', tmp, '-'], input=conv.encode('latin-1'))
    if p.returncode != 0:
        dump = out + '.failed.s'
        with open(dump, 'w', encoding='latin-1') as f:
            f.write(conv)
        die("as failed (converted source kept in %s)" % dump)
    if os.environ.get('AS_X86_NO_MIRROR') or os.environ.get('AS_X86_NO_GSTATE'):
        os.replace(tmp, out)
        return
    sys.path.insert(0, HERE)
    import mirror_sections
    if mirror_sections.process('game', tmp, out) != 0:
        die("mirror section rename failed")
    os.remove(tmp)


if __name__ == '__main__':
    main(sys.argv[1:])
