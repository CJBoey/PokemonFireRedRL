"""Build the native library: ``PYTHONPATH=rl:python python3 -m pokerl.native.build`` (or ``make -C rl/pokerl/native``).

Writes ``gen_spec.h`` (field-group positions, sizes, constants and the spec hash, from
:mod:`pokerl.native.spec`) and compiles ``pokerl_native.c`` into ``libpokerl_native.so`` next to this
file: 64-bit, ``-O3 -fPIC -shared`` (SSE2 intrinsics only, the x86-64 baseline), no ``-ffast-math`` and
``-ffp-contract=off``: no float operation is reassociated or fused (the reward's float order is fixed). Deliberately
outside the project's CMake tree, which builds the game -m32.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "pokerl_native.c")
HEADER = os.path.join(HERE, "gen_spec.h")
LIBRARY = os.path.join(HERE, "libpokerl_native.so")
CFLAGS = ["-std=c11", "-O3", "-fPIC", "-shared", "-m64", "-ffp-contract=off", "-fno-math-errno",
          "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror"]


def write_header() -> bool:
    """(Re)writes gen_spec.h; True if it changed."""
    from . import spec
    text = spec.header()
    old = None
    if os.path.exists(HEADER):
        with open(HEADER) as f:
            old = f.read()
    if old != text:
        with open(HEADER, "w") as f:
            f.write(text)
        return True
    return False


def build(cc: str = None, verbose: bool = True) -> str:
    changed = write_header()
    cc = cc or os.environ.get("CC", "gcc")
    newest = max(os.path.getmtime(p) for p in (SOURCE, HEADER, __file__))
    if not changed and os.path.exists(LIBRARY) and os.path.getmtime(LIBRARY) >= newest:
        if verbose:
            print(f"{LIBRARY} is up to date")
        return LIBRARY
    tmp = LIBRARY + ".tmp"
    cmd = [cc, *CFLAGS, "-o", tmp, SOURCE, "-lm"]
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=HERE)
    os.replace(tmp, LIBRARY)  # atomic: a running process keeps its mapped copy
    return LIBRARY


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cc", default=None)
    ap.add_argument("--header-only", action="store_true")
    a = ap.parse_args(argv)
    if a.header_only:
        write_header()
        return 0
    build(a.cc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
