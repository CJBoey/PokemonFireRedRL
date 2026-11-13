# Native reward/mask/trail library (64-bit; NOT part of the -m32 CMake tree). The header is generated
# from pokerl.native.spec, so the build runs through Python: make -C rl/pokerl/native
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
PYTHON ?= python3

.PHONY: all clean
all:
	cd $(ROOT) && PYTHONPATH=rl:python $(PYTHON) -m pokerl.native.build

clean:
	rm -f libpokerl_native.so libpokerl_native.so.tmp gen_spec.h
