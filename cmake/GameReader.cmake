# pn_add_game_reader_library(<name> <sources...>)
#
# Static library of C sources that READ game state from the host side
# (observations, autoplay probes, renderer adapters). They include the game's
# headers, so they are compiled with exactly the game's preprocessing and ABI
# flags (game/Makefile.native: defines, force-included gba/portable_abi.h,
# -funsigned-char, -malign-data=abi, ...) and therefore see the retail struct
# layouts. They are host code, not game code: their own variables are host state
# (NOT part of the snapshotted game regions), and they must never write game
# memory. Link the result together with pnsim.
#
# Keep the flag list in sync with game/Makefile.native (CPPFLAGS, ARCH_CFLAGS).
set(PN_GAME_READER_FLAGS
  -iquote ${PN_GAME_DIR}/include -Wno-trigraphs
  -DFIRERED -DREVISION=0 -DENGLISH -DMODERN=1 -DPORTABLE -DNONMATCHING -DUBFIX
  -include ${PN_GAME_DIR}/include/gba/portable_abi.h
  -Ui386 -Ulinux -Uunix -funsigned-char -U_FORTIFY_SOURCE
  -std=gnu17 -O2 -msse2 -mfpmath=sse -malign-data=abi -fwrapv -fno-strict-aliasing
  -fno-delete-null-pointer-checks -fno-aggressive-loop-optimizations -fno-common
  -fno-pic -fno-stack-protector
  -Wno-packed-not-aligned -Wno-ignored-qualifiers -Wno-unused-parameter)

function(pn_add_game_reader_library name)
  add_library(${name} STATIC ${ARGN})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:C>:${PN_GAME_READER_FLAGS}>)
  target_link_libraries(${name} PUBLIC pnsim)
endfunction()
