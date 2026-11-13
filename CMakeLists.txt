# fireredsim: host side (sim/, cli/, tests/) built with CMake + Ninja.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j32
#   ctest --test-dir build
#
# The game and the platform layer are built by game/Makefile.native (pret-style
# asset and data pipeline) into two static archives plus a linker fragment, in
# <build dir>/game-native (every build directory has its own objects, so build
# directories of different source states or flags never mix game code);
# this file drives that build and links everything as 32-bit (i386, ILP32),
# non-PIE executables (ARCHITECTURE.md sections 1.2 and 1.4) with the mirror
# memory layout (docs/MIRROR_LAYOUT.md): the executable's code and read-only
# data at 0x08000000 (the GBA's ROM window), EWRAM/IWRAM/IO/palette/VRAM/OAM/
# flash at their GBA addresses with every game RAM variable at its retail ROM
# address, native-only state in .pnstate at 0x10000000.
cmake_minimum_required(VERSION 3.20)
project(fireredsim C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# Everything is 32-bit: the game's globals live at fixed addresses of a non-PIE
# i386 executable.
set(PN_ARCH_FLAGS -m32 -msse2 -mfpmath=sse -fno-pie)
add_compile_options(${PN_ARCH_FLAGS} -Wall -Wextra)
add_link_options(-m32 -no-pie -static-libgcc -static-libstdc++)

# ---- game + platform (game/Makefile.native) ---------------------------------
set(PN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(PN_GAME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/game)
if(NOT EXISTS ${PN_GAME_DIR}/.pret_commit)
  message(FATAL_ERROR "game/ holds only this repository's own files: run tools/fetch_game.sh first (it downloads "
                      "pret/pokefirered and applies game/native/pokefirered.patch)")
endif()
# Per build directory: Makefile.native's OBJ_DIR (objects, libfirered.a, libplatform.a,
# mirror.ld). Not game/build/native, which every build directory used to share. The
# asset pipeline's outputs (graphics, generated data) still go to the source tree.
set(PN_GAME_BUILD ${CMAKE_BINARY_DIR}/game-native)
set(PN_LIBFIRERED ${PN_GAME_BUILD}/libfirered.a)
set(PN_LIBPLATFORM ${PN_GAME_BUILD}/libplatform.a)
set(PN_MIRROR_LD ${PN_GAME_BUILD}/mirror.ld)

# Parallelism of the game build (make -j); -DPN_NPROC=8 limits it.
if(NOT PN_NPROC)
  include(ProcessorCount)
  ProcessorCount(PN_NPROC)
  if(PN_NPROC EQUAL 0)
    set(PN_NPROC 8)
  endif()
endif()

# Code-generation options of the game and platform code (README.md section 3,
# docs/BENCHMARKS.md "Compiler flags"). The defaults are the reference build that
# the differential tests use; none of them touches a layout- or semantics-relevant
# flag (game/Makefile.native ARCH_CFLAGS).
#   PN_FAST=ON          the training build: PN_GAME_OPT=-O3 and PN_STACK_GUARD=OFF
#                       unless those are set explicitly; PGO on top: tools/pgo_build.sh
#   PN_GAME_OPT         optimisation level (default -O2)
#   PN_GAME_TUNE        extra flags, e.g. -march=znver4 (machine-specific binary)
#   PN_PGO              "" | generate | use (tools/pgo_build.sh drives both stages;
#                       the .gcda profiles stay in <build dir>/game-native)
#   PN_STACK_GUARD      check the game stack's guard area after every frame
#                       (platform/frame.c CheckGameStack; default ON)
option(PN_FAST "Training build: game code at -O3, no stack guard check" OFF)
if(PN_FAST)
  set(pn_default_opt -O3)
  set(pn_default_guard OFF)
else()
  set(pn_default_opt -O2)
  set(pn_default_guard ON)
endif()
if(NOT DEFINED PN_GAME_OPT OR PN_GAME_OPT STREQUAL "")
  set(PN_GAME_OPT_EFFECTIVE ${pn_default_opt})
else()
  set(PN_GAME_OPT_EFFECTIVE ${PN_GAME_OPT})
endif()
set(PN_GAME_OPT "" CACHE STRING "Optimisation level of game + platform code (empty: -O2, -O3 with PN_FAST)")
set(PN_GAME_TUNE "" CACHE STRING "Extra code-generation flags for game + platform code (e.g. -march=znver4)")
set(PN_PGO "" CACHE STRING "Profile-guided optimisation stage of game + platform code: '', generate or use")
set_property(CACHE PN_PGO PROPERTY STRINGS "" generate use)
if(NOT DEFINED PN_STACK_GUARD OR PN_STACK_GUARD STREQUAL "")
  set(PN_STACK_GUARD_EFFECTIVE ${pn_default_guard})
else()
  set(PN_STACK_GUARD_EFFECTIVE ${PN_STACK_GUARD})
endif()
set(PN_STACK_GUARD "" CACHE STRING "Game stack guard check after every frame: ON/OFF (empty: ON, OFF with PN_FAST)")

set(pn_tune ${PN_GAME_TUNE})
if(PN_PGO STREQUAL "generate")
  # -fprofile-update=single: one thread per process runs game code.
  string(APPEND pn_tune " -fprofile-generate -fprofile-update=single")
elseif(PN_PGO STREQUAL "use")
  # Functions the training run never reached keep their normal optimisation.
  string(APPEND pn_tune " -fprofile-use -fprofile-partial-training -Wno-missing-profile")
elseif(NOT PN_PGO STREQUAL "")
  message(FATAL_ERROR "PN_PGO must be empty, generate or use (got '${PN_PGO}')")
endif()
string(STRIP "${pn_tune}" pn_tune)
set(pn_platform_defs "")
if(NOT PN_STACK_GUARD_EFFECTIVE)
  set(pn_platform_defs -DPN_STACK_GUARD=0)
endif()
set(PN_GAME_MAKE_OPTIONS "PN_OPT=${PN_GAME_OPT_EFFECTIVE}" "PN_TUNE=${pn_tune}" "PN_PLATFORM_DEFS=${pn_platform_defs}")
message(STATUS "fireredsim game code: ${PN_GAME_OPT_EFFECTIVE} tune='${pn_tune}' stack guard=${PN_STACK_GUARD_EFFECTIVE}")
if(PN_PGO STREQUAL "generate")
  # The instrumented executables need libgcov (the game objects reference it).
  add_link_options(-fprofile-generate)
endif()

add_custom_target(game_libs ALL
  COMMAND make -C ${PN_GAME_DIR} -f Makefile.native -j${PN_NPROC} libs OBJ_DIR=${PN_GAME_BUILD} ${PN_GAME_MAKE_OPTIONS}
  BYPRODUCTS ${PN_LIBFIRERED} ${PN_LIBPLATFORM} ${PN_MIRROR_LD}
  COMMENT "Building the game and the platform layer (game/Makefile.native -> ${PN_GAME_BUILD})"
  USES_TERMINAL VERBATIM)

add_library(pn_game INTERFACE)
add_dependencies(pn_game game_libs)
target_include_directories(pn_game INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/platform)
target_link_options(pn_game INTERFACE
  -Wl,-Ttext-segment=0x08000000
  -Wl,-T,${PN_MIRROR_LD}
  -Wl,--whole-archive ${PN_LIBFIRERED} ${PN_LIBPLATFORM} -Wl,--no-whole-archive)
target_link_libraries(pn_game INTERFACE m)

# ---- sim: host library -------------------------------------------------------
add_library(pnsim STATIC
  sim/sim.cpp
  sim/symbols.cpp
  sim/input_script.cpp)
target_include_directories(pnsim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/sim ${CMAKE_CURRENT_SOURCE_DIR}/platform)
target_link_libraries(pnsim PUBLIC pn_game)

# Executables that contain the game get a link map and the mirror-layout
# address check (every ROM RAM symbol at its ROM address and size).
function(pn_game_executable target)
  target_link_options(${target} PRIVATE -Wl,-Map,$<TARGET_FILE:${target}>.map)
  # The archives come from a custom target: relink whenever they change.
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${PN_LIBFIRERED} ${PN_LIBPLATFORM} ${PN_MIRROR_LD})
  # Not for the PGO training build (PN_PGO=generate): its executables only collect
  # profiles, and the profiler's counters are writable data outside the state regions.
  if(NOT PN_PGO STREQUAL "generate")
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND python3 ${PN_ROOT}/tools/mirror/check_addresses.py --quiet $<TARGET_FILE:${target}>
      COMMENT "Checking the mirror memory layout of ${target}")
  endif()
endfunction()

# ---- tests -----------------------------------------------------------------------
enable_testing()
add_executable(test_determinism tests/determinism/test_determinism.cpp)
target_link_libraries(test_determinism PRIVATE pnsim)
pn_game_executable(test_determinism)

set(PN_PALLET_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/worker/new_game_to_pallet.txt)
add_test(NAME determinism_replay_twice
  COMMAND test_determinism replay-twice ${PN_PALLET_SCRIPT})
add_test(NAME determinism_snapshot_restore
  COMMAND test_determinism snapshot-restore ${PN_PALLET_SCRIPT})
add_test(NAME determinism_interleaved_envs
  COMMAND test_determinism interleaved ${PN_PALLET_SCRIPT})
add_test(NAME determinism_reset_template
  COMMAND test_determinism reset ${PN_PALLET_SCRIPT})

include(cmake/GameReader.cmake)

# observer (obs/), renderer (render/), worker process (worker/)
add_subdirectory(obs)
add_subdirectory(render)
add_subdirectory(worker)
