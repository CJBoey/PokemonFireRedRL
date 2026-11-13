# worker/: the environment pool library and fireredsim-worker, the shared-memory host process that
# python/fireredsim drives (worker/pn_ipc.h is the protocol).
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPN_FAST=ON
#   cmake --build build --target fireredsim-worker

# The bedroom start script, embedded so that the worker is self-contained.
set(PN_WORKER_BEDROOM_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/new_game_to_pallet.txt)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${PN_WORKER_BEDROOM_SCRIPT})
file(READ ${PN_WORKER_BEDROOM_SCRIPT} PN_BEDROOM_SCRIPT_TEXT)
configure_file(bedroom_script.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/bedroom_script.cpp @ONLY)

pn_add_game_reader_library(pnworker_probe save_start_probe.c)
pn_add_game_reader_library(pnworker_preset start_preset.c)

add_library(pnenvpool STATIC
  envpool.cpp
  worker_loop.cpp
  save_start_tools.cpp
  render_adapter.cpp
  obs_bridge.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/bedroom_script.cpp)
target_include_directories(pnenvpool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pnenvpool PUBLIC pnsim pnrender pnworker_probe pnworker_preset pnobs)

add_executable(fireredsim-worker worker_main.cpp)
target_link_libraries(fireredsim-worker PRIVATE pnenvpool)
pn_game_executable(fireredsim-worker)
