# obs/: the observer. observation.h is the C ABI the worker fills (one PnObs record per env and
# step); the C sources are compiled with the game's headers and flags (cmake/GameReader.cmake),
# plus a C++ symbol resolver. python/fireredsim/obs_dtype.py is generated from the header.
pn_add_game_reader_library(pnobs
  src/observe.c
  src/wait.c
  src/skill_events.c
  src/learned_moves.c
  src/obs_resolve.cpp)
target_include_directories(pnobs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(pnobs PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-unused-function>)

add_custom_target(obs_dtype_py ALL
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/gen_obs_dtype.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/observation.h ${CMAKE_SOURCE_DIR}/tools/gen_obs_dtype.py
  COMMENT "Generating python/fireredsim/obs_dtype.py")
