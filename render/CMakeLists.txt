# render/: the GBA PPU renderer the worker uses for pixel observations (render/README.md).
# Consumers: target_link_libraries(x PRIVATE pnrender) and #include <pn_render.h>.
option(PNRENDER_NO_AVX2 "Compile the AVX2 kernels out (only the generic implementation remains)" OFF)

# ---- the library ---------------------------------------------------------------------------------
add_library(pnrender STATIC
    src/pn_render.c
    src/pn_ppu.c)
target_include_directories(pnrender PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(pnrender PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS ON)
# Always optimised (-O2, after the build type's flags), whatever the parent's build type: a frame
# must stay well under 1 ms. -fwrapv: the mGBA-derived affine-sprite arithmetic overflows int for
# extreme (fuzz-only) matrices; wrapping is what mGBA's own binary does (README, "Purity").
# Scoped to the target.
target_compile_options(pnrender PRIVATE -O2 -fwrapv -Wall -Wextra)
# The AVX2 kernels are compiled with function-level target attributes and selected at run time
# (pn_render_auto_isa); nothing here changes the ISA of the rest of the library.
if(PNRENDER_NO_AVX2)
    target_compile_definitions(pnrender PRIVATE PNRENDER_NO_AVX2)
endif()
