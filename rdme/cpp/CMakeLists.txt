cmake_minimum_required(VERSION 3.24)
project(graph_multigrid_fsp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(_default_numerics "${CMAKE_CURRENT_SOURCE_DIR}/../../../numerics" ABSOLUTE)
set(NUMERICS_SOURCE_DIR "${_default_numerics}" CACHE PATH "Path to the numerics source tree")
set(NUMERICS_BUILD_DIR "${NUMERICS_SOURCE_DIR}/build-clang" CACHE PATH
    "Path to a configured numerics build")

add_library(numerics_prebuilt STATIC IMPORTED)
set_target_properties(numerics_prebuilt PROPERTIES
    IMPORTED_LOCATION "${NUMERICS_BUILD_DIR}/libnumerics.a"
    INTERFACE_INCLUDE_DIRECTORIES "${NUMERICS_SOURCE_DIR}/include")

find_package(PkgConfig REQUIRED)
pkg_check_modules(KLU REQUIRED IMPORTED_TARGET klu)
pkg_check_modules(SUITESPARSE_CONFIG REQUIRED IMPORTED_TARGET SuiteSparse_config)

add_executable(rdme_full_multigrid rdme_full_multigrid.cpp)
target_link_libraries(rdme_full_multigrid PRIVATE numerics_prebuilt PkgConfig::KLU PkgConfig::SUITESPARSE_CONFIG)

# Exact-SSA spatial front demonstration; independent of the FSP solver.
add_executable(schlogl_spatial_front schlogl_spatial_front.cpp)

enable_testing()
add_test(NAME rdme_full_multigrid_smoke
    COMMAND rdme_full_multigrid --cells 4 --cap 12 --initial 8 --check)
add_test(NAME rdme_full_multigrid_2d
    COMMAND rdme_full_multigrid --cells 2 --dimension 2 --cap 8 --check --cycles 2)
add_test(NAME rdme_full_multigrid_3d
    COMMAND rdme_full_multigrid --cells 2 --dimension 3 --cap 4 --initial 2 --check --cycles 2)
add_test(NAME rdme_full_multigrid_population_block
    COMMAND rdme_full_multigrid --cells 4 --cap 8 --initial 4 --smoother block-ilu --sweeps 1 --cycles 2 --check)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME rdme_reflection_fsp
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../paper/scripts/check_reflection_fsp.py
            --executable $<TARGET_FILE:rdme_full_multigrid>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/reflection_fsp)
    add_test(NAME rdme_structured_population_blocks
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../paper/scripts/check_structured_population_blocks.py
            --executable $<TARGET_FILE:rdme_full_multigrid>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/structured_population_blocks)
    add_test(NAME schlogl_spatial_front_validation
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../paper/scripts/check_spatial_schlogl_front.py
            --executable $<TARGET_FILE:schlogl_spatial_front>)
    add_test(NAME rdme_adaptive_population_fsp
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../paper/scripts/check_adaptive_population_fsp.py
            --executable $<TARGET_FILE:rdme_full_multigrid>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/adaptive_population_fsp)
endif()
add_test(NAME rdme_full_multigrid_pair_chain
    COMMAND rdme_full_multigrid --cells 4 --cap 8 --initial 4 --smoother chain --sweeps 1 --cycles 2 --check)
add_test(NAME rdme_full_multigrid_pair_chain_2d
    COMMAND rdme_full_multigrid --cells 4 --dimension 2 --cap 4 --initial 2 --linear 1 --shift .5 --smoother chain --sweeps 1 --cycles 2 --check)
add_test(NAME rdme_full_multigrid_graph_neighborhood
    COMMAND rdme_full_multigrid --cells 4 --cap 8 --initial 4 --smoother overlap --sweeps 1 --cycles 2 --check)
add_test(NAME rdme_full_multigrid_overlap_2d
    COMMAND rdme_full_multigrid --cells 4 --dimension 2 --cap 4 --initial 2 --linear 1 --shift .5 --smoother overlap --sweeps 1 --cycles 2 --check)

add_executable(test_cascading_smoother test_cascading_smoother.cpp)
add_test(NAME cascading_smoother_algebra COMMAND test_cascading_smoother)
add_test(NAME rdme_cascade_linear
    COMMAND rdme_full_multigrid --cells 8 --cap 3 --initial 2 --linear 1 --shift .5 --smoother cascade --cascade-depth 2 --sweeps 1 --cycles 2 --check)
add_test(NAME rdme_cascade_schlogl
    COMMAND rdme_full_multigrid --cells 4 --cap 12 --initial 8 --smoother cascade --cascade-depth 3 --sweeps 1 --cycles 2 --check)
add_test(NAME rdme_cascade_truncated
    COMMAND rdme_full_multigrid --cells 8 --cap 3 --initial 2 --linear 1 --shift .5 --smoother cascade --cascade-depth 3 --cascade-row-cap 8 --sweeps 1 --cycles 2 --cycle-type k --check)

# Full block fits use LAPACK's rank-revealing least-squares driver.
find_path(BLOCK_CASCADE_LAPACKE_INCLUDE_DIR lapacke.h REQUIRED
    HINTS /opt/homebrew/opt/openblas/include /usr/local/opt/openblas/include
    PATH_SUFFIXES openblas)
find_library(BLOCK_CASCADE_LAPACKE_LIBRARY NAMES openblas lapacke REQUIRED
    HINTS /opt/homebrew/opt/openblas/lib /usr/local/opt/openblas/lib)
add_executable(test_block_cascading_smoother test_block_cascading_smoother.cpp)
target_link_libraries(test_block_cascading_smoother PRIVATE numerics_prebuilt)
foreach(_target rdme_full_multigrid test_block_cascading_smoother)
    target_include_directories(${_target} PRIVATE ${BLOCK_CASCADE_LAPACKE_INCLUDE_DIR})
    target_link_libraries(${_target} PRIVATE ${BLOCK_CASCADE_LAPACKE_LIBRARY})
    target_compile_definitions(${_target} PRIVATE NUMERICS_HAS_BLAS)
endforeach()

add_test(NAME block_cascading_smoother_algebra COMMAND test_block_cascading_smoother)
add_test(NAME rdme_block_cascade_linear
    COMMAND rdme_full_multigrid --cells 8 --cap 3 --initial 2 --linear 1 --shift .5 --smoother block-cascade --cascade-depth 2 --cascade-order reverse-pre --sweeps 1 --cycles 2 --cycle-type k --check)
add_test(NAME rdme_block_cascade_schlogl
    COMMAND rdme_full_multigrid --cells 4 --cap 8 --initial 4 --smoother block-cascade --cascade-depth 2 --sweeps 1 --cycles 2 --check)
add_test(NAME rdme_block_cascade_forward
    COMMAND rdme_full_multigrid --cells 4 --cap 8 --initial 4 --smoother block-cascade --cascade-depth 2 --cascade-order forward --sweeps 1 --cycles 2 --cycle-type k --check)
