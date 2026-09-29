cmake_minimum_required(VERSION 3.20)
project(markovkit VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    numerics
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../numerics"
)
FetchContent_MakeAvailable(numerics)

add_library(markovkit
    src/cme/state_space.cpp
    src/cme/generator.cpp
    src/cme/reversible.cpp
    src/fsp/solution.cpp
    src/fsp/solvers/adaptive_fsp.cpp
    src/fsp/solvers/krylov_fsp.cpp
)
add_library(markovkit::markovkit ALIAS markovkit)

target_include_directories(markovkit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(markovkit PUBLIC numerics::numerics)

enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
