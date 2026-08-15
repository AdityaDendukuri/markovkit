cmake_minimum_required(VERSION 3.20)
project(markovkit VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch numerics dependency
FetchContent_Declare(
  numerics
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../numerics"
)
FetchContent_MakeAvailable(numerics)

# Fetch approxchol dependency
FetchContent_Declare(
  approxchol
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../approxchol-cpp"
)
FetchContent_MakeAvailable(approxchol)

# Precompiled Static Library Target
add_library(markovkit
    src/fsp/state_space.cpp
    src/fsp/generator.cpp
    src/fsp/solvers/adaptive_fsp.cpp
    src/fsp/solvers/krylov_fsp.cpp
    src/else/subnetwork.cpp
    src/else/network.cpp
    src/else/density.cpp
    src/else/fourier.cpp
    src/else/ensemble.cpp
    src/else/exponential_fit.cpp
    src/ssa/gillespie.cpp
    src/ssa/tau_leaping.cpp
)

add_library(markovkit::markovkit ALIAS markovkit)

target_include_directories(markovkit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(markovkit PUBLIC
    numerics::numerics
    approxchol
)

enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
