cmake_minimum_required(VERSION 3.20)
project(markovkit VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# 1. numerics dependency
if(EXISTS "/Users/adityadendukuri/numerics/CMakeLists.txt")
    FetchContent_Declare(numerics
        SOURCE_DIR /Users/adityadendukuri/numerics
    )
else()
    FetchContent_Declare(numerics
        GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
        GIT_TAG        main
    )
endif()

set(NUMERICS_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(NUMERICS_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(NUMERICS_BUILD_DOCS       OFF CACHE BOOL "" FORCE)
set(NUMERICS_BUILD_REPORT     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(numerics)

# 2. approxchol dependency
if(EXISTS "/Users/adityadendukuri/approxchol-cpp/CMakeLists.txt")
    FetchContent_Declare(approxchol
        SOURCE_DIR /Users/adityadendukuri/approxchol-cpp
    )
    FetchContent_MakeAvailable(approxchol)
endif()

# markovkit target
add_library(markovkit
    src/fsp/state_space.cpp
    src/fsp/generator.cpp
    src/fsp/solution.cpp
    src/fsp/solvers/adaptive_fsp.cpp
    src/fsp/solvers/krylov_fsp.cpp
    src/else/subnetwork.cpp
    src/else/exponential_fit.cpp
)
add_library(markovkit::markovkit ALIAS markovkit)

target_compile_features(markovkit PUBLIC cxx_std_20)
target_include_directories(markovkit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(markovkit PUBLIC numerics::numerics)
if(TARGET approxchol::approxchol)
    target_link_libraries(markovkit PUBLIC approxchol::approxchol)
endif()

# Build tests and examples
option(MARKOVKIT_BUILD_TESTS "Build markovkit tests" ON)
if(MARKOVKIT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(MARKOVKIT_BUILD_EXAMPLES "Build markovkit examples" ON)
if(MARKOVKIT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
