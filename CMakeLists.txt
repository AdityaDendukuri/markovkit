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

# header-only, like subsweep
add_library(markovkit INTERFACE)
add_library(markovkit::markovkit ALIAS markovkit)

target_include_directories(markovkit INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(markovkit INTERFACE numerics::numerics)

enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
