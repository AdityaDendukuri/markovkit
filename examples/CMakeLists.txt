# examples/CMakeLists.txt
# Auto-discovers all *.cpp example files and creates executable targets linked to markovkit::markovkit.
# Also creates a custom meta-target 'markovkit_examples' to build all examples at once.

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_custom_target(markovkit_examples ALL)

foreach(SRC ${EXAMPLE_SOURCES})
    get_filename_component(TARGET_NAME ${SRC} NAME_WE)
    add_executable(${TARGET_NAME} ${SRC})
    target_link_libraries(${TARGET_NAME} PRIVATE markovkit::markovkit)
    add_dependencies(markovkit_examples ${TARGET_NAME})
endforeach()
