# examples/CMakeLists.txt
# Umbrella target to build all examples: cmake --build build --target markovkit_examples
add_custom_target(markovkit_examples ALL)

# Helper function to register an example target with full control
function(add_markovkit_example TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE markovkit::markovkit)
    add_dependencies(markovkit_examples ${TARGET_NAME})
endfunction()

# ==============================================================================
# Explicitly Registered Showcase Examples
# Add your custom example target here!
# ==============================================================================

add_markovkit_example(gene_expression        gene_expression.cpp)
add_markovkit_example(plot_gene_expression   plot_gene_expression.cpp)
add_markovkit_example(oregonator_example     oregonator_example.cpp)
add_markovkit_example(toggle_switch_example  toggle_switch_example.cpp)
