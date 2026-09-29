add_executable(test_fsp test_fsp.cpp)
target_link_libraries(test_fsp PRIVATE markovkit::markovkit)
add_test(NAME test_fsp COMMAND test_fsp)

add_executable(test_ssa test_ssa.cpp)
target_link_libraries(test_ssa PRIVATE markovkit::markovkit)
add_test(NAME test_ssa COMMAND test_ssa)

add_executable(test_rdme test_rdme.cpp)
target_link_libraries(test_rdme PRIVATE markovkit::markovkit)
add_test(NAME test_rdme COMMAND test_rdme)

# ELSE suites report through check.hpp so they still verify under NDEBUG.
foreach(suite law shedding ensemble density laplacian factor_update)
    add_executable(test_else_${suite} test_else_${suite}.cpp)
    target_link_libraries(test_else_${suite} PRIVATE markovkit::markovkit)
    target_include_directories(test_else_${suite} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME test_else_${suite} COMMAND test_else_${suite})
endforeach()
