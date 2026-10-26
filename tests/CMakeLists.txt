add_executable(test_fsp test_fsp.cpp)
target_link_libraries(test_fsp PRIVATE markovkit::markovkit)
add_test(NAME test_fsp COMMAND test_fsp)

add_executable(test_ssa test_ssa.cpp)
target_link_libraries(test_ssa PRIVATE markovkit::markovkit)
add_test(NAME test_ssa COMMAND test_ssa)

add_executable(test_rdme test_rdme.cpp)
target_link_libraries(test_rdme PRIVATE markovkit::markovkit)
add_test(NAME test_rdme COMMAND test_rdme)
