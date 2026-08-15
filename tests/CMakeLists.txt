add_executable(test_fsp test_fsp.cpp)
target_link_libraries(test_fsp PRIVATE markovkit::markovkit)
add_test(NAME test_fsp COMMAND test_fsp)

add_executable(test_else test_else.cpp)
target_link_libraries(test_else PRIVATE markovkit::markovkit)
add_test(NAME test_else COMMAND test_else)

add_executable(test_fess test_fess.cpp)
target_link_libraries(test_fess PRIVATE markovkit::markovkit)
add_test(NAME test_fess COMMAND test_fess)
