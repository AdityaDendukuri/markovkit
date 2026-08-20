add_executable(fsp_oregonator fsp_oregonator.cpp)
target_link_libraries(fsp_oregonator PRIVATE markovkit::markovkit)

add_executable(fsp_toggle_switch fsp_toggle_switch.cpp)
target_link_libraries(fsp_toggle_switch PRIVATE markovkit::markovkit)

add_executable(else_oregonator else_oregonator.cpp)
target_link_libraries(else_oregonator PRIVATE markovkit::markovkit)

add_executable(else_birth_death else_birth_death.cpp)
target_link_libraries(else_birth_death PRIVATE markovkit::markovkit)

add_executable(ssa_birth_death ssa_birth_death.cpp)
target_link_libraries(ssa_birth_death PRIVATE markovkit::markovkit)

add_executable(michaelis_menten michaelis_menten.cpp)
target_link_libraries(michaelis_menten PRIVATE markovkit::markovkit)

add_executable(laplacian_committor laplacian_committor.cpp)
target_link_libraries(laplacian_committor PRIVATE markovkit::markovkit)

add_executable(c_lapl c_lapl.cpp)
target_link_libraries(c_lapl PRIVATE markovkit::markovkit)
