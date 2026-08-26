add_executable(fsp_oregonator fsp_oregonator.cpp)
target_link_libraries(fsp_oregonator PRIVATE markovkit::markovkit numerics::plot)

add_executable(fsp_toggle_switch fsp_toggle_switch.cpp)
target_link_libraries(fsp_toggle_switch PRIVATE markovkit::markovkit numerics::plot)

add_executable(ssa_birth_death ssa_birth_death.cpp)
target_link_libraries(ssa_birth_death PRIVATE markovkit::markovkit numerics::plot)

add_executable(michaelis_menten michaelis_menten.cpp)
target_link_libraries(michaelis_menten PRIVATE markovkit::markovkit numerics::plot)

add_executable(find_switching find_switching.cpp)
target_link_libraries(find_switching PRIVATE markovkit::markovkit)
