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

add_executable(density_laplacian density_laplacian.cpp)
target_link_libraries(density_laplacian PRIVATE markovkit::markovkit)

add_executable(laplacian_trajectories laplacian_trajectories.cpp)
target_link_libraries(laplacian_trajectories PRIVATE markovkit::markovkit)

add_executable(dense_cholesky_cme dense_cholesky_cme.cpp)
target_link_libraries(dense_cholesky_cme PRIVATE markovkit::markovkit)
