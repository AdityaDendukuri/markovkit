add_executable(fsp_oregonator fsp_oregonator.cpp)
target_link_libraries(fsp_oregonator PRIVATE markovkit::markovkit numerics::plot)

add_executable(fsp_toggle_switch fsp_toggle_switch.cpp)
target_link_libraries(fsp_toggle_switch PRIVATE markovkit::markovkit numerics::plot)

add_executable(ssa_birth_death ssa_birth_death.cpp)
target_link_libraries(ssa_birth_death PRIVATE markovkit::markovkit numerics::plot)

add_executable(rdme_multigrid_benchmark rdme_multigrid_benchmark.cpp)
target_link_libraries(rdme_multigrid_benchmark PRIVATE markovkit::markovkit)

add_executable(rdme_two_particle_annihilation rdme_two_particle_annihilation.cpp)
target_link_libraries(rdme_two_particle_annihilation PRIVATE markovkit::markovkit numerics::plot)

add_executable(rdme_multigrid_analysis rdme_multigrid_analysis.cpp)
target_link_libraries(rdme_multigrid_analysis PRIVATE markovkit::markovkit numerics::plot)
