#pragma once

#include "cme/reversible.hpp"
#include "else/trajectory.hpp"
#include <numerics.hpp>
#include <vector>

namespace else_sim {

/// Connected neighborhood grown in descending stationary weight.
[[nodiscard]] std::vector<num::idx>
laplacian_neighborhood(const num::spmat &laplacian,
                       const std::vector<double> &stationary_sqrt, num::idx initial,
                       num::idx capacity);

/// Convert one symmetric Laplacian restriction into a Markov generator.
/// Build successive restrictions, retaining every previous boundary destination.
[[nodiscard]] std::vector<cme::RestrictedGenerator>
laplacian_restrictions(const num::spmat &laplacian,
                       const std::vector<double> &stationary_sqrt, num::idx initial,
                       num::idx capacity, int count);

/// Sample a trajectory by eliminating reversible Laplacian subnetworks.
[[nodiscard]] markovkit::Trajectory
laplacian_else_trajectory(const cme::ReversibleLaplacian &generator, num::idx initial,
                          double initial_time, double final_time, ELSEOptions options = {},
                          int seed = 42);

} // namespace else_sim
