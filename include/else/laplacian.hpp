#pragma once

#include "else/else_solver.hpp"
#include <numerics.hpp>
#include <vector>

namespace else_sim {

/// Connected neighborhood grown in descending stationary weight.
[[nodiscard]] std::vector<num::idx>
laplacian_neighborhood(const num::SparseMatrix& laplacian,
                       const std::vector<double>& stationary_sqrt,
                       num::idx initial,
                       num::idx capacity);

/// Convert one symmetric Laplacian restriction into a Markov generator.
[[nodiscard]] DensitySubnetwork
laplacian_density_subnetwork(const num::SparseMatrix& laplacian,
                             const std::vector<double>& stationary_sqrt,
                             const std::vector<num::idx>& states);

/// Build successive restrictions, retaining every previous boundary destination.
[[nodiscard]] std::vector<DensitySubnetwork>
laplacian_density_subnetworks(const num::SparseMatrix& laplacian,
                              const std::vector<double>& stationary_sqrt,
                              num::idx initial,
                              num::idx capacity,
                              int count);

/// Sample a trajectory by eliminating reversible Laplacian subnetworks.
[[nodiscard]] markovkit::Trajectory
laplacian_else_trajectory(const num::SparseMatrix& laplacian,
                          const std::vector<double>& stationary_sqrt,
                          num::idx initial,
                          double initial_time,
                          double final_time,
                          ELSEOptions options = {},
                          int seed = 42);

} // namespace else_sim
