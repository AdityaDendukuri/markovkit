#pragma once

#include "else/else_solver.hpp"
#include <numerics.hpp>
#include <vector>

namespace else_sim {

/// A symmetric Laplacian and the weights defining its reversible Markov generator.
class ReversibleLaplacian {
public:
  ReversibleLaplacian(const num::SparseMatrix& laplacian,
                      const std::vector<double>& stationary_sqrt);

  [[nodiscard]] const num::SparseMatrix& matrix() const {
    return *laplacian_;
  }
  [[nodiscard]] const std::vector<double>& stationary_sqrt() const {
    return *stationary_sqrt_;
  }

private:
  const num::SparseMatrix* laplacian_;
  const std::vector<double>* stationary_sqrt_;
};

/// Connected neighborhood grown in descending stationary weight.
[[nodiscard]] std::vector<num::idx>
laplacian_neighborhood(const num::SparseMatrix& laplacian,
                       const std::vector<double>& stationary_sqrt,
                       num::idx initial,
                       num::idx capacity);

/// Convert one symmetric Laplacian restriction into a Markov generator.
[[nodiscard]] RestrictedGenerator
restrict_laplacian(const num::SparseMatrix& laplacian,
                   const std::vector<double>& stationary_sqrt,
                   const std::vector<num::idx>& states);

/// Build successive restrictions, retaining every previous boundary destination.
[[nodiscard]] std::vector<RestrictedGenerator>
laplacian_restrictions(const num::SparseMatrix& laplacian,
                       const std::vector<double>& stationary_sqrt,
                       num::idx initial,
                       num::idx capacity,
                       int count);

/// Sample a trajectory by eliminating reversible Laplacian subnetworks.
[[nodiscard]] markovkit::Trajectory
laplacian_else_trajectory(const ReversibleLaplacian& generator,
                          num::idx initial,
                          double initial_time,
                          double final_time,
                          ELSEOptions options = {},
                          int seed = 42);

} // namespace else_sim
