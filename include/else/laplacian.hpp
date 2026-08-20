#pragma once

#include "else/else_solver.hpp"
#include <numerics.hpp>
#include <vector>

namespace else_sim {

// Connected neighborhood prioritized by stationary weight.
[[nodiscard]] std::vector<num::idx>
laplacian_neighborhood(const num::SparseMatrix& laplacian,
                       const std::vector<double>& stationary_sqrt,
                       num::idx initial,
                       num::idx capacity);

// Build one generic ELSE density step from a symmetric Laplacian restriction.
[[nodiscard]] DensityStep laplacian_density_step(const num::SparseMatrix& laplacian,
                                                 const std::vector<double>& stationary_sqrt,
                                                 const std::vector<num::idx>& states);

// Build successive steps, retaining every boundary destination in the next support.
[[nodiscard]] std::vector<DensityStep>
laplacian_density_steps(const num::SparseMatrix& laplacian,
                        const std::vector<double>& stationary_sqrt,
                        num::idx initial,
                        num::idx capacity,
                        int step_count);

} // namespace else_sim
