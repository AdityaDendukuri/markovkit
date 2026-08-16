/// @file else/fourier.hpp
/// @brief Talbot contour Laplace inversion resolvent solver for subnetwork
/// state propagation.
#pragma once

#include <numerics.hpp>
#include <vector>

namespace markovkit::else_solver {

/// @brief Weideman-Talbot contour node and weight parameters for Laplace
/// inversion.
struct TalbotNode {
  num::cplx s; ///< Shift parameter s in C
  num::cplx w; ///< Contour weight w in C
};

/// @brief Compute Weideman Talbot contour nodes and weights for time t.
std::vector<TalbotNode> talbot_nodes(num::idx N, num::real t);

/// @brief Reconstruct subnetwork state vector e^{Rt} v via Talbot contour
/// resolvent solves.
num::Vector
fourier_subnetwork_solve(const num::Matrix& R, const num::Vector& v, num::real t, num::idx N = 32);

} // namespace markovkit::else_solver
