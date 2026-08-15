/// @file else/fourier.hpp
/// @brief Talbot contour Laplace inversion resolvent solver for subnetwork state propagation.
#pragma once

#include <numerics.hpp>
#include <vector>
#include <complex>

namespace markovkit::else_solver {

using Complex = std::complex<double>;

/// @brief Weideman-Talbot contour node and weight parameters for Laplace inversion.
struct TalbotNode {
    Complex s; ///< Shift parameter s in C
    Complex w; ///< Contour weight w in C
};

/// @brief Compute Weideman Talbot contour nodes and weights for time t.
std::vector<TalbotNode> talbot_nodes(std::size_t N, double t);

/// @brief Reconstruct subnetwork state vector e^{Rt} v via Talbot contour resolvent solves.
num::Vector fourier_subnetwork_solve(const num::Matrix& R, const num::Vector& v, double t, std::size_t N = 32);

} // namespace markovkit::else_solver
