/// @file else/fourier.hpp
/// @brief Talbot contour Laplace inversion resolvent solver for subnetwork state propagation.
#pragma once

#include <numerics.hpp>
#include <vector>
#include <complex>
#include <cmath>

namespace markovkit::else_solver {

using Complex = std::complex<double>;

/// @brief Weideman-Talbot contour node and weight parameters for Laplace inversion.
struct TalbotNode {
    Complex s; ///< Shift parameter s in C
    Complex w; ///< Contour weight w in C
};

/// @brief Compute Weideman Talbot contour nodes and weights for time t.
inline std::vector<TalbotNode> talbot_nodes(std::size_t N, double t) {
    if (N % 2 != 0) N += 1; // N must be even
    std::vector<TalbotNode> nodes;
    nodes.reserve(N);

    const double pi = std::acos(-1.0);
    const Complex i_unit(0.0, 1.0);

    const double dtheta = 2.0 * pi / static_cast<double>(N);
    const double a = 0.5017;
    const double b = 0.6407;
    const double c = 0.6122;
    const double d = 0.2645;

    for (std::size_t k = 1; k <= N; ++k) {
        double theta = -pi + (static_cast<double>(k) - 0.5) * dtheta;
        double b_theta = b * theta;
        double sin_b = std::sin(b_theta);
        double cos_b = std::cos(b_theta);
        double cotb = cos_b / sin_b;
        double csc2b = 1.0 / (sin_b * sin_b);

        Complex zeta = static_cast<double>(N) * (a * theta * cotb - c + i_unit * (d * theta));
        Complex z = zeta / t;

        Complex dzeta = static_cast<double>(N) * (a * (cotb - b * theta * csc2b) + i_unit * d);
        Complex dz = dzeta / t;

        Complex w = (dtheta / (2.0 * pi * i_unit)) * std::exp(z * t) * dz;

        nodes.push_back({z, w});
    }
    return nodes;
}

/// @brief Reconstruct subnetwork state vector e^{Rt} v via Talbot contour resolvent solves:
///        e^{Rt} v = Re { \sum_{k=1}^N w_k (s_k I - R)^{-1} v }
inline num::Vector fourier_subnetwork_solve(const num::Matrix& R, const num::Vector& v, double t, std::size_t N = 32) {
    num::idx n = R.rows();
    num::Vector result(n, 0.0);

    auto nodes = talbot_nodes(N, t);

    for (const auto& node : nodes) {
        // Call num::resolvent_solve to compute (s_k * I - R)^{-1} v
        std::vector<Complex> sol = num::resolvent_solve(node.s, R, v);
        for (num::idx i = 0; i < n; ++i) {
            Complex contrib = node.w * sol[i];
            result[i] += contrib.real();
        }
    }
    return result;
}

} // namespace markovkit::else_solver
