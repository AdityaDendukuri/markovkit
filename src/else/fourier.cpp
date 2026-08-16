/// @file src/else/fourier.cpp
/// @brief Compiled Talbot contour resolvent Laplace inversion routines using
/// num::resolvent_solve_batch.
#include "markovkit/else/fourier.hpp"

namespace markovkit::else_solver {

std::vector<TalbotNode> talbot_nodes(num::idx N, num::real t) {
  if (N % 2 != 0)
    N += 1;
  std::vector<TalbotNode> nodes;
  nodes.reserve(N);

  const num::real pi = std::acos(-1.0);
  const num::cplx i_unit(0.0, 1.0);

  const num::real dtheta = 2.0 * pi / static_cast<num::real>(N);
  const num::real a = 0.5017;
  const num::real b = 0.6407;
  const num::real c = 0.6122;
  const num::real d = 0.2645;

  for (num::idx k = 1; k <= N; ++k) {
    num::real theta = -pi + (static_cast<num::real>(k) - 0.5) * dtheta;
    num::real b_theta = b * theta;
    num::real sin_b = std::sin(b_theta);
    num::real cos_b = std::cos(b_theta);
    num::real cotb = cos_b / sin_b;
    num::real csc2b = 1.0 / (sin_b * sin_b);

    num::cplx zeta = static_cast<num::real>(N) * (a * theta * cotb - c + i_unit * (d * theta));
    num::cplx z = zeta / t;

    num::cplx dzeta = static_cast<num::real>(N) * (a * (cotb - b * theta * csc2b) + i_unit * d);
    num::cplx dz = dzeta / t;

    num::cplx w = (dtheta / (2.0 * pi * i_unit)) * std::exp(z * t) * dz;

    nodes.push_back({z, w});
  }
  return nodes;
}

num::Vector
fourier_subnetwork_solve(const num::Matrix& R, const num::Vector& v, num::real t, num::idx N) {
  num::idx n = R.rows();
  num::Vector result(n, 0.0);

  auto nodes = talbot_nodes(N, t);
  std::vector<num::cplx> shifts(nodes.size());
  for (std::size_t k = 0; k < nodes.size(); ++k)
    shifts[k] = nodes[k].s;

  // Offload batched complex resolvent solves to numerics: (s_k I - R)^{-1} v
  auto batch_solutions = num::resolvent_solve_batch(shifts, R, v);

  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const auto& sol = batch_solutions[k];
    num::cplx w = nodes[k].w;
    for (num::idx i = 0; i < n; ++i) {
      num::cplx contrib = w * sol[i];
      result[i] += contrib.real();
    }
  }
  return result;
}

} // namespace markovkit::else_solver
