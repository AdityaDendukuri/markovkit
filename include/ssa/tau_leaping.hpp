#pragma once
#include "ssa/gillespie.hpp"

namespace ssa {

struct LeapOptions {
  double fixed_step = 0.0;
  double epsilon = 0.03;
  double min_step = 1e-6;
};

// Adaptive Poisson tau-leaping simulation
Trajectory tau_leaping(const fsp::ReactionSystem& model,
                       const std::vector<double>& rates,
                       const MultiIndex& initial,
                       double t0,
                       double tf,
                       LeapOptions options = {},
                       int seed = 42);

} // namespace ssa
