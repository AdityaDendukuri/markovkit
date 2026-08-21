#pragma once
#include "ssa/gillespie.hpp"

namespace ssa {

struct LeapOptions {
  double fixed_step = 0.0; // Positive values disable adaptive step selection.
  double epsilon = 0.03;   // Expected total events per adaptive step.
  double min_step = 1e-6;
};

/// Poisson tau-leaping with a simple activity-based adaptive step.
markovkit::Trajectory tau_leaping(const markovkit::ReactionSystem& model,
                                  const std::vector<double>& rates,
                                  const markovkit::State& initial,
                                  double t0,
                                  double tf,
                                  LeapOptions options = {},
                                  int seed = 42);

} // namespace ssa
