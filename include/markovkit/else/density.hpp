/// @file else/density.hpp
/// @brief FESSDensityEvolution: continuous probability density propagation over
/// subnetworks.
#pragma once

#include "markovkit/else/network.hpp"
#include "markovkit/else/options.hpp"
#include "markovkit/else/subnetwork.hpp"
#include "markovkit/fsp/detail/flat_hash_map.hpp"
#include <vector>

namespace markovkit::else_solver {

struct FESSDensityStep {
  fsp::FlatHashMap<fsp::MultiIndex, double> density;
  double expected_time_increment = 0.0;
  double active_mass = 1.0;
  int network_size = 0;
  FESSSheddingStats shedding;
};

struct FESSDensityEvolution {
  std::vector<fsp::FlatHashMap<fsp::MultiIndex, double>> densities;
  std::vector<double> expected_times;
  std::vector<int> support_sizes;
  std::vector<int> network_sizes;
  double wall_time = 0.0;
};

/// @brief Compute density evolution P(x, t) across steps over subnetwork
/// generator solves.
FESSDensityEvolution fess_density_evolution(
    const fsp::FlatHashMap<fsp::MultiIndex, double> &initial_density,
    const fsp::ReactionSystem &model, const std::vector<double> &rates,
    int steps = 10, FESSOptions options = {});

} // namespace markovkit::else_solver
