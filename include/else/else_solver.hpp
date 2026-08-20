#pragma once
#include "else/subnetwork.hpp"
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace else_sim {

struct ELSEOptions {
  num::idx capacity = 60;
  int expansion_depth = 1;
  double tolerance = 1e-12;
  double horizon = std::numeric_limits<double>::infinity();
};

struct ELSEStepResult {
  MultiIndex next_state;
  MultiIndex preexit_state;
  double waiting_time = 0.0;
  double mean_waiting_time = 0.0;
  bool terminated = false;
};

struct Trajectory {
  std::vector<double> t;
  std::vector<MultiIndex> states;
};

struct EnsembleResult {
  std::vector<double> t;
  std::vector<std::vector<MultiIndex>> paths;
  std::vector<std::vector<double>> mean_trajectory;

  void save_csv(const std::string& filename) const;
};

struct DensityStep {
  std::vector<MultiIndex> states;
  fsp::FlatHashMap<MultiIndex, int> index;
  num::SparseMatrix generator{0, 0, {}, {}, {0}};
  std::vector<double> exit_rates;
  std::vector<MultiIndex> destinations;
  std::vector<ExitEdge> edges;
  std::map<MultiIndex, double> start;
  double mass = 1.0;
  double elapsed = 0.0;
};

// Advance one ELSE step from state
ELSEStepResult else_step(fsp::StateSpace& workspace,
                         const MultiIndex& state,
                         const fsp::ReactionSystem& model,
                         const std::vector<double>& rates,
                         const ELSEOptions& options,
                         std::mt19937& rng);

// Simulate single ELSE trajectory until tf
Trajectory else_trajectory(const fsp::ReactionSystem& model,
                           const std::vector<double>& rates,
                           const MultiIndex& initial,
                           double t0,
                           double tf,
                           ELSEOptions options = {},
                           int seed = 42);

// Simulate multi-trajectory shared-subnetwork ensemble
EnsembleResult else_ensemble(const fsp::ReactionSystem& model,
                             const std::vector<double>& rates,
                             const MultiIndex& initial,
                             std::size_t num_trajectories,
                             double t0,
                             double tf,
                             ELSEOptions options = {},
                             int seed = 42);

// Advance probability density through subnetworks
std::vector<DensityStep> else_density(const fsp::ReactionSystem& model,
                                      const std::vector<double>& rates,
                                      const MultiIndex& initial,
                                      int steps,
                                      ELSEOptions options = {});

} // namespace else_sim
