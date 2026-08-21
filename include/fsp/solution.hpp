#pragma once
#include "markovkit/state.hpp"
#include <map>
#include <vector>

namespace fsp {

/// Probability values and the states to which they correspond.
struct FSPSnapshot {
  std::vector<markovkit::State> states;
  std::vector<double> probs;
};

/// Saved finite-state probability distributions over time.
struct FSPSolution {
  std::vector<double> times;
  std::vector<FSPSnapshot> snapshots;
  std::vector<int> state_space_sizes;
  int n_species = 0;

  [[nodiscard]] int n_snapshots() const {
    return static_cast<int>(times.size());
  }

  // Computes species means per snapshot: (n_snapshots x n_species)
  [[nodiscard]] std::vector<std::vector<double>> mean_trajectory() const;

  // Marginal distribution of one species at given snapshot index
  [[nodiscard]] std::map<int, double> marginal(int snapshot_idx, int species) const;
};

} // namespace fsp
