#pragma once
#include "fsp/multi_index.hpp"
#include <map>
#include <vector>

namespace fsp {

// Single snapshot of the probability distribution
struct FSPSnapshot {
  std::vector<MultiIndex> states;
  std::vector<double> probs;
};

// Solution trajectory across time snapshots
struct FSPSolution {
  std::vector<double> t;
  std::vector<FSPSnapshot> snapshots;
  std::vector<int> state_space_sizes;
  int n_species = 0;

  [[nodiscard]] int n_snapshots() const { return static_cast<int>(t.size()); }

  // Computes species means per snapshot: (n_snapshots x n_species)
  [[nodiscard]] std::vector<std::vector<double>> mean_trajectory() const;

  // Marginal distribution of one species at given snapshot index
  [[nodiscard]] std::map<int, double> marginal(int snapshot_idx, int species) const;
};

} // namespace fsp
