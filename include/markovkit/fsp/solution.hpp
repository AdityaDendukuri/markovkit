/// @file solution.hpp
/// @brief FSPSolution: snapshots and post-processing for CME solutions.
#pragma once

#include "markovkit/fsp/state_space.hpp"
#include <map>
#include <vector>

namespace fsp {

/// @brief One snapshot of the probability distribution at time t.
struct FSPSnapshot {
  std::vector<MultiIndex> states;
  std::vector<double> probs;
};

/// @brief Collection of snapshots from an FSP solve.
struct FSPSolution {
  std::vector<double> t;              ///< snapshot times
  std::vector<FSPSnapshot> snapshots; ///< (states, probs) at each time
  std::vector<int> state_space_sizes; ///< |S| at each snapshot
  int n_species;                      ///< number of species

  int n_snapshots() const { return (int)t.size(); }

  /// @brief Compute mean state at each snapshot.
  ///
  /// Returns an n_snapshots x n_species matrix stored row-major as a
  /// vector<vector<double>>.
  std::vector<std::vector<double>> mean_trajectory() const;

  /// @brief Compute marginal distribution of a single species at snapshot i.
  ///
  /// Returns sorted (values, probs) pairs.
  std::map<int, double> marginal(int snapshot_idx, int species) const;
};

} // namespace fsp
