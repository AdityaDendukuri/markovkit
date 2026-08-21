#include "fsp/solution.hpp"

namespace fsp {

std::vector<std::vector<double>> FSPSolution::mean_trajectory() const {
  // Collapse each snapshot to species means.
  std::vector<std::vector<double>> result;
  result.reserve(snapshots.size());

  for (const auto& snap : snapshots) {
    std::vector<double> means(n_species, 0.0);
    for (std::size_t i = 0; i < snap.states.size(); ++i) {
      // Weight each state count by its probability.
      for (int s = 0; s < n_species; ++s) {
        means[s] += snap.probs[i] * snap.states[i][s];
      }
    }
    result.push_back(std::move(means));
  }
  return result;
}

std::map<int, double> FSPSolution::marginal(int snapshot_idx, int species) const {
  // Aggregate joint-state mass by one species count.
  std::map<int, double> marg;
  const auto& snap = snapshots.at(snapshot_idx);
  for (std::size_t i = 0; i < snap.states.size(); ++i) {
    marg[snap.states[i][species]] += snap.probs[i];
  }
  return marg;
}

} // namespace fsp
