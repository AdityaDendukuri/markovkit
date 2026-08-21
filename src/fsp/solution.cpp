#include "fsp/solution.hpp"
#include "markovkit/statistics.hpp"

namespace fsp {

std::vector<std::vector<double>> FSPSolution::mean_trajectory() const {
  // Collapse each snapshot to species means.
  std::vector<std::vector<double>> result;
  result.reserve(snapshots.size());

  for (const auto& snap : snapshots) {
    result.push_back(markovkit::species_means(snap.states, snap.probs));
  }
  return result;
}

std::map<int, double> FSPSolution::marginal(int snapshot_idx, int species) const {
  const auto& snap = snapshots.at(snapshot_idx);
  return markovkit::marginal_distribution(
      snap.states, snap.probs, static_cast<std::size_t>(species));
}

} // namespace fsp
