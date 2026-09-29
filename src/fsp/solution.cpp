#include "fsp/solution.hpp"
#include "markovkit/statistics.hpp"

namespace fsp {

num::mat fsp_solution::mean_trajectory() const {
    // Collapse each snapshot to one row of species means.
    num::mat result(snapshots.size(), static_cast<num::idx>(n_species), 0.0);
    for (num::idx snapshot = 0; snapshot < snapshots.size(); ++snapshot) {
        const auto &snap = snapshots[snapshot];
        const num::vec means = markovkit::species_means(snap.states, snap.probs);
        for (num::idx species = 0; species < means.size(); ++species) {
            result(snapshot, species) = means[species];
        }
    }
    return result;
}

num::map<int, double> fsp_solution::marginal(int snapshot_idx, int species) const {
    const auto &snap = snapshots.at(snapshot_idx);
    return markovkit::marginal_distribution(snap.states, snap.probs,
                                            static_cast<std::size_t>(species));
}

} // namespace fsp
