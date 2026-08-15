/// @file solution.cpp
/// @brief FSPSolution post-processing.
#include "markovkit/fsp/solution.hpp"

namespace fsp {

std::vector<std::vector<double>> FSPSolution::mean_trajectory() const {
    int nt = n_snapshots();
    std::vector<std::vector<double>> result(nt, std::vector<double>(n_species, 0.0));
    for (int i = 0; i < nt; ++i) {
        const FSPSnapshot& snap = snapshots[i];
        int ns = (int)snap.states.size();
        for (int k = 0; k < ns; ++k) {
            double p = snap.probs[k];
            for (int d = 0; d < n_species; ++d) {
                result[i][d] += snap.states[k][d] * p;
            }
        }
    }
    return result;
}

std::map<int, double> FSPSolution::marginal(int snapshot_idx, int species) const {
    const FSPSnapshot& snap = snapshots[snapshot_idx];
    std::map<int, double> dist;
    for (std::size_t k = 0; k < snap.states.size(); ++k) {
        int v = snap.states[k][species];
        dist[v] += snap.probs[k];
    }
    return dist;
}

} // namespace fsp
