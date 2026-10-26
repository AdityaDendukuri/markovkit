#pragma once
#include "container/matrix.hpp"
#include "core/types.hpp"
#include "markovkit/state.hpp"

namespace fsp {

/// Probability values and the states to which they correspond.
struct fsp_snapshot {
    num::array<markovkit::state> states;
    num::vec<double> probs;
};

/// Saved finite-state probability distributions over time.
struct fsp_solution {
    num::array<double> times;
    num::array<fsp_snapshot> snapshots;
    num::array<int> state_space_sizes;
    int n_species = 0;

    [[nodiscard]] int n_snapshots() const { return static_cast<int>(times.size()); }

    // Computes species means per snapshot: (n_snapshots x n_species)
    [[nodiscard]] num::mat<double> mean_trajectory() const;

    // Marginal distribution of one species at given snapshot index
    [[nodiscard]] num::map<int, double> marginal(int snapshot_idx, int species) const;
};

} // namespace fsp
