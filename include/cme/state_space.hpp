#pragma once
#include "cme/boundary.hpp"
#include "cme/reaction_system.hpp"
#include "core/types.hpp"
#include "stats/selection.hpp"
#include "stochastic/rng.hpp"
#include <cstddef>
#include <random>
#include <utility>

namespace cme {

/// Active states, aligned probabilities, and state-to-local-index lookup.
struct state_space {
    num::array<markovkit::state> states;
    num::array<double> probs;
    num::unordered_map<markovkit::state, int, markovkit::state_hash> index;

    [[nodiscard]] int size() const { return static_cast<int>(states.size()); }
    [[nodiscard]] bool empty() const { return states.empty(); }

    // Position of state, or -1 if absent
    [[nodiscard]] int find(const markovkit::state &x) const {
        auto it = index.find(x);
        return it != index.end() ? it->second : -1;
    }

    // Add state if absent
    void add_state(const markovkit::state &x, double prob = 0.0);

    // Remove states flagged true in remove_mask
    void remove_states(const num::array<bool> &remove_mask);

    // Add `depth` shells of stoichiometric neighbors inside the boundary.
    void expand(const reaction_system &model, const boundary_condition &bc, int depth = 1);

    // Add one propensity-weighted neighbor per state and shell.
    void expand_ssa(const reaction_system &model, const num::array<double> &rates, double t,
                    const boundary_condition &bc, int depth, num::rng &rng);

    // Prune low-probability candidates whose outgoing flux is negligible.
    // An empty `precomputed_exit_rates` recomputes total propensities locally.
    // Returns (total outgoing flux, number removed).
    std::pair<double, int> compress(const reaction_system &model, const num::array<double> &rates,
                                    double t, double prob_quantile, double flux_tolerance = 1e-6,
                                    num::view<const double> precomputed_exit_rates = {});

    // Normalize probs to sum to 1.0; returns previous sum
    double renormalize();
};

} // namespace cme
