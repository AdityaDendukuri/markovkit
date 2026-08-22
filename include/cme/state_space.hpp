#pragma once
#include "cme/boundary.hpp"
#include "cme/reaction_system.hpp"
#include <cstddef>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cme {

/// Active states, aligned probabilities, and state-to-local-index lookup.
struct StateSpace {
    std::vector<markovkit::State> states;
    std::vector<double> probs;
    std::unordered_map<markovkit::State, int> index;

    [[nodiscard]] int size() const { return static_cast<int>(states.size()); }
    [[nodiscard]] bool empty() const { return states.empty(); }

    // Position of state, or -1 if absent
    [[nodiscard]] int find(const markovkit::State &x) const {
        auto it = index.find(x);
        return it != index.end() ? it->second : -1;
    }

    // Add state if absent
    void add_state(const markovkit::State &x, double prob = 0.0);

    // Remove states flagged true in remove_mask
    void remove_states(const std::vector<bool> &remove_mask);

    // Add `depth` shells of stoichiometric neighbors inside the boundary.
    void expand(const ReactionSystem &model, const BoundaryCondition &bc, int depth = 1);

    // Add one propensity-weighted neighbor per state and shell.
    void expand_ssa(const ReactionSystem &model, const std::vector<double> &rates, double t,
                    const BoundaryCondition &bc, int depth, std::mt19937 &rng);

    // Prune low-probability candidates whose outgoing flux is negligible.
    // Returns (total outgoing flux, number removed).
    std::pair<double, int> compress(const ReactionSystem &model, const std::vector<double> &rates,
                                    double t, double prob_quantile, double flux_tolerance = 1e-6,
                                    const std::vector<double> *precomputed_exit_rates = nullptr);

    // Normalize probs to sum to 1.0; returns previous sum
    double renormalize();
};

} // namespace cme
