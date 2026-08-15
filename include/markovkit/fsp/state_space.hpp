/// @file state_space.hpp
/// @brief StateSpace: active states + probability vector for FSP.
///
/// Maintains a compact (indexed) set of active discrete states together with
/// their probabilities.  The interface mirrors the Julia StateSpace struct:
///   - add_state / remove_states / renormalize
///   - expand (stoichiometric neighbors)
///   - compress (flux-aware pruning)
#pragma once

#include "markovkit/fsp/multi_index.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include "markovkit/fsp/boundary.hpp"
#include <vector>
#include <random>
#include "markovkit/fsp/detail/flat_hash_map.hpp"
#include <algorithm>
#include <numeric>
#include <cstddef>

namespace fsp {

/// @brief Compact set of active states with associated probabilities.
struct StateSpace {
    std::vector<MultiIndex>                   states;  ///< active states (compact order)
    std::vector<double>                       probs;   ///< probability of each state
    fsp::FlatHashMap<MultiIndex, int>       index;   ///< state -> compact index (0-based)

    // Global ID support (for incremental generator rebuilds)
    fsp::FlatHashMap<MultiIndex, int>       global_id;
    std::vector<MultiIndex>                   id_to_state;
    int                                       next_id = 0;

    int  size()  const { return (int)states.size(); }
    bool dirty = true; ///< true whenever the state set changed since last build_generator

    /// @brief Add a state with given probability. No-op if already present.
    void add_state(const MultiIndex& x, double prob = 0.0);

    /// @brief Remove states where remove_mask[i] == true. Rebuilds index.
    void remove_states(const std::vector<bool>& remove_mask);

    /// @brief Expand by adding stoichiometric neighbors satisfying bc.
    void expand(const ReactionSystem& model, const BoundaryCondition& bc, int depth = 1);

    /// @brief SSA-guided expansion: for each state, sample one reaction by
    ///        propensity and add its successor if inside bc.
    void expand_ssa(const ReactionSystem& model,
                    const std::vector<double>& rates,
                    double t,
                    const BoundaryCondition& bc,
                    int depth,
                    std::mt19937& rng);

    /// @brief Flux-weighted pruning. Returns (total_flux, n_removed).
    /// @param precomputed_exit_rates  If non-null, used as total exit rate per state
    ///                                instead of re-evaluating propensities.
    std::pair<double, int> compress(const ReactionSystem& model,
                                    const std::vector<double>& rates,
                                    double t,
                                    double prob_quantile,
                                    double flux_tolerance = 1e-6,
                                    const std::vector<double>* precomputed_exit_rates = nullptr);

    /// @brief Normalize probabilities to sum to 1. Returns old sum.
    double renormalize();

    /// @brief Get global IDs of current active states (for incremental rebuild).
    std::vector<int> get_global_ids() const;
};

} // namespace fsp
