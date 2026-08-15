/// @file state_space.hpp
/// @brief StateSpace: active states + probability vector for FSP.
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
#include <utility>

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

    void add_state(const MultiIndex& x, double prob = 0.0);
    void remove_states(const std::vector<bool>& remove_mask);
    void expand(const ReactionSystem& model, const BoundaryCondition& bc, int depth = 1);
    void expand_ssa(const ReactionSystem& model,
                    const std::vector<double>& rates,
                    double t,
                    const BoundaryCondition& bc,
                    int depth,
                    std::mt19937& rng);
    std::pair<double, int> compress(const ReactionSystem& model,
                                    const std::vector<double>& rates,
                                    double t,
                                    double prob_quantile,
                                    double flux_tolerance = 1e-6,
                                    const std::vector<double>* precomputed_exit_rates = nullptr);
    double renormalize();
    std::vector<int> get_global_ids() const;
};

} // namespace fsp
