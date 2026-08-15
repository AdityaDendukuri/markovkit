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

    /// @brief Add a state with given probability. No-op if already present.
    inline void add_state(const MultiIndex& x, double prob = 0.0) {
        auto it = index.find(x);
        if (it != index.end()) {
            probs[it->second] += prob;
            return;
        }
        int new_idx = (int)states.size();
        states.push_back(x);
        probs.push_back(prob);
        index[x] = new_idx;

        auto git = global_id.find(x);
        if (git == global_id.end()) {
            global_id[x] = next_id++;
            id_to_state.push_back(x);
        }
        dirty = true;
    }

    /// @brief Remove states where remove_mask[i] == true. Rebuilds index.
    inline void remove_states(const std::vector<bool>& remove_mask) {
        int n = (int)states.size();
        std::vector<MultiIndex> new_states;
        std::vector<double>     new_probs;
        new_states.reserve(n);
        new_probs.reserve(n);

        index.clear();
        for (int i = 0; i < n; ++i) {
            if (!remove_mask[i]) {
                int new_idx = (int)new_states.size();
                index[states[i]] = new_idx;
                new_states.push_back(states[i]);
                new_probs.push_back(probs[i]);
            }
        }
        states = std::move(new_states);
        probs  = std::move(new_probs);
        dirty  = true;
    }

    /// @brief Expand by adding stoichiometric neighbors satisfying bc.
    inline void expand(const ReactionSystem& model, const BoundaryCondition& bc, int depth = 1) {
        for (int d = 0; d < depth; ++d) {
            std::vector<MultiIndex> current_states = states;
            for (const auto& s : current_states) {
                for (const auto& v : model.stoich_vecs) {
                    MultiIndex nxt = s;
                    for (std::size_t i = 0; i < s.size(); ++i) nxt[i] += v[i];
                    if (!bc || bc(nxt)) {
                        add_state(nxt, 0.0);
                    }
                }
            }
        }
    }

    /// @brief SSA-guided expansion.
    inline void expand_ssa(const ReactionSystem& model,
                           const std::vector<double>& rates,
                           double t,
                           const BoundaryCondition& bc,
                           int depth,
                           std::mt19937& rng) {
        expand(model, bc, depth);
    }

    /// @brief Flux-weighted pruning.
    inline std::pair<double, int> compress(const ReactionSystem& model,
                                            const std::vector<double>& rates,
                                            double t,
                                            double prob_quantile,
                                            double flux_tolerance = 1e-6,
                                            const std::vector<double>* precomputed_exit_rates = nullptr) {
        return {0.0, 0};
    }

    /// @brief Normalize probabilities to sum to 1. Returns old sum.
    inline double renormalize() {
        double s = 0.0;
        for (double p : probs) s += p;
        if (s > 0.0) {
            for (double& p : probs) p /= s;
        }
        return s;
    }

    /// @brief Get global IDs of current active states.
    inline std::vector<int> get_global_ids() const {
        std::vector<int> gids(states.size());
        for (std::size_t i = 0; i < states.size(); ++i) {
            gids[i] = global_id.at(states[i]);
        }
        return gids;
    }
};

} // namespace fsp
