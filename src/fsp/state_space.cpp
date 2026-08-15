/// @file state_space.cpp
/// @brief StateSpace implementation.
#include "markovkit/fsp/state_space.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace fsp {

void StateSpace::add_state(const MultiIndex& x, double prob) {
    if (index.count(x)) return;  // already present — no change
    dirty = true;
    int i = (int)states.size();
    states.push_back(x);
    probs.push_back(prob);
    index[x] = i;
    if (!global_id.count(x)) {
        global_id[x] = next_id;
        id_to_state.push_back(x);
        ++next_id;
    }
}

void StateSpace::remove_states(const std::vector<bool>& remove_mask) {
    if (std::none_of(remove_mask.begin(), remove_mask.end(), [](bool b){ return b; }))
        return;  // nothing actually removed
    dirty = true;
    int n = (int)states.size();
    std::vector<MultiIndex> new_states;
    std::vector<double>     new_probs;
    new_states.reserve(n);
    new_probs.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!remove_mask[i]) {
            new_states.push_back(states[i]);
            new_probs.push_back(probs[i]);
        }
    }
    states = std::move(new_states);
    probs  = std::move(new_probs);
    // Rebuild compact index
    index.clear();
    for (int i = 0; i < (int)states.size(); ++i)
        index[states[i]] = i;
}

void StateSpace::expand(const ReactionSystem& model, const BoundaryCondition& bc, int depth) {
    for (int d = 0; d < depth; ++d) {
        // Snapshot the count (not the data) — iterate only pre-existing states.
        // Access by index so reallocation of states vector is safe.
        int n_before = (int)states.size();
        for (int i = 0; i < n_before; ++i) {
            MultiIndex x = states[i];  // value copy: safe across push_back reallocs
            for (const MultiIndex& nu : model.stoich_vecs) {
                MultiIndex y = x + nu;
                if (!index.count(y) && bc(y))
                    add_state(y, 0.0);
            }
        }
    }
}

void StateSpace::expand_ssa(const ReactionSystem& model,
                            const std::vector<double>& rates,
                            double t,
                            const BoundaryCondition& bc,
                            int depth,
                            std::mt19937& rng)
{
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    int nk = (int)model.n_reactions();
    std::vector<double> props(nk);  // reuse across all states
    for (int d = 0; d < depth; ++d) {
        int n_before = (int)states.size();
        for (int i = 0; i < n_before; ++i) {
            MultiIndex x = states[i];  // value copy: safe across push_back reallocs
            double total = 0.0;
            for (int k = 0; k < nk; ++k) {
                double a = std::max(0.0, model.propensities[k](x, rates, t));
                props[k] = a;
                total += a;
            }
            if (total <= 0.0) continue;

            double threshold = uniform(rng) * total;
            double csum = 0.0;
            int chosen = -1;
            for (int k = 0; k < nk; ++k) {
                csum += props[k];
                if (threshold <= csum) { chosen = k; break; }
            }
            if (chosen < 0) continue;

            MultiIndex y = x + model.stoich_vecs[chosen];
            if (!index.count(y) && bc(y))
                add_state(y, 0.0);
        }
    }
}

std::pair<double, int> StateSpace::compress(const ReactionSystem& model,
                                             const std::vector<double>& rates,
                                             double t,
                                             double prob_quantile,
                                             double flux_tolerance,
                                             const std::vector<double>* precomputed_exit_rates)
{
    int n = (int)states.size();
    if (n == 0) return {0.0, 0};

    // Compute flux for each state: p[i] * sum_k a_k(x[i])
    std::vector<double> flux(n);
    if (precomputed_exit_rates) {
        for (int i = 0; i < n; ++i)
            flux[i] = probs[i] * (*precomputed_exit_rates)[i];
    } else {
        for (int i = 0; i < n; ++i) {
            double w = 0.0;
            for (auto& prop : model.propensities)
                w += prop(states[i], rates, t);
            flux[i] = probs[i] * w;
        }
    }
    double total_flux = 0.0;
    for (double f : flux) total_flux += f;
    double flux_threshold = total_flux * flux_tolerance;

    // Sort by probability to find quantile candidates
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[a] < probs[b];
    });

    int n_candidates = (int)std::round(std::clamp(prob_quantile, 0.0, 1.0) * n);

    std::vector<bool> remove_mask(n, false);
    int n_removed = 0;
    for (int k = 0; k < n_candidates; ++k) {
        int i = order[k];
        if (flux_tolerance <= 0.0 || flux[i] < flux_threshold) {
            remove_mask[i] = true;
            ++n_removed;
        }
    }

    remove_states(remove_mask);
    return {total_flux, n_removed};
}

double StateSpace::renormalize() {
    double s = 0.0;
    for (double p : probs) s += p;
    if (s > 0.0) {
        for (double& p : probs) p /= s;
    }
    return s;
}

std::vector<int> StateSpace::get_global_ids() const {
    std::vector<int> ids;
    ids.reserve(states.size());
    for (const MultiIndex& x : states) {
        auto it = global_id.find(x);
        ids.push_back(it != global_id.end() ? it->second : -1);
    }
    return ids;
}

} // namespace fsp
