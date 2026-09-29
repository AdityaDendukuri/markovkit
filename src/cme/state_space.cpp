#include "cme/state_space.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace cme {

void state_space::add_state(const markovkit::state &x, double prob) {
    // Keep state storage and lookup indices synchronized.
    if (index.count(x)) {
        return;
    }
    int i = static_cast<int>(states.size());
    states.push_back(x);
    probs.push_back(prob);
    index[x] = i;
}

void state_space::remove_states(const num::array<bool> &remove_mask) {
    if (std::none_of(remove_mask.begin(), remove_mask.end(), [](bool b) { return b; })) {
        return;
    }
    int n = static_cast<int>(states.size());
    num::array<markovkit::state> new_states;
    num::array<double> new_probs;
    new_states.reserve(n);
    new_probs.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!remove_mask[i]) {
            new_states.push_back(states[i]);
            new_probs.push_back(probs[i]);
        }
    }
    states = std::move(new_states);
    probs = std::move(new_probs);
    // Local indices change after compaction, so rebuild the lookup once.
    index.clear();
    for (int i = 0; i < static_cast<int>(states.size()); ++i) {
        index[states[i]] = i;
    }
}

void state_space::expand(const reaction_system &model, const boundary_condition &bc, int depth) {
    for (int d = 0; d < depth; ++d) {
        // Only expand states present at the start of this shell.
        int n_before = static_cast<int>(states.size());
        for (int i = 0; i < n_before; ++i) {
            markovkit::state x = states[i];
            for (const markovkit::state &nu : model.changes) {
                markovkit::state y = x + nu;
                if (!index.count(y) && bc(y)) {
                    add_state(y, 0.0);
                }
            }
        }
    }
}

void state_space::expand_ssa(const reaction_system &model, const num::array<double> &rates,
                             double t, const boundary_condition &bc, int depth, std::mt19937 &rng) {
    // Sample one likely successor per state and shell.
    num::vec props(model.size());
    for (int d = 0; d < depth; ++d) {
        int n_before = static_cast<int>(states.size());
        for (int i = 0; i < n_before; ++i) {
            markovkit::state x = states[i];
            const double total = model.evaluate_propensities(x, rates, t, props);
            if (total <= 0.0) {
                continue;
            }

            const auto chosen = static_cast<int>(num::sample_categorical(props, rng));

            markovkit::state y = x + model.changes[chosen];
            if (!index.count(y) && bc(y)) {
                add_state(y, 0.0);
            }
        }
    }
}

std::pair<double, int> state_space::compress(const reaction_system &model,
                                             const num::array<double> &rates, double t,
                                             double prob_quantile, double flux_tolerance,
                                             num::view<const double> precomputed_exit_rates) {
    int n = size();
    if (n <= 1) {
        return {0.0, 0};
    }

    num::vec local_rates;
    num::view<const double> exit_rates = precomputed_exit_rates;
    if (exit_rates.empty()) {
        // Compute activity locally when the generator did not provide it.
        local_rates = num::vec(static_cast<num::idx>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            local_rates[i] = model.total_propensity(states[i], rates, t);
        }
        exit_rates = local_rates;
    }

    num::vec flux(static_cast<num::idx>(n), 0.0);
    // Estimate the activity carried by each state.
    double total_flux = 0.0;
    for (int i = 0; i < n; ++i) {
        flux[i] = probs[i] * exit_rates[i];
        total_flux += flux[i];
    }

    int n_candidates = std::min(static_cast<int>(std::round(prob_quantile * n)), n - 1);
    if (n_candidates <= 0) {
        return {total_flux, 0};
    }

    const auto candidate_idx = num::smallest_indices(num::view<const double>(probs), n_candidates);

    // Probability selects candidates; flux decides whether removal is safe.
    double threshold = flux_tolerance * total_flux;
    num::array<bool> remove_mask(n, false);
    int n_removed = 0;
    for (int i = 0; i < n_candidates; ++i) {
        const int idx = static_cast<int>(candidate_idx[i]);
        if (flux[idx] < threshold) {
            remove_mask[idx] = true;
            ++n_removed;
        }
    }

    if (n_removed > 0) {
        remove_states(remove_mask);
    }
    return {total_flux, n_removed};
}

double state_space::renormalize() {
    // Clip propagation roundoff before restoring unit mass.
    return num::clip_and_normalize_nonnegative(probs);
}

} // namespace cme
