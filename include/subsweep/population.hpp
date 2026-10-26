// Subnetwork sweeping with retained walker counts.
#pragma once

#include "subsweep/exit.hpp"
#include "subsweep/laplace.hpp"
#include "subsweep/selection.hpp"
#include "subsweep/sweep.hpp"
#include <algorithm>
#include <optional>
#include <utility>

namespace subsweep {

template <typename State>
struct population_sweep_result {
    array<sweep_transform> transforms;
    array<array<idx>> global_state;
    array<State> states;
    idx initial_row = 0;

    [[nodiscard]] idx sweeps() const { return transforms.size(); }
};

template <typename State, detail::oracle<State> Rates, typename SelectD, typename ComputeZ>
[[nodiscard]] population_sweep_result<State>
sweep_population(const Rates &rates, const State &initial, idx sweeps, SelectD select_D,
                 ComputeZ compute_Z, idx walkers, real negligible_mass, unsigned seed) {
    if (sweeps == 0) {
        throw std::invalid_argument("population subsweep needs a positive sweep count");
    }
    if (walkers == 0) {
        throw std::invalid_argument("population subsweep needs a positive walker count");
    }
    population_sweep_result<State> sequence;
    unordered_map<State, idx> global_row;
    auto global_index = [&](const State &state) {
        auto [entry, added] = global_row.emplace(state, sequence.states.size());
        if (added) {
            sequence.states.push_back(state);
        }
        return entry->second;
    };

    num::rng random(seed);
    unordered_map<State, real> rho{{initial, 1.0}};
    // empirical resampling weight for each entrance to the current sweep
    unordered_map<State, real> resampling_weight{{initial, 1.0}};
    using factor = factor_of<ComputeZ, State>;
    // D and u are empty, and Z is absent, before the first sweep
    subnetwork<State> D;
    std::optional<factor> Z;
    num::vec<real> u;

    for (idx sweep = 0; sweep < sweeps; ++sweep) {
        // The entrances are the sampled boundary law of the preceding sweep.
        array<std::pair<State, real>> entrances;
        real total = 0.0;
        for (const auto &[state, mass] : rho) {
            entrances.push_back({state, mass});
            total += mass;
        }
        if (!(total > negligible_mass)) {
            break;
        }

        // the states of the last D, whose transitions into entering states form T
        array<State> previous(D.states().begin(), D.states().end());
        const subnetwork<State> *previous_D = D.size() ? &D : nullptr;
        const num::vec<real> *previous_u = u.size() ? &u : nullptr;
        update_subnetwork(D, select_D(entrances, previous_D, previous_u), rates);
        num::vec<real> rho_D(D.size(), 0.0);
        for (const auto &[x, weight] : entrances) {
            auto i = D.find(x);
            if (!i) {
                throw std::invalid_argument("each entrance must belong to the subnetwork");
            }
            rho_D[*i] += weight / total;
        }

        // the global index of each state of D, for the reconstruction
        array<idx> global;
        for (idx i = 0; i < D.size(); ++i) {
            global.push_back(global_index(D.state(i)));
        }

        if (!previous.empty()) {
            array<idx> rows, columns;
            array<real> values;
            for (idx i = 0; i < previous.size(); ++i) {
                for (auto [j, rate] : detail::outgoing(rates, previous[i], D)) {
                    auto weight = resampling_weight.find(D.state(j));
                    if (rate > 0.0 && weight != resampling_weight.end()) {
                        rows.push_back(j);
                        columns.push_back(i);
                        values.push_back(rate * weight->second);
                    }
                }
            }
            sequence.transforms.back().T =
                num::spmat::from_triplets(D.size(), previous.size(), rows, columns, values);
        } else {
            sequence.initial_row = *D.find(initial);
        }

        sequence.transforms.push_back({D.matrix()});
        sequence.global_state.push_back(std::move(global));

        update_factor(Z, D, compute_Z);
        u = occupation_times(*Z, rho_D);
        num::vec<real> w = D.exit_rates();
        num::vec<real> last_interior_probability = last_interior_law(u, w);
        array<idx> last_interior =
            sample_last_interior_counts(last_interior_probability, walkers, random);
        unordered_map<State, idx> first_exterior =
            sample_first_exterior_counts(rates, D, last_interior, random);

        // sampled_mass / exact_mass carries the empirical population at every Laplace node
        real exact_total = 0.0;
        for (idx j = 0; j < D.size(); ++j) {
            exact_total += u[j] * w[j];
        }
        rho.clear();
        resampling_weight.clear();
        for (auto &[state, count] : first_exterior) {
            real sampled_mass = static_cast<real>(count) / static_cast<real>(walkers);
            real exact_mass = 0.0;
            for (auto [j, rate] : detail::incoming(rates, state, D)) {
                exact_mass += u[j] * rate;
            }
            if (!(exact_mass > 0.0)) {
                throw std::runtime_error("a sampled entrance has zero exact exit mass");
            }
            resampling_weight.emplace(state, sampled_mass * exact_total / exact_mass);
            rho.emplace(std::move(state), sampled_mass);
        }
    }
    return sequence;
}

// Retain walker counts through `sweeps` successive subnetworks.
template <typename State, detail::oracle<State> Rates>
[[nodiscard]] population_sweep_result<State>
subsweep_population(const Rates &rates, const State &initial, idx sweeps,
                    const sweep_settings &settings = {}) {
    return with_regime<State, Rates>(settings.regime, rates, [&](auto compute_Z) {
        return sweep_population(rates, initial, sweeps,
                                capacity_selection<State>(rates, settings, false), compute_Z,
                                settings.walker_count(), settings.negligible_mass, settings.seed);
    });
}

// Begin with a supplied finite subnetwork, then select later subnetworks from
// the sampled entrances using `settings`.
template <typename State, detail::oracle<State> Rates>
[[nodiscard]] population_sweep_result<State>
subsweep_population(const Rates &rates, const State &initial, idx sweeps,
                    array<State> initial_subnetwork, const sweep_settings &settings = {}) {
    auto select_D = [select = capacity_selection<State>(rates, settings, false),
                     initial_subnetwork = std::move(initial_subnetwork), first = true](
                        const array<std::pair<State, real>> &entrances,
                        const subnetwork<State> *previous, const num::vec<real> *u) mutable {
        if (first) {
            first = false;
            return std::move(initial_subnetwork);
        }
        return select(entrances, previous, u);
    };
    return with_regime<State, Rates>(settings.regime, rates, [&](auto compute_Z) {
        return sweep_population(rates, initial, sweeps, select_D, compute_Z,
                                settings.walker_count(), settings.negligible_mass, settings.seed);
    });
}

// the density at the contour `nodes`
template <typename State>
[[nodiscard]] distribution<State>
reconstruct_density(const population_sweep_result<State> &sequence,
                    view<const num::contour_node> nodes, bool normalize = true) {
    array<num::vec<real>> local = inverse_laplace(sequence.transforms, sequence.initial_row, nodes);
    // sum each state's contributions over the sweeps, then clip and normalize once
    distribution<State> result{sequence.states, num::vec<real>(sequence.states.size(), 0.0)};
    for (idx sweep = 0; sweep < local.size(); ++sweep) {
        for (idx i = 0; i < local[sweep].size(); ++i) {
            result.probability[sequence.global_state[sweep][i]] += local[sweep][i];
        }
    }
    clip_density(result.probability, normalize);
    return result;
}

} // namespace subsweep
