// Reversible subnetworks drawn from a symmetric graph Laplacian.
#pragma once

#include "else/algorithms/selection.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include "else/quantities/law.hpp"
#include "else/restriction/restriction.hpp"
#include "stochastic/categorical.hpp"
#include "structures/containers/indexed_priority_queue.hpp"
#include <functional>
#include <stdexcept>

namespace else_sim {

// A connected neighborhood of `origin`, grown in descending stationary
// weight. Growth stops at `capacity` states or when the frontier is
// exhausted, so the result is always connected and contains `origin`.
template <typename SparseMat>
[[nodiscard]] array<idx> laplacian_neighborhood(const SparseMat &laplacian,
                                                view<const real> stationary_sqrt, idx origin,
                                                idx capacity) {
    const idx n = laplacian.n_rows();
    if (origin >= n || capacity == 0 || capacity > n) {
        throw std::invalid_argument("invalid Laplacian neighborhood request");
    }
    if (stationary_sqrt.size() != n) {
        throw std::invalid_argument("stationary weights must have one entry per state");
    }

    array<bool> selected(n, false);
    num::indexed_priority_queue<real, idx, std::greater<real>> frontier(n);
    array<idx> states;
    states.reserve(capacity);

    const auto expose = [&](idx state) {
        for (auto k = laplacian.row_ptr()[state]; k < laplacian.row_ptr()[state + 1]; ++k) {
            const idx neighbor = static_cast<idx>(laplacian.col_idx()[k]);
            if (neighbor != state && !selected[neighbor] && !frontier.contains(neighbor)) {
                frontier.push(neighbor, stationary_sqrt[neighbor]);
            }
        }
    };

    selected[origin] = true;
    states.push_back(origin);
    expose(origin);

    while (states.size() < capacity && !frontier.empty()) {
        const idx state = frontier.top_index();
        frontier.pop();
        selected[state] = true;
        states.push_back(state);
        expose(state);
    }
    return states;
}

namespace detail {

// Local labels, global->local index map, and restricted stationary weights
// for `states`.
template <typename State>
struct local_states {
    unordered_map<idx, idx> local;
    array<State> labels;
    num::vec stationary;
};

template <typename State>
[[nodiscard]] local_states<State> localize(view<const real> stationary_sqrt,
                                           const array<idx> &states) {
    local_states<State> result;
    result.local.reserve(states.size());
    result.labels.reserve(states.size());
    result.stationary = num::vec(states.size());
    for (idx i = 0; i < states.size(); ++i) {
        result.local.emplace(states[i], i);
        result.labels.push_back(State{static_cast<int>(states[i])});
        result.stationary[i] = stationary_sqrt[states[i]] * stationary_sqrt[states[i]];
    }
    return result;
}

// The generator triplets and boundary transitions for `states`, restricting
// the Laplacian's row a to R(a,b) = -L(a,b) h_b/h_a and folding rows outside
// `states` into boundary transitions.
template <typename SparseMat, typename State>
[[nodiscard]] generator_build<State>
build_laplacian_generator(const SparseMat &laplacian, view<const real> stationary_sqrt,
                          const array<idx> &states, const unordered_map<idx, idx> &local) {
    generator_build<State> built;
    for (idx a = 0; a < states.size(); ++a) {
        const idx global_a = states[a];
        for (auto k = laplacian.row_ptr()[global_a]; k < laplacian.row_ptr()[global_a + 1]; ++k) {
            const idx global_b = static_cast<idx>(laplacian.col_idx()[k]);
            const real entry = static_cast<real>(laplacian.values()[k]);

            if (global_b == global_a) {
                built.rows.push_back(a);
                built.columns.push_back(a);
                built.values.push_back(-entry);
                continue;
            }

            const real rate = -entry * stationary_sqrt[global_b] / stationary_sqrt[global_a];
            if (rate <= 0.0) {
                continue;
            }

            const auto found = local.find(global_b);
            if (found == local.end()) {
                built.boundary.push_back({a, State{static_cast<int>(global_b)}, rate});
            } else {
                built.rows.push_back(a);
                built.columns.push_back(found->second);
                built.values.push_back(rate);
            }
        }
    }
    return built;
}

} // namespace detail

// Restrict a symmetric Laplacian to `states` as R = -H^-1 L H, with
// H = diag(sqrt(pi)) and detailed balance holding at pi = h^2. States are
// labelled by global index, so this composes with `make_density_chain` the
// same way a reaction-system restriction does.
template <typename SparseMat, typename State = num::multi_index>
[[nodiscard]] subnetwork<State> laplacian_subnetwork(const SparseMat &laplacian,
                                                     view<const real> stationary_sqrt,
                                                     const array<idx> &states) {
    detail::local_states<State> local_states = detail::localize<State>(stationary_sqrt, states);
    detail::generator_build<State> built = detail::build_laplacian_generator<SparseMat, State>(
        laplacian, stationary_sqrt, states, local_states.local);

    auto generator = num::spmat::from_triplets(states.size(), states.size(), built.rows,
                                               built.columns, built.values);
    return make_subnetwork(std::move(local_states.labels), std::move(generator),
                           std::move(built.boundary), std::move(local_states.stationary));
}

// Successive restrictions, each grown from the boundary of the last.
// Suitable as the input to `make_density_chain`.
template <typename SparseMat, typename State = num::multi_index>
[[nodiscard]] array<subnetwork<State>>
laplacian_restrictions(const SparseMat &laplacian, view<const real> stationary_sqrt, idx origin,
                       idx capacity, int count, unsigned seed = 42,
                       density_resampling resampling = density_resampling::weighted_support,
                       idx particles = 0, array<unordered_map<State, real>> *entrance_scales = nullptr) {
    if (count < 1) {
        throw std::invalid_argument("Laplacian restrictions require count >= 1");
    }

    array<idx> window = laplacian_neighborhood(laplacian, stationary_sqrt, origin, capacity);
    unordered_map<State, real> arrival{{State{static_cast<int>(origin)}, 1.0}};
    num::rng random(seed);
    ensemble_options resampling_options;
    resampling_options.capacity = capacity;
    resampling_options.resampling = resampling;
    resampling_options.density_particles = particles;

    array<subnetwork<State>> chain;
    chain.reserve(static_cast<idx>(count));
    unordered_map<State, real> entrance_scale{{State{static_cast<int>(origin)}, 1.0}};
    if (entrance_scales) {
        entrance_scales->clear();
        entrance_scales->reserve(static_cast<idx>(count));
    }

    for (int step = 0; step < count; ++step) {
        chain.push_back(laplacian_subnetwork<SparseMat, State>(laplacian, stationary_sqrt, window));
        if (entrance_scales)
            entrance_scales->push_back(entrance_scale);
        if (step + 1 == count || chain.back().boundary.empty()) {
            break;
        }

        real mass = 0.0;
        for (const auto &[state, weight] : arrival)
            mass += weight;
        num::vec entrance(size(chain.back()), 0.0);
        for (const auto &[state, weight] : arrival) {
            const idx position = find(chain.back(), state);
            if (position < size(chain.back()))
                entrance[position] = weight / mass;
        }
        const num::vec occupation = solve_transpose(chain.back(), entrance);

        unordered_map<State, real> next_arrival;
        for (const auto &transition : chain.back().boundary) {
            next_arrival[transition.destination] += transition.rate * occupation[transition.source];
        }
        if (next_arrival.empty()) {
            break;
        }
        arrival = resample_density_support(next_arrival, resampling_options, random);
        entrance_scale.clear();
        real exact_total = 0.0;
        real sampled_total = 0.0;
        for (const auto &[state, weight] : next_arrival) {
            exact_total += weight;
        }
        for (const auto &[state, weight] : arrival) {
            sampled_total += weight;
        }
        for (const auto &[state, sampled_weight] : arrival) {
            const real exact_weight = next_arrival.at(state);
            if (exact_weight > 0.0 && sampled_total > 0.0)
                entrance_scale[state] =
                    (sampled_weight / sampled_total) / (exact_weight / exact_total);
        }
        window.clear();
        window.reserve(arrival.size());
        for (const auto &[state, weight] : arrival)
            window.push_back(static_cast<idx>(state[0]));
    }
    return chain;
}

// One ELSE trajectory over a reversible Laplacian. Rebuilds the neighborhood
// around the current state at every macrostep, so the window follows the
// walk.
template <typename ReversibleLaplacian, typename State = num::multi_index>
[[nodiscard]] trajectory<State>
laplacian_else_trajectory(const ReversibleLaplacian &generator, idx origin, real initial_time,
                          real final_time, ensemble_options options = {}, unsigned seed = 42) {
    const auto &laplacian = generator.matrix();
    const view<const real> stationary_sqrt(generator.stationary_sqrt());
    if (origin >= laplacian.n_rows() || initial_time > final_time) {
        throw std::invalid_argument("invalid Laplacian trajectory request");
    }

    num::rng random(seed);
    trajectory<State> trajectory;
    trajectory.times.push_back(initial_time);
    trajectory.states.push_back(State{static_cast<int>(origin)});

    idx current = origin;
    real time = initial_time;

    for (idx step = 0; step < options.maximum_steps && time < final_time; ++step) {
        const auto window =
            laplacian_neighborhood(laplacian, stationary_sqrt, current, options.capacity);
        const auto subnetwork =
            laplacian_subnetwork<decltype(laplacian), State>(laplacian, stationary_sqrt, window);
        if (subnetwork.escape_states.empty()) {
            break;
        }

        const idx entrance = find(subnetwork, State{static_cast<int>(current)});
        const array<idx> entrances{entrance};
        const entrance_law_result law = entrance_law(subnetwork, view<const idx>(entrances));

        const num::vec beta = escape_distribution(subnetwork, law, 0);
        num::categorical_sampler escape{beta.span()};
        const idx escape_state = subnetwork.escape_states[escape(random)];
        const real waiting = conditional_escape_time(law, 0, escape_state);

        array<idx> destinations;
        array<real> rates;
        for (const auto &transition : subnetwork.boundary) {
            if (transition.source == escape_state) {
                destinations.push_back(static_cast<idx>(transition.destination[0]));
                rates.push_back(transition.rate);
            }
        }
        if (!(waiting > 0.0) || destinations.empty() || time + waiting >= final_time) {
            break;
        }

        num::categorical_sampler channel{view<const real>(rates)};
        time += waiting;
        current = destinations[channel(random)];
        trajectory.times.push_back(time);
        trajectory.states.push_back(State{static_cast<int>(current)});
    }

    if (final_time < std::numeric_limits<real>::infinity() &&
        trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.back());
    }
    return trajectory;
}

} // namespace else_sim
