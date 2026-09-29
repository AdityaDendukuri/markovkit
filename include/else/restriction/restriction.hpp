// Restrict a reaction system to a finite state set.
#pragma once

#include "else/core/state_graph.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace else_sim {

// Apply a stoichiometric change to a state.
template <typename State, typename Change>
[[nodiscard]] State apply_change(State state, const Change &change) {
    for (idx i = 0; i < change.size(); ++i) {
        state[i] += change[i];
    }
    return state;
}

// Block level of every state, or an empty vector when `level` is `nullptr`.
template <typename State, typename LevelFunction>
[[nodiscard]] array<idx> compute_levels(const array<State> &states, LevelFunction level) {
    if constexpr (std::is_same_v<LevelFunction, std::nullptr_t>) {
        return {};
    } else {
        array<idx> levels(states.size());
        for (idx i = 0; i < states.size(); ++i)
            levels[i] = static_cast<idx>(level(states[i]));
        return levels;
    }
}

namespace detail {

template <typename State>
struct cached_transition {
    State destination{};
    real rate = 0.0;
};

template <typename State>
struct restriction_cache {
    array<State> states;
    unordered_map<State, idx> position;
    array<array<cached_transition<State>>> outgoing;
    unordered_map<State, array<idx>> incoming;
    array<array<real>> row_values;
    array<array<idx>> row_columns;
    array<array<boundary_transition<State>>> row_boundary;
};

template <typename State>
void rebuild_cached_row(restriction_cache<State> &cache, idx source) {
    auto &values = cache.row_values[source];
    auto &columns = cache.row_columns[source];
    auto &boundary = cache.row_boundary[source];
    values.clear();
    columns.clear();
    boundary.clear();

    real total = 0.0;
    for (const auto &transition : cache.outgoing[source]) {
        const auto found = cache.position.find(transition.destination);
        if (found == cache.position.end()) {
            boundary.push_back({source, transition.destination, transition.rate});
        } else {
            columns.push_back(found->second);
            values.push_back(transition.rate);
        }
        total += transition.rate;
    }
    columns.push_back(source);
    values.push_back(-total);
}

inline void erase_source(array<idx> &sources, idx source) {
    sources.erase(std::remove(sources.begin(), sources.end(), source), sources.end());
}

// The generator triplets and boundary transitions for a finite state set.
template <typename State>
struct generator_build {
    array<idx> rows, columns;
    array<real> values;
    array<boundary_transition<State>> boundary;
};

// Row convention: entry (i,j) is the rate from state i to state j; the
// diagonal is the total outgoing rate, so row sums are -(R*1).
template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] generator_build<State>
build_generator(const ReactionSystem &model, const Rates &rates, const array<State> &states) {
    unordered_map<State, idx> position;
    position.reserve(states.size());
    for (idx i = 0; i < states.size(); ++i) {
        position.emplace(states[i], i);
    }

    generator_build<State> built;
    for (idx source = 0; source < states.size(); ++source) {
        real outgoing = 0.0;
        for (idx reaction = 0; reaction < model.changes.size(); ++reaction) {
            const real rate =
                static_cast<real>(model.propensities[reaction](states[source], rates, real(0)));
            if (rate <= 0.0) {
                continue;
            }

            State destination = apply_change(states[source], model.changes[reaction]);
            const auto found = position.find(destination);
            if (found == position.end()) {
                built.boundary.push_back({source, std::move(destination), rate});
            } else {
                built.rows.push_back(source);
                built.columns.push_back(found->second);
                built.values.push_back(rate);
            }
            outgoing += rate;
        }

        built.rows.push_back(source);
        built.columns.push_back(source);
        built.values.push_back(-outgoing);
    }
    return built;
}

} // namespace detail

template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] detail::restriction_cache<State> make_restriction_cache(const ReactionSystem &model,
                                                                      const Rates &rates,
                                                                      const array<State> &states) {
    detail::restriction_cache<State> cache;
    cache.states = states;
    cache.position.reserve(states.size());
    for (idx i = 0; i < states.size(); ++i)
        cache.position.emplace(states[i], i);
    cache.outgoing.resize(states.size());
    cache.row_values.resize(states.size());
    cache.row_columns.resize(states.size());
    cache.row_boundary.resize(states.size());
    for (idx source = 0; source < states.size(); ++source) {
        for (idx reaction = 0; reaction < model.changes.size(); ++reaction) {
            const real rate =
                static_cast<real>(model.propensities[reaction](states[source], rates, real(0)));
            if (rate > 0.0) {
                State destination = apply_change(states[source], model.changes[reaction]);
                cache.incoming[destination].push_back(source);
                cache.outgoing[source].push_back({std::move(destination), rate});
            }
        }
        detail::rebuild_cached_row(cache, source);
    }
    return cache;
}

template <typename ReactionSystem, typename Rates, typename State>
void refresh_restriction_cache(const ReactionSystem &model, const Rates &rates,
                               const array<State> &states,
                               detail::restriction_cache<State> &cache) {
    if (cache.states.size() != states.size() || cache.outgoing.size() != states.size()) {
        cache = make_restriction_cache(model, rates, states);
        return;
    }
    array<idx> changed;
    for (idx source = 0; source < states.size(); ++source)
        if (cache.states[source] != states[source])
            changed.push_back(source);
    if (changed.empty())
        return;

    const array<State> old_states = cache.states;
    for (idx source : changed)
        cache.position.erase(old_states[source]);
    for (idx source : changed)
        cache.position.emplace(states[source], source);

    array<bool> affected(states.size(), false);
    for (idx source : changed) {
        affected[source] = true;
        for (const auto &transition : cache.outgoing[source]) {
            auto found = cache.incoming.find(transition.destination);
            if (found == cache.incoming.end())
                continue;
            detail::erase_source(found->second, source);
            if (found->second.empty())
                cache.incoming.erase(found);
        }
        cache.outgoing[source].clear();
        for (idx reaction = 0; reaction < model.changes.size(); ++reaction) {
            const real rate =
                static_cast<real>(model.propensities[reaction](states[source], rates, real(0)));
            if (rate > 0.0) {
                State destination = apply_change(states[source], model.changes[reaction]);
                cache.incoming[destination].push_back(source);
                cache.outgoing[source].push_back({std::move(destination), rate});
            }
        }
    }

    for (idx source : changed) {
        for (const State *state : {&old_states[source], &states[source]}) {
            const auto found = cache.incoming.find(*state);
            if (found != cache.incoming.end())
                for (idx incoming_source : found->second)
                    affected[incoming_source] = true;
        }
    }
    cache.states = states;
    for (idx source = 0; source < states.size(); ++source)
        if (affected[source])
            detail::rebuild_cached_row(cache, source);
}

template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] subnetwork<State>
restriction_cached(const ReactionSystem &model, const Rates &rates, array<State> states,
                   detail::restriction_cache<State> &cache, LevelFunction level = nullptr,
                   bool factorize = true) {
    refresh_restriction_cache(model, rates, states, cache);
    array<idx> columns, row_ptr(states.size() + 1, 0);
    array<real> values;
    array<boundary_transition<State>> boundary;
    idx matrix_entries = 0;
    idx boundary_entries = 0;
    for (idx source = 0; source < states.size(); ++source) {
        matrix_entries += cache.row_columns[source].size();
        boundary_entries += cache.row_boundary[source].size();
    }
    columns.reserve(matrix_entries);
    values.reserve(matrix_entries);
    boundary.reserve(boundary_entries);
    for (idx source = 0; source < states.size(); ++source) {
        columns.insert(columns.end(), cache.row_columns[source].begin(),
                       cache.row_columns[source].end());
        values.insert(values.end(), cache.row_values[source].begin(),
                      cache.row_values[source].end());
        boundary.insert(boundary.end(), cache.row_boundary[source].begin(),
                        cache.row_boundary[source].end());
        row_ptr[source + 1] = columns.size();
    }
    num::spmat generator(states.size(), states.size(), std::move(values), std::move(columns),
                         std::move(row_ptr));
    const array<idx> levels = compute_levels(states, level);
    return make_subnetwork(std::move(states), std::move(generator), std::move(boundary), {},
                           view<const idx>(levels), factorize);
}

// The principal restriction R = R_bar(S,S) of a reaction system. `model` must
// expose `changes` (stoichiometry per reaction) and `propensities`.
template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] subnetwork<State> restriction(const ReactionSystem &model, const Rates &rates,
                                            array<State> states, LevelFunction level = nullptr,
                                            bool factorize = true) {
    detail::generator_build<State> built = detail::build_generator(model, rates, states);
    auto generator = num::spmat::from_triplets(states.size(), states.size(), built.rows,
                                               built.columns, built.values);
    const array<idx> levels = compute_levels(states, level);
    return make_subnetwork(std::move(states), std::move(generator), std::move(built.boundary), {},
                           view<const idx>(levels), factorize);
}

// Restrict a reversible reaction system and factor its symmetric normalization.
template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] subnetwork<State>
reversible_restriction(const ReactionSystem &model, const Rates &rates, array<State> states,
                       view<const real> stationary_sqrt, LevelFunction level = nullptr) {
    if (stationary_sqrt.size() != states.size())
        throw std::invalid_argument("one stationary weight is required per state");

    subnetwork<State> unfactored = restriction(model, rates, states, nullptr, false);
    num::vec stationary(states.size());
    for (idx i = 0; i < states.size(); ++i)
        stationary[i] = stationary_sqrt[i] * stationary_sqrt[i];

    const array<idx> levels = compute_levels(states, level);
    return make_subnetwork(std::move(states), std::move(unfactored.generator),
                           std::move(unfactored.boundary), std::move(stationary),
                           view<const idx>(levels));
}

template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] subnetwork<State>
restriction(const ReactionSystem &model, const Rates &rates, const state_graph<State> &graph,
            const active_slots &active, LevelFunction level = nullptr, bool factorize = true) {
    return restriction(model, rates, active_states(graph, active), level, factorize);
}

} // namespace else_sim
