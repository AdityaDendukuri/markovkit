// Finite ensemble ELSE.
#pragma once

#include "else/algorithms/selection.hpp"
#include "else/core/state_graph.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include "else/quantities/factor_update.hpp"
#include "else/quantities/law.hpp"
#include "else/quantities/shedding.hpp"
#include "else/restriction/restriction.hpp"
#include "stochastic/categorical.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace else_sim {

namespace detail {

// The distinct entrances of the active trajectories with their multiplicities.
template <typename State>
struct entrances {
    array<State> states;   // i_k
    array<idx> counts;     // c_k
    unordered_map<State, idx> row; // i_k -> k
    idx active = 0;        // m_active
};

template <typename State>
struct cached_entrance_law {
    array<State> entrances;
    unordered_map<State, idx> row;
    entrance_law_result law;
};

template <typename State>
[[nodiscard]] idx size(const entrances<State> &entrances) {
    return entrances.states.size();
}

struct heuristic_score_cache {
    array<real> values;
    array<idx> generations;
    idx generation = 0;
};

template <typename State>
[[nodiscard]] entrances<State> group_by_entrance(const array<State> &current,
                                                 const array<bool> &unfinished) {
    entrances<State> entrances;
    for (idx t = 0; t < current.size(); ++t) {
        if (!unfinished[t]) {
            continue;
        }
        ++entrances.active;
        const auto [entry, added] = entrances.row.emplace(current[t], entrances.states.size());
        if (added) {
            entrances.states.push_back(current[t]);
            entrances.counts.push_back(0);
        }
        ++entrances.counts[entry->second];
    }
    return entrances;
}

// rho = sum_k (c_k/m_active) at entrance i_k.
template <typename State>
[[nodiscard]] num::vec entrance_mixture(const subnetwork<State> &subnetwork,
                                        const entrances<State> &entrances) {
    num::vec mixture(size(subnetwork), 0.0);
    for (idx k = 0; k < size(entrances); ++k) {
        const idx position = find(subnetwork, entrances.states[k]);
        if (position < size(subnetwork)) {
            mixture[position] =
                static_cast<real>(entrances.counts[k]) / static_cast<real>(entrances.active);
        }
    }
    return mixture;
}

// Close a trajectory out at `final_time`.
template <typename State>
void finish(trajectory<State> &trajectory, real &time, real final_time) {
    time = final_time;
    if (final_time < std::numeric_limits<real>::infinity() &&
        trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.back());
    }
}

// Grow the workspace around `entrances`, then shed back to `options.capacity`
// if needed. When nothing expanded (`depth == 0`), scores are reused from
// `previous_scores` (set by `record_heuristic_scores` last macrostep) instead
// of a fresh restriction and factorization.
template <typename ReactionSystem, typename Rates, typename State>
void select_active_states(const ReactionSystem &model, const Rates &rates,
                          const entrances<State> &entrances, const ensemble_options &options,
                          const heuristic_score_cache &previous_scores, state_graph<State> &graph,
                          active_slots &active) {
    const idx old_size = size(active);
    const int depth = old_size < options.capacity ? options.expansion_depth : 0;
    expand_workspace(model, rates, entrances.states, depth, options.tolerance, graph, active);

    if (size(active) <= options.capacity) {
        return;
    }

    array<idx> protected_states;
    protected_states.reserve(size(entrances));
    for (const State &entrance : entrances.states) {
        const idx state_id = find(graph, entrance);
        const idx position = find(active, state_id);
        if (position < size(active)) {
            protected_states.push_back(position);
        }
    }

    num::vec scores(size(active), 0.0);
    if (depth == 0 && previous_scores.generation != 0) {
        for (idx i = 0; i < size(active); ++i) {
            const idx state_id = active.state_ids[i];
            if (state_id < previous_scores.generations.size() &&
                previous_scores.generations[state_id] == previous_scores.generation)
                scores[i] = previous_scores.values[state_id];
        }
    } else {
        const subnetwork<State> expanded = restriction(model, rates, graph, active);
        const shedding_state_result state =
            shedding_state(expanded, entrance_mixture(expanded, entrances));
        scores = score_states(expanded, state, options);
    }

    const auto shed =
        lowest_scores(scores.span(), protected_states, size(active) - options.capacity);
    array<bool> remove(size(active), false);
    for (idx slot : shed)
        remove[slot] = true;
    retain(active, remove, old_size);
}

// The subnetwork and entrance law for one macrostep, and whether it should
// become the next reuse base.
template <typename State>
struct subnetwork_resolution {
    subnetwork<State> subnetwork;
    entrance_law_result law;
    std::optional<num::vec> inverse_diagonal;
    std::optional<num::vec> exit_time;
    bool store_as_base = false;
    bool stopped = false; // No escape from any entrance; the ensemble is done.
    idx changed_state_slots = 0;
    bool reuse_attempted = false;
    bool reuse_accepted = false;
    idx block_count = 0;
    idx reused_prefix_blocks = 0;
    idx reused_prefix_states = 0;
    idx reused_entrance_law_rows = 0;
    double restriction_and_factor_seconds = 0.0;
    double factor_update_seconds = 0.0;
    double entrance_law_seconds = 0.0;
};

struct reused_entrance_law {
    entrance_law_result law;
    idx reused_rows = 0;
};

template <typename State>
[[nodiscard]] bool block_law_reuse_is_profitable(const entrances<State> &entrances,
                                                 view<const idx> changed,
                                                 const cached_entrance_law<State> &cached,
                                                 bool inverse_diagonal_already_needed = false) {
    idx matched = 0;
    for (const State &entrance : entrances.states)
        matched += cached.row.contains(entrance) ? 1 : 0;
    const idx update_rank = 2 * changed.size();
    return matched != 0 &&
           (inverse_diagonal_already_needed ? update_rank < 2 * matched : update_rank < matched);
}

// Slots whose state id differs between `base_ids` and `active` (empty if
// their sizes differ).
[[nodiscard]] inline array<idx> changed_slots(const array<idx> &base_ids,
                                              const active_slots &active) {
    array<idx> changed;
    if (base_ids.size() != size(active))
        return changed;
    for (idx i = 0; i < size(active); ++i)
        if (base_ids[i] != active.state_ids[i])
            changed.push_back(i);
    return changed;
}

// Position of each entrance within `subnetwork`.
template <typename State>
[[nodiscard]] array<idx> compute_entrance_rows(const subnetwork<State> &subnetwork,
                                               const entrances<State> &entrances) {
    array<idx> rows(size(entrances));
    for (idx k = 0; k < size(entrances); ++k) {
        rows[k] = find(subnetwork, entrances.states[k]);
        if (rows[k] >= size(subnetwork)) {
            throw std::runtime_error("an entrance was shed from its own subnetwork");
        }
    }
    return rows;
}

// Woodbury-updated entrance law against `base`, or nullopt if its residual
// check fails and the caller should fall back to a fresh factorization.
template <typename State>
[[nodiscard]] std::optional<entrance_law_result>
try_reuse_law(const subnetwork<State> &base, const subnetwork<State> &subnetwork,
              view<const idx> entrance_rows, view<const idx> changed) {
    try {
        const low_rank_delta delta =
            row_column_delta(base.operator_matrix, subnetwork.operator_matrix, changed);
        const auto update = make_factor_update(base, delta.left, delta.right);
        entrance_law_result law = entrance_law(subnetwork, entrance_rows, update);

        num::vec rhs(size(subnetwork), 0.0);
        rhs[entrance_rows.front()] = 1.0;
        num::vec solution(size(subnetwork), 0.0);
        for (idx j = 0; j < size(subnetwork); ++j)
            solution[j] = law.occupation(0, j);
        if (relative_residual(num::transpose(subnetwork.operator_matrix), solution, rhs) > 1e-8)
            return std::nullopt;
        return law;
    } catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

// Reuse cached U and V rows for entrances that persist across a block update.
// New entrances are solved against the suffix-updated factor.  Updating a
// cached rows need two rank-sized transpose solves shared by all such rows, so
// use the update only when it saves right-hand sides overall.
template <typename State>
[[nodiscard]] std::optional<reused_entrance_law> try_reuse_block_law(
    const subnetwork<State> &base, const subnetwork<State> &subnetwork,
    const entrances<State> &entrances, view<const idx> entrance_rows, view<const idx> changed,
    const cached_entrance_law<State> &cached,
    const full_factor_update<else_sim::subnetwork<State>> *prepared_update = nullptr) {
    try {
        array<idx> matched_current, matched_cached, fresh_current, fresh_rows;
        for (idx k = 0; k < size(entrances); ++k) {
            const auto found = cached.row.find(entrances.states[k]);
            if (found == cached.row.end()) {
                fresh_current.push_back(k);
                fresh_rows.push_back(entrance_rows[k]);
            } else {
                matched_current.push_back(k);
                matched_cached.push_back(found->second);
            }
        }
        if (!block_law_reuse_is_profitable(entrances, changed, cached, prepared_update != nullptr))
            return std::nullopt;

        const idx n = size(subnetwork);
        entrance_law_result selected{num::mat(matched_current.size(), n, 0.0),
                                     num::mat(matched_current.size(), n, 0.0)};
        for (idx row = 0; row < matched_current.size(); ++row) {
            const idx cached_row = matched_cached[row];
            for (idx column = 0; column < n; ++column) {
                selected.occupation(row, column) = cached.law.occupation(cached_row, column);
                selected.second_occupation(row, column) =
                    cached.law.second_occupation(cached_row, column);
            }
        }

        entrance_law_result reused;
        if (prepared_update) {
            reused = update_entrance_law(selected, *prepared_update);
        } else {
            const low_rank_delta delta =
                row_column_delta(base.operator_matrix, subnetwork.operator_matrix, changed);
            const auto update = make_entrance_law_update(base, delta.left, delta.right);
            reused = update_entrance_law(selected, update);
        }
        entrance_law_result fresh;
        if (!fresh_rows.empty())
            fresh = entrance_law(subnetwork, view<const idx>(fresh_rows));

        entrance_law_result law{num::mat(size(entrances), n, 0.0),
                                num::mat(size(entrances), n, 0.0)};
        for (idx row = 0; row < matched_current.size(); ++row) {
            const idx destination = matched_current[row];
            for (idx column = 0; column < n; ++column) {
                law.occupation(destination, column) = reused.occupation(row, column);
                law.second_occupation(destination, column) = reused.second_occupation(row, column);
            }
        }
        for (idx row = 0; row < fresh_current.size(); ++row) {
            const idx destination = fresh_current[row];
            for (idx column = 0; column < n; ++column) {
                law.occupation(destination, column) = fresh.occupation(row, column);
                law.second_occupation(destination, column) = fresh.second_occupation(row, column);
            }
        }

        for (idx row : matched_current) {
            num::vec rhs(n, 0.0), occupation(n, 0.0), second(n, 0.0);
            rhs[entrance_rows[row]] = 1.0;
            for (idx column = 0; column < n; ++column) {
                occupation[column] = law.occupation(row, column);
                second[column] = law.second_occupation(row, column);
            }
            if (relative_residual(num::transpose(subnetwork.operator_matrix), occupation, rhs) >
                    1e-8 ||
                relative_residual(num::transpose(subnetwork.operator_matrix), second, occupation) >
                    1e-8)
                return std::nullopt;
        }
        return reused_entrance_law{std::move(law), matched_current.size()};
    } catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

// Restrict the active workspace and compute its entrance law. Block factors
// reuse their unchanged prefix and refactor the affected suffix; dense factors
// use Woodbury for a small row-and-column update. If the subnetwork has no
// escape state, every unfinished trajectory is closed out here.
template <typename ReactionSystem, typename Rates, typename State, typename LevelFunction>
[[nodiscard]] subnetwork_resolution<State> resolve_subnetwork_and_law(
    const ReactionSystem &model, const Rates &rates, const entrances<State> &entrances,
    const state_graph<State> &graph, const active_slots &active, LevelFunction level,
    const ensemble_options &options, const std::optional<subnetwork<State>> &base,
    const array<idx> &base_state_ids, const std::optional<cached_entrance_law<State>> &cached_law,
    const std::optional<num::vec> &base_inverse_diagonal, array<trajectory<State>> &trajectories,
    num::vec &times, const array<bool> &unfinished, real final_time,
    detail::restriction_cache<State> *restriction_cache = nullptr) {
    const array<idx> changed = changed_slots(base_state_ids, active);
    const bool reusable_base =
        options.reuse_factorization && base && base_state_ids.size() == size(active);
    const bool block_suffix = reusable_base && uses_block_factor(*base);
    const idx automatic_reuse_slots = 3;
    const idx maximum_reuse_slots =
        options.maximum_reuse_slots == 0 ? automatic_reuse_slots : options.maximum_reuse_slots;
    const bool reuse = reusable_base && (block_suffix || changed.size() <= maximum_reuse_slots);

    const auto restriction_start = std::chrono::steady_clock::now();
    const auto active_state_values = active_states(graph, active);
    subnetwork<State> subnetwork = (restriction_cache && reusable_base)
                                       ? restriction_cached(model, rates, active_state_values,
                                                            *restriction_cache, level, !reuse)
                                       : restriction(model, rates, graph, active, level, !reuse);
    if (restriction_cache && options.reuse_factorization && !reusable_base &&
        size(active) >= options.capacity)
        *restriction_cache = make_restriction_cache(model, rates, active_state_values);
    double restriction_and_factor_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - restriction_start).count();

    if (subnetwork.escape_states.empty()) {
        for (idx t = 0; t < trajectories.size(); ++t) {
            if (unfinished[t]) {
                finish(trajectories[t], times[t], final_time);
            }
        }
        return {.subnetwork = std::move(subnetwork),
                .law = {},
                .store_as_base = false,
                .stopped = true,
                .changed_state_slots = changed.size(),
                .reuse_attempted = reuse,
                .reuse_accepted = false};
    }

    // Solve UM = E once, then VM = U, for every distinct entrance at once.
    const array<idx> entrance_rows = compute_entrance_rows(subnetwork, entrances);

    entrance_law_result law;
    bool store_as_base = !reuse;
    bool reuse_accepted = false;
    idx reused_entrance_law_rows = 0;
    std::optional<num::vec> current_inverse_diagonal;
    std::optional<num::vec> current_exit_time;
    block_suffix_info suffix_info;
    const auto law_start = std::chrono::steady_clock::now();
    double factor_update_seconds = 0.0;
    if (reuse && !changed.empty()) {
        bool suffix_updated = false;
        if (block_suffix) {
            const auto update_start = std::chrono::steady_clock::now();
            suffix_updated = refactor_block_suffix(*base, subnetwork, changed, &suffix_info);
            factor_update_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start)
                    .count();
        }
        if (suffix_updated) {
            std::optional<full_factor_update<else_sim::subnetwork<State>>> full_update;
            if (options.rule == shedding_rule::exact_cut_time && base_inverse_diagonal) {
                try {
                    const low_rank_delta delta = row_column_delta(
                        base->operator_matrix, subnetwork.operator_matrix, changed);
                    const bool reuse_uv = cached_law && block_law_reuse_is_profitable(
                                                            entrances, changed, *cached_law, true);
                    full_update.emplace(
                        reuse_uv ? make_full_factor_update(*base, delta.left, delta.right)
                                 : make_inverse_diagonal_update(*base, delta.left, delta.right));
                    current_inverse_diagonal.emplace(
                        updated_inverse_diagonal(*full_update, base_inverse_diagonal->span()));
                    current_exit_time.emplace(solve(*full_update, num::vec(size(subnetwork), 1.0)));
                } catch (const std::runtime_error &) {
                    current_inverse_diagonal.reset();
                    full_update.reset();
                }
            }
            if (cached_law) {
                if (auto reused =
                        try_reuse_block_law(*base, subnetwork, entrances, entrance_rows, changed,
                                            *cached_law, full_update ? &*full_update : nullptr)) {
                    law = std::move(reused->law);
                    reused_entrance_law_rows = reused->reused_rows;
                } else {
                    law = entrance_law(subnetwork, view<const idx>(entrance_rows));
                }
            } else {
                law = entrance_law(subnetwork, view<const idx>(entrance_rows));
            }
            store_as_base = true;
            reuse_accepted = true;
        } else if (!block_suffix) {
            if (auto reused = try_reuse_law(*base, subnetwork, entrance_rows, changed)) {
                law = std::move(*reused);
                reuse_accepted = true;
            } else {
                subnetwork = restriction_cached(model, rates, active_state_values,
                                                *restriction_cache, level, true);
                law = entrance_law(subnetwork, view<const idx>(entrance_rows));
                store_as_base = true;
            }
        } else {
            subnetwork = restriction_cached(model, rates, active_state_values, *restriction_cache,
                                            level, true);
            law = entrance_law(subnetwork, view<const idx>(entrance_rows));
            store_as_base = true;
        }
    } else if (reuse) {
        law = entrance_law(*base, view<const idx>(entrance_rows));
        reuse_accepted = true;
    } else {
        law = entrance_law(subnetwork, view<const idx>(entrance_rows));
    }
    if (options.rule != shedding_rule::expected_entries && !current_exit_time) {
        if (reuse && changed.empty()) {
            current_exit_time.emplace(solve(*base, num::vec(size(subnetwork), 1.0)));
        } else if (reuse_accepted && !block_suffix && !store_as_base) {
            const low_rank_delta delta =
                row_column_delta(base->operator_matrix, subnetwork.operator_matrix, changed);
            const auto update = make_inverse_diagonal_update(*base, delta.left, delta.right);
            current_exit_time.emplace(solve(update, num::vec(size(subnetwork), 1.0)));
            if (options.rule == shedding_rule::exact_cut_time && base_inverse_diagonal)
                current_inverse_diagonal.emplace(
                    updated_inverse_diagonal(update, base_inverse_diagonal->span()));
        } else {
            current_exit_time.emplace(solve(subnetwork, num::vec(size(subnetwork), 1.0)));
        }
    }
    if (options.rule == shedding_rule::exact_cut_time && !current_inverse_diagonal) {
        if (reuse && changed.empty() && base_inverse_diagonal)
            current_inverse_diagonal = *base_inverse_diagonal;
        else
            current_inverse_diagonal.emplace(inverse_diagonal(subnetwork));
    }
    const double entrance_law_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - law_start).count() -
        factor_update_seconds;

    return {.subnetwork = std::move(subnetwork),
            .law = std::move(law),
            .inverse_diagonal = std::move(current_inverse_diagonal),
            .exit_time = std::move(current_exit_time),
            .store_as_base = store_as_base,
            .stopped = false,
            .changed_state_slots = changed.size(),
            .reuse_attempted = reuse,
            .reuse_accepted = reuse_accepted,
            .block_count = suffix_info.block_count,
            .reused_prefix_blocks = suffix_info.reused_prefix_blocks,
            .reused_prefix_states = suffix_info.reused_prefix_states,
            .reused_entrance_law_rows = reused_entrance_law_rows,
            .restriction_and_factor_seconds = restriction_and_factor_seconds,
            .factor_update_seconds = factor_update_seconds,
            .entrance_law_seconds = entrance_law_seconds};
}

// Cache a per-state score for the next macrostep's `select_active_states`
// when nothing expands: the diagonal-weighted mixture occupation, minus this
// state's own share of the entrance mixture when it is itself an entrance.
template <typename State>
void record_heuristic_scores(const subnetwork<State> &subnetwork, const entrance_law_result &law,
                             const entrances<State> &entrances, const active_slots &active,
                             const ensemble_options &options,
                             const std::optional<num::vec> &inverse_diagonal,
                             const std::optional<num::vec> &exit_time,
                             heuristic_score_cache &previous_scores) {
    num::vec mixture_occupation(size(subnetwork), 0.0);
    ++previous_scores.generation;
    if (previous_scores.generation == 0) {
        std::fill(previous_scores.generations.begin(), previous_scores.generations.end(), 0);
        ++previous_scores.generation;
    }
    idx required = 0;
    for (idx state_id : active.state_ids)
        required = std::max(required, state_id + 1);
    previous_scores.values.resize(required, 0.0);
    previous_scores.generations.resize(required, 0);
    for (idx k = 0; k < size(entrances); ++k) {
        const real weight = static_cast<real>(entrances.counts[k]) / entrances.active;
        for (idx j = 0; j < size(subnetwork); ++j)
            mixture_occupation[j] += weight * law.occupation(k, j);
    }
    num::vec scores;
    if (options.rule == shedding_rule::expected_entries) {
        scores = expected_entries(
            subnetwork, {entrance_mixture(subnetwork, entrances), mixture_occupation, {}});
    } else {
        if (!exit_time)
            throw std::logic_error("cut-time score requires an exit-time solve");
        shedding_state_result state{entrance_mixture(subnetwork, entrances), mixture_occupation,
                                    *exit_time};
        if (options.rule == shedding_rule::exact_cut_time) {
            if (!inverse_diagonal)
                throw std::logic_error("exact cut-time score requires an inverse diagonal");
            scores = exact_cut_time_losses(state, inverse_diagonal->span());
        } else {
            scores = score_states(subnetwork, state, options);
        }
    }
    for (idx j = 0; j < size(subnetwork); ++j) {
        const idx state_id = active.state_ids[j];
        previous_scores.values[state_id] = scores[j];
        previous_scores.generations[state_id] = previous_scores.generation;
    }
}

} // namespace detail

// Advance `count` trajectories together, sharing one subnetwork per
// macrostep: trajectories are grouped by current state, so the d distinct
// entrances need only one factorization's U = EZ and V = EZ^2.
template <typename ReactionSystem, typename Rates, typename State = num::multi_index,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] array<trajectory<State>>
else_ensemble(const ReactionSystem &model, const Rates &rates, const State &initial, idx count,
              real initial_time, real final_time, ensemble_options options = {}, unsigned seed = 42,
              LevelFunction level = nullptr, ensemble_diagnostics *diagnostics = nullptr) {
    if (count == 0) {
        throw std::invalid_argument("ensemble requires at least one trajectory");
    }
    if (initial_time > final_time) {
        throw std::invalid_argument("initial time must not exceed final time");
    }
    if (options.capacity == 0) {
        throw std::invalid_argument("capacity must be positive");
    }

    array<trajectory<State>> trajectories(count);
    array<State> current(count, initial);
    num::vec times(count, initial_time);
    array<idx> steps(count, 0);
    array<num::rng> generators;
    generators.reserve(count);
    for (idx t = 0; t < count; ++t) {
        trajectories[t].times.push_back(initial_time);
        trajectories[t].states.push_back(initial);
        generators.emplace_back(seed + static_cast<unsigned>(t));
    }

    state_graph<State> graph;
    active_slots active;
    detail::heuristic_score_cache previous_scores;
    std::optional<subnetwork<State>> base;
    array<idx> base_state_ids;
    std::optional<detail::cached_entrance_law<State>> cached_law;
    std::optional<num::vec> base_inverse_diagonal;
    detail::restriction_cache<State> restriction_cache;

    while (true) {
        array<bool> unfinished(count, false);
        for (idx t = 0; t < count; ++t) {
            unfinished[t] = times[t] < final_time && steps[t] < options.maximum_steps;
        }
        const auto entrances = detail::group_by_entrance(current, unfinished);
        if (entrances.active == 0) {
            break;
        }

        const auto selection_start = std::chrono::steady_clock::now();
        detail::select_active_states(model, rates, entrances, options, previous_scores, graph,
                                     active);
        const double selection_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - selection_start)
                .count();

        auto resolution = detail::resolve_subnetwork_and_law(
            model, rates, entrances, graph, active, level, options, base, base_state_ids,
            cached_law, base_inverse_diagonal, trajectories, times, unfinished, final_time,
            &restriction_cache);
        if (resolution.stopped) {
            break;
        }
        subnetwork<State> &subnetwork = resolution.subnetwork;
        const entrance_law_result &law = resolution.law;

        const auto score_start = std::chrono::steady_clock::now();
        detail::record_heuristic_scores(subnetwork, law, entrances, active, options,
                                        resolution.inverse_diagonal, resolution.exit_time,
                                        previous_scores);
        const double score_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - score_start).count();

        const auto escape_setup_start = std::chrono::steady_clock::now();
        array<num::categorical_sampler> escape_sampler;
        escape_sampler.reserve(size(entrances));
        for (idx k = 0; k < size(entrances); ++k) {
            const num::vec beta = escape_distribution(subnetwork, law, k);
            escape_sampler.emplace_back(beta.span());
        }
        const double escape_setup_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - escape_setup_start)
                .count();

        const auto sampling_start = std::chrono::steady_clock::now();
        for (idx t = 0; t < count; ++t) {
            if (!unfinished[t]) {
                continue;
            }
            const idx k = entrances.row.at(current[t]);
            auto &generator = generators[t];

            // Sample the escape state J, then the channel C leaving it.
            const idx escape_position = escape_sampler[k](generator);
            const idx escape_state = subnetwork.escape_states[escape_position];
            const real waiting = conditional_escape_time(law, k, escape_state);
            const auto &destinations = subnetwork.escape_destinations[escape_position];

            if (!(waiting > 0.0) || destinations.empty() || times[t] + waiting >= final_time) {
                detail::finish(trajectories[t], times[t], final_time);
                continue;
            }

            num::categorical_sampler channel{
                view<const real>(subnetwork.escape_channel_rates[escape_position])};
            times[t] += waiting;
            current[t] = destinations[channel(generator)];
            trajectories[t].times.push_back(times[t]);
            trajectories[t].states.push_back(current[t]);
            ++steps[t];
        }
        const double sampling_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - sampling_start)
                .count();

        if (diagnostics) {
            diagnostics->steps.push_back(
                {.active_trajectories = entrances.active,
                 .distinct_entrances = detail::size(entrances),
                 .subnetwork_size = size(subnetwork),
                 .changed_state_slots = resolution.changed_state_slots,
                 .reuse_attempted = resolution.reuse_attempted,
                 .reuse_accepted = resolution.reuse_accepted,
                 .fresh_factorization = !resolution.reuse_accepted,
                 .block_count = resolution.block_count,
                 .reused_prefix_blocks = resolution.reused_prefix_blocks,
                 .reused_prefix_states = resolution.reused_prefix_states,
                 .reused_entrance_law_rows = resolution.reused_entrance_law_rows,
                 .selection_seconds = selection_seconds,
                 .restriction_and_factor_seconds = resolution.restriction_and_factor_seconds,
                 .factor_update_seconds = resolution.factor_update_seconds,
                 .entrance_law_seconds = resolution.entrance_law_seconds,
                 .score_seconds = score_seconds,
                 .escape_setup_seconds = escape_setup_seconds,
                 .sampling_seconds = sampling_seconds});
        }

        if (options.reuse_factorization && size(active) == options.capacity) {
            bool current_is_block_base = false;
            if (resolution.store_as_base) {
                base_state_ids = active.state_ids;
                base.emplace(std::move(subnetwork));
                current_is_block_base = uses_block_factor(*base);
                base_inverse_diagonal = std::move(resolution.inverse_diagonal);
            } else if (resolution.reuse_accepted && resolution.changed_state_slots == 0 && base) {
                current_is_block_base = uses_block_factor(*base);
            }

            if (current_is_block_base) {
                detail::cached_entrance_law<State> next;
                next.entrances = entrances.states;
                for (idx k = 0; k < detail::size(entrances); ++k)
                    next.row.emplace(entrances.states[k], k);
                next.law = std::move(resolution.law);
                cached_law.emplace(std::move(next));
            } else if (resolution.store_as_base) {
                cached_law.reset();
            }
        }
    }
    return trajectories;
}

// One trajectory, the `count == 1` case of `else_ensemble`.
template <typename ReactionSystem, typename Rates, typename State = num::multi_index,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] trajectory<State> else_trajectory(const ReactionSystem &model, const Rates &rates,
                                                const State &initial, real initial_time,
                                                real final_time, ensemble_options options = {},
                                                unsigned seed = 42, LevelFunction level = nullptr) {
    auto ensemble =
        else_ensemble(model, rates, initial, 1, initial_time, final_time, options, seed, level);
    return std::move(ensemble.front());
}

} // namespace else_sim
