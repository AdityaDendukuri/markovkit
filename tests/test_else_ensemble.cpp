/// @file tests/test_else_ensemble.cpp
/// @brief Finite-m ensemble ELSE.
///
/// Two of these checks are analytic rather than comparative. A pure birth
/// process has only one escape route, so its conditional mean escape time is the
/// unconditional one and the macrostep length is known in closed form. And the
/// paper's claim that sharing a subnetwork changes nothing is checked directly:
/// the grouped block solve must reproduce what a solve per entrance would give.

#include "check.hpp"
#include "core/types.hpp"

#include "else/algorithms/ensemble.hpp"
#include "else/core/state_graph.hpp"
#include "else/quantities/law.hpp"
#include "else/restriction/restriction.hpp"

#include <functional>

namespace {

using state = num::multi_index;
using rates = num::array<num::real>;
using num::idx;
using num::real;

/// The minimal reaction-system interface `restriction` and the ensemble require.
struct system {
    num::array<num::array<int>> changes;
    num::array<std::function<real(const state &, const rates &, real)>> propensities;

    [[nodiscard]] real total_propensity(const state &state, const rates &rates, real time) const {
        real total = 0.0;
        for (const auto &propensity : propensities) {
            total += propensity(state, rates, time);
        }
        return total;
    }
};

/// Pure birth at a constant rate: one reaction, one escape route.
system pure_birth() {
    system model;
    model.changes = {{1}};
    model.propensities = {[](const state &, const rates &r, real) { return r[0]; }};
    return model;
}

/// Birth-death with constant birth and linear death.
system birth_death() {
    system model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const state &, const rates &r, real) { return r[0]; },
        [](const state &x, const rates &r, real) { return r[1] * static_cast<real>(x[0]); },
    };
    return model;
}

} // namespace

void test_pure_birth_macrostep_is_analytic() {
    // Depth-1 expansion around a single entrance gives the window {x, x+1}.
    // A pure birth process must traverse both before escaping, so the escape
    // time is the sum of two exponentials and its mean is exactly 2/lambda.
    const real lambda = 2.5;
    const rates rates{lambda};
    const auto model = pure_birth();

    else_sim::ensemble_options options;
    options.capacity = 32;
    options.expansion_depth = 1;
    options.maximum_steps = 4;

    const auto trajectory =
        else_sim::else_trajectory(model, rates, state{0}, 0.0, 1e9, options, 7u);

    check::that(trajectory.times.size() >= 5, "four macrosteps were recorded");
    for (num::idx step = 1; step + 1 < trajectory.times.size(); ++step) {
        check::close(trajectory.times[step] - trajectory.times[step - 1], 2.0 / lambda,
                     "macrostep length is 2/lambda", 1e-12);
        check::that(trajectory.states[step][0] == trajectory.states[step - 1][0] + 2,
                    "each macrostep advances two states");
    }
    check::done("pure-birth macrostep matches its closed form");
}

void test_shared_subnetwork_matches_per_entrance_solves() {
    // The paper's remark: grouping trajectories changes neither the escape
    // distribution nor the conditional mean. Solve the block once for all
    // entrances, then once per entrance, and require agreement.
    const rates rates{1.4, 0.3};
    const auto model = birth_death();

    num::array<state> states;
    for (int copies = 1; copies <= 9; ++copies) {
        states.push_back(state{copies});
    }
    const auto subnetwork = else_sim::restriction(model, rates, states);

    const num::array<idx> entrances{0, 3, 4, 8};
    const auto grouped = else_sim::entrance_law(subnetwork, num::view<const idx>(entrances));

    for (idx k = 0; k < entrances.size(); ++k) {
        const num::array<idx> single{entrances[k]};
        const auto alone = else_sim::entrance_law(subnetwork, num::view<const idx>(single));

        for (idx j = 0; j < else_sim::size(subnetwork); ++j) {
            check::close(grouped.occupation(k, j), alone.occupation(0, j),
                         "grouped U row equals the solitary solve");
            check::close(grouped.second_occupation(k, j), alone.second_occupation(0, j),
                         "grouped V row equals the solitary solve");
        }

        const auto grouped_beta = else_sim::escape_distribution(subnetwork, grouped, k);
        const auto alone_beta = else_sim::escape_distribution(subnetwork, alone, 0);
        for (idx b = 0; b < grouped_beta.size(); ++b) {
            check::close(grouped_beta[b], alone_beta[b], "grouped beta equals the solitary beta");
        }
        for (idx b : subnetwork.escape_states) {
            check::close(else_sim::conditional_escape_time(grouped, k, b),
                         else_sim::conditional_escape_time(alone, 0, b),
                         "grouped mu equals the solitary mu");
        }
    }
    check::done("sharing a subnetwork does not change the entrance laws");
}

void test_trajectories_are_well_formed() {
    const rates rates{1.4, 0.3};
    const auto model = birth_death();

    else_sim::ensemble_options options;
    options.capacity = 12; // small enough that shedding runs
    options.expansion_depth = 1;

    const auto ensemble =
        else_sim::else_ensemble(model, rates, state{4}, 24, 0.0, 6.0, options, 2026u);

    check::that(ensemble.size() == 24, "one trajectory per requested path");
    for (const auto &trajectory : ensemble) {
        check::that(trajectory.times.size() == trajectory.states.size(),
                    "each recorded time has a state");
        check::that(!trajectory.times.empty(), "trajectory is non-empty");
        check::close(trajectory.times.front(), 0.0, "starts at the initial time");
        check::that(trajectory.states.front() == state{4}, "starts at the initial state");
        check::close(trajectory.times.back(), 6.0, "ends at the final time");

        for (num::idx step = 1; step < trajectory.times.size(); ++step) {
            check::that(trajectory.times[step] >= trajectory.times[step - 1],
                        "times are non-decreasing");
            check::that(trajectory.states[step][0] >= 0, "copy numbers stay non-negative");
        }
    }
    check::done("trajectories are well formed under shedding");
}

void test_ensemble_is_reproducible() {
    const rates rates{1.4, 0.3};
    const auto model = birth_death();
    else_sim::ensemble_options options;
    options.capacity = 10;

    const auto first = else_sim::else_ensemble(model, rates, state{3}, 8, 0.0, 4.0, options, 99u);
    const auto again = else_sim::else_ensemble(model, rates, state{3}, 8, 0.0, 4.0, options, 99u);

    for (num::idx t = 0; t < first.size(); ++t) {
        check::that(first[t].times.size() == again[t].times.size(),
                    "same seed gives the same step count");
        for (num::idx step = 0; step < first[t].times.size(); ++step) {
            check::close(first[t].times[step], again[t].times[step], "same seed gives same times");
            check::that(first[t].states[step] == again[t].states[step],
                        "same seed gives same states");
        }
    }
    check::done("the ensemble is reproducible at a fixed seed");
}

void test_incremental_restriction_matches_fresh_rebuilds() {
    const rates rates{1.4, 0.3};
    const auto model = birth_death();
    num::array<state> states;
    for (int copies = 1; copies <= 9; ++copies)
        states.push_back(state{copies});

    auto cache = else_sim::make_restriction_cache(model, rates, states);
    const num::array<num::array<state>> replacements{
        {state{1}, state{2}, state{14}, state{4}, state{5}, state{6}, state{7}, state{8}, state{9}},
        {state{1}, state{2}, state{14}, state{4}, state{11}, state{6}, state{7}, state{8},
         state{9}},
        {state{1}, state{2}, state{3}, state{4}, state{5}, state{6}, state{7}, state{8}, state{9}},
    };

    for (const auto &replacement : replacements) {
        const auto cached =
            else_sim::restriction_cached(model, rates, replacement, cache, nullptr, false);
        const auto fresh = else_sim::restriction(model, rates, replacement, nullptr, false);
        const num::mat cached_generator = num::dense(cached.generator);
        const num::mat fresh_generator = num::dense(fresh.generator);
        for (idx i = 0; i < replacement.size(); ++i) {
            for (idx j = 0; j < replacement.size(); ++j)
                check::close(cached_generator(i, j), fresh_generator(i, j),
                             "cached generator matches a fresh restriction");
            check::close(cached.escape_rates[i], fresh.escape_rates[i],
                         "cached escape rates match a fresh restriction");
        }
        check::that(cached.escape_destinations == fresh.escape_destinations,
                    "cached escape destinations match a fresh restriction");
        check::that(cached.escape_channel_rates == fresh.escape_channel_rates,
                    "cached escape channels match a fresh restriction");
    }
    check::done("incremental restrictions match fresh rebuilds");
}

void test_state_graph_identities_persist() {
    else_sim::state_graph<state> graph;
    const idx first = else_sim::insert(graph, state{2});
    const idx second = else_sim::insert(graph, state{5});

    check::that(first != second, "distinct states get distinct identities");
    check::that(else_sim::insert(graph, state{2}) == first,
                "re-inserting returns the original identity");
    check::that(else_sim::find(graph, state{5}) == second, "lookup finds an inserted state");
    check::that(else_sim::find(graph, state{9}) == else_sim::invalid_index,
                "lookup reports an unknown state");
    check::that(graph.states[first] == state{2}, "identities index back to their state");
    check::that(else_sim::size(graph) == 2, "size counts distinct states");
    check::done("state identities persist across re-entry");
}

void test_exact_cut_time_reuse_matches_fresh_solves() {
    const rates rates{1.4, 0.3};
    const auto model = birth_death();
    const auto level = [](const state &state) { return static_cast<idx>(state[0]); };

    else_sim::ensemble_options reused_options;
    reused_options.capacity = 33;
    reused_options.rule = else_sim::shedding_rule::exact_cut_time;
    reused_options.maximum_steps = 30;
    else_sim::ensemble_diagnostics diagnostics;
    const auto reused = else_sim::else_ensemble(model, rates, state{20}, 96, 0.0, 50.0,
                                                reused_options, 2027u, level, &diagnostics);

    auto fresh_options = reused_options;
    fresh_options.reuse_factorization = false;
    const auto fresh = else_sim::else_ensemble(model, rates, state{20}, 96, 0.0, 50.0,
                                               fresh_options, 2027u, level);
    for (idx t = 0; t < reused.size(); ++t) {
        check::that(reused[t].states == fresh[t].states,
                    "Woodbury score reuse preserves sampled states");
        check::that(reused[t].times.size() == fresh[t].times.size(),
                    "Woodbury score reuse preserves the step count");
        for (idx step = 0; step < reused[t].times.size(); ++step)
            check::close(reused[t].times[step], fresh[t].times[step],
                         "Woodbury score reuse preserves sampled times", 1e-10);
    }
    check::that(!diagnostics.steps.empty(), "the exact-cut-time run records diagnostics");
    check::done("exact cut-time Woodbury reuse matches fresh solves");
}

int main() {
    std::printf("else ensemble\n");
    test_pure_birth_macrostep_is_analytic();
    test_shared_subnetwork_matches_per_entrance_solves();
    test_trajectories_are_well_formed();
    test_ensemble_is_reproducible();
    test_incremental_restriction_matches_fresh_rebuilds();
    test_state_graph_identities_persist();
    test_exact_cut_time_reuse_matches_fresh_solves();
    return check::report("else ensemble");
}
