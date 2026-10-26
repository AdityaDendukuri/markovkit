// the paper's subnetwork selection and support sparsification (Sections 3 and 4)
#pragma once

#include "container/swap_remove.hpp"
#include "stats/selection.hpp"
#include "subsweep/exit.hpp"
#include "subsweep/scores.hpp"
#include "subsweep/subnetwork.hpp"
#include "subsweep/sweep.hpp"
#include <algorithm>
#include <utility>

namespace subsweep {

// rho on D from the weighted entrances, normalized
template <typename State>
[[nodiscard]] num::vec<real> entrance_measure(const subnetwork<State> &D,
                                              const array<std::pair<State, real>> &entrances) {
    real total = 0.0;
    for (const auto &[x, weight] : entrances) {
        total += weight;
    }
    num::vec<real> rho(D.size(), 0.0);
    for (const auto &[x, weight] : entrances) {
        rho[*D.find(x)] += weight / total;
    }
    return rho;
}

// admit the entrances, then grow `depth` layers while below `capacity`; true if grown
template <typename State, listed_oracle<State> Rates>
bool grow(const Rates &rates, array<State> &states, const array<std::pair<State, real>> &entrances,
          int depth, idx capacity, real negligible_rate) {
    unordered_set<State> included(states.begin(), states.end());
    array<State> layer;
    for (const auto &[x, weight] : entrances) {
        if (included.insert(x).second) {
            states.push_back(x);
        }
        layer.push_back(x);
    }
    bool grown = false;
    for (int step = 0; step < depth && states.size() < capacity; ++step) {
        array<State> next;
        for (const State &x : layer) {
            for (const auto &[y, rate] : rates.transitions(x)) {
                if (included.insert(y).second && detail::total_rate(rates, y) > negligible_rate) {
                    states.push_back(y);
                    next.push_back(y);
                    grown = true;
                }
            }
        }
        layer = std::move(next);
    }
    return grown;
}

// remove the `count` lowest scores, never an entrance; ties go to the highest
// level, then the latest
template <typename State>
void shed(array<State> &states, const num::vec<real> &score,
          const array<std::pair<State, real>> &entrances, idx count, const array<idx> &levels) {
    unordered_set<State> entering;
    for (const auto &[x, weight] : entrances) {
        entering.insert(x);
    }
    array<idx> candidates =
        num::filter(states.size(), [&](idx i) { return !entering.contains(states[i]); });
    auto sheds_first = [&](idx a, idx b) {
        if (score[a] != score[b]) {
            return score[a] < score[b];
        }
        if (!levels.empty() && levels[a] != levels[b]) {
            return levels[a] > levels[b];
        }
        return a > b;
    };
    std::ranges::sort(candidates, sheds_first);
    candidates.resize(std::min(count, candidates.size()));
    num::remove_indices(states, candidates);
}

// expected-visit scores from a fresh factor
template <typename State, detail::oracle<State> Rates>
[[nodiscard]] num::vec<real> fresh_visit_scores(const Rates &rates, const array<State> &states,
                                                const array<std::pair<State, real>> &entrances) {
    subnetwork<State> E = make_subnetwork(rates, states);
    num::vec<real> rho = entrance_measure(E, entrances);
    num::vec<real> u = occupation_times(num::lu(E.matrix(), num::sparse), rho);
    num::vec<real> score = expected_visit_scores(E.total_rates(), u, rho);
    num::vec<real> result(states.size(), 0.0);
    for (idx i = 0; i < states.size(); ++i) {
        result[i] = score[*E.find(states[i])];
    }
    return result;
}

// expected-visit scores from the last sweep; a state new to D scores zero
template <typename State>
[[nodiscard]] num::vec<real>
last_visit_scores(const subnetwork<State> &previous, const num::vec<real> &u,
                  const array<std::pair<State, real>> &entrances, const array<State> &states) {
    num::vec<real> score = expected_visit_scores(previous.total_rates(), u,
                                                 entrance_measure(previous, entrances));
    num::vec<real> result(states.size(), 0.0);
    for (idx i = 0; i < states.size(); ++i) {
        if (const auto j = previous.find(states[i])) {
            result[i] = score[*j];
        }
    }
    return result;
}

// the paper's selection rule: admit the entrances, grow, shed back to capacity.
// `keep_last` starts from the last D, so that its factor can be reused
template <typename State, detail::oracle<State> Rates>
[[nodiscard]] auto capacity_selection(const Rates &rates, const sweep_settings &settings,
                                      bool keep_last = true) {
    return [&rates, settings, keep_last, last_entrances = array<std::pair<State, real>>{}](
               const array<std::pair<State, real>> &entrances, const subnetwork<State> *previous,
               const num::vec<real> *u) mutable {
        array<State> states;
        if (keep_last && previous) {
            states.assign(previous->states().begin(), previous->states().end());
        }
        bool grown = false;
        if constexpr (listed_oracle<Rates, State>) {
            grown = grow(rates, states, entrances, settings.depth, settings.capacity,
                         settings.negligible_rate);
        } else {
            unordered_set<State> included(states.begin(), states.end());
            for (const auto &[x, weight] : entrances) {
                if (included.insert(x).second) {
                    states.push_back(x);
                }
            }
        }
        if (states.size() > settings.capacity) {
            num::vec<real> score = grown || !previous || !keep_last
                                       ? fresh_visit_scores(rates, states, entrances)
                                       : last_visit_scores(*previous, *u, last_entrances, states);
            array<idx> levels;
            if constexpr (leveled_oracle<Rates, State>) {
                if (settings.shed_by_level) {
                    for (const State &x : states) {
                        levels.push_back(rates.level(x));
                    }
                }
            }
            shed(states, score, entrances, states.size() - settings.capacity, levels);
        }
        last_entrances = entrances;
        return states;
    };
}

} // namespace subsweep
