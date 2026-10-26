// breadth-first enumeration from a user-supplied set of states
#pragma once

#include "subsweep/chain.hpp"

namespace subsweep {

template <typename State, listed_oracle<State> Rates>
[[nodiscard]] array<State> enumerate_states(const Rates &rates, const array<State> &seeds,
                                            int depth) {
    array<State> states, frontier;
    unordered_set<State> included;
    for (const State &state : seeds) {
        if (included.insert(state).second) {
            states.push_back(state);
            frontier.push_back(state);
        }
    }
    for (int step = 0; step < depth && !frontier.empty(); ++step) {
        array<State> next;
        for (const State &state : frontier) {
            for (auto &[destination, rate] : rates.transitions(state)) {
                if (rate > 0.0 && included.insert(destination).second) {
                    states.push_back(destination);
                    next.push_back(std::move(destination));
                }
            }
        }
        frontier = std::move(next);
    }
    return states;
}

} // namespace subsweep
