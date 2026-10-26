#pragma once
#include "container/vector.hpp"

#include "core/types.hpp"
#include "markovkit/state.hpp"
#include <stdexcept>

namespace markovkit {

/// Piecewise-constant sample path. Reactions record only actual jumps.
struct trajectory {
    num::array<double> times;
    num::array<state> states;
    num::array<num::idx> reactions;
};

/// Extract one species from every recorded trajectory state.
inline num::vec<double> trajectory_component(const trajectory &path, std::size_t species) {
    num::vec<double> values(path.states.size());
    for (num::idx index = 0; index < values.size(); ++index) {
        const auto &state = path.states[index];
        if (species >= state.size()) {
            throw std::out_of_range("trajectory species index is out of range");
        }
        values[index] = state[species];
    }
    return values;
}

/// Locate the piecewise-constant trajectory state at each observation time.
inline num::array<num::idx> trajectory_indices_at(const trajectory &path,
                                                  num::view<const double> observation_times) {
    if (path.times.empty() || path.times.size() != path.states.size()) {
        throw std::invalid_argument("trajectory times and states must be nonempty and aligned");
    }
    num::array<num::idx> indices;
    indices.reserve(observation_times.size());
    num::idx state_index = 0;
    for (std::size_t observation = 0; observation < observation_times.size(); ++observation) {
        const double time = observation_times[observation];
        if (observation > 0 && time < observation_times[observation - 1]) {
            throw std::invalid_argument("observation times must be ordered");
        }
        while (state_index + 1 < path.times.size() && path.times[state_index + 1] <= time) {
            ++state_index;
        }
        indices.push_back(state_index);
    }
    return indices;
}

} // namespace markovkit
