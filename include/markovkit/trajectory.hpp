#pragma once

#include "core/types.hpp"
#include "markovkit/state.hpp"
#include <stdexcept>

namespace markovkit {

/// Piecewise-constant sample path. Reactions records only actual jumps.
struct trajectory {
    num::array<double> times;
    num::array<state> states;
    num::array<num::idx> reactions;
};

/// Extract one species from every recorded trajectory state.
template <typename Traj>
inline num::vec trajectory_component(const Traj &trajectory, std::size_t species) {
    num::vec values(trajectory.states.size());
    for (num::idx index = 0; index < values.size(); ++index) {
        const auto &state = trajectory.states[index];
        if (species >= state.size()) {
            throw std::out_of_range("trajectory species index is out of range");
        }
        values[index] = state[species];
    }
    return values;
}

/// Locate the piecewise-constant trajectory state at each observation time.
template <typename Traj>
inline num::array<num::idx> trajectory_indices_at(const Traj &trajectory,
                                                  num::view<const double> observation_times) {
    if (trajectory.times.empty() || trajectory.times.size() != trajectory.states.size()) {
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
        while (state_index + 1 < trajectory.times.size() &&
               trajectory.times[state_index + 1] <= time) {
            ++state_index;
        }
        indices.push_back(state_index);
    }
    return indices;
}

} // namespace markovkit
