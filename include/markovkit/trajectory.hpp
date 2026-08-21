#pragma once

#include "markovkit/state.hpp"
#include <span>
#include <stdexcept>
#include <vector>

namespace markovkit {

/// Piecewise-constant sample path. Reactions records only actual jumps.
struct Trajectory {
  std::vector<double> times;
  std::vector<State> states;
  std::vector<std::size_t> reactions;
};

/// Extract one species from every recorded trajectory state.
inline std::vector<double> trajectory_component(const Trajectory& trajectory, std::size_t species) {
  std::vector<double> values;
  values.reserve(trajectory.states.size());
  for (const auto& state : trajectory.states) {
    if (species >= state.size()) {
      throw std::out_of_range("trajectory species index is out of range");
    }
    values.push_back(state[species]);
  }
  return values;
}

/// Locate the piecewise-constant trajectory state at each observation time.
inline std::vector<std::size_t> trajectory_indices_at(const Trajectory& trajectory,
                                                      std::span<const double> observation_times) {
  if (trajectory.times.empty() || trajectory.times.size() != trajectory.states.size()) {
    throw std::invalid_argument("trajectory times and states must be nonempty and aligned");
  }
  std::vector<std::size_t> indices;
  indices.reserve(observation_times.size());
  std::size_t state_index = 0;
  for (std::size_t observation = 0; observation < observation_times.size(); ++observation) {
    const double time = observation_times[observation];
    if (observation > 0 && time < observation_times[observation - 1]) {
      throw std::invalid_argument("observation times must be ordered");
    }
    while (state_index + 1 < trajectory.times.size() && trajectory.times[state_index + 1] <= time) {
      ++state_index;
    }
    indices.push_back(state_index);
  }
  return indices;
}

} // namespace markovkit
