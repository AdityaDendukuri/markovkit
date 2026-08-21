#pragma once

#include "markovkit/state.hpp"
#include <vector>

namespace markovkit {

/// Piecewise-constant sample path. Reactions records only actual jumps.
struct Trajectory {
  std::vector<double> times;
  std::vector<State> states;
  std::vector<std::size_t> reactions;
};

} // namespace markovkit
