#pragma once
#include "fsp/multi_index.hpp"
#include "fsp/reaction_system.hpp"
#include <vector>

namespace ssa {

using MultiIndex = fsp::MultiIndex;
using State = fsp::MultiIndex;

struct Trajectory {
  std::vector<double> t;
  std::vector<MultiIndex> states;
  std::vector<std::size_t> reactions;

  [[nodiscard]] std::size_t steps() const { return reactions.size(); }
};

// Gillespie exact direct stochastic simulation
Trajectory gillespie(const fsp::ReactionSystem& model,
                     const std::vector<double>& rates,
                     const MultiIndex& initial,
                     double t0,
                     double tf,
                     int seed = 42);

} // namespace ssa
