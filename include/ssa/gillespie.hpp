#pragma once
#include "markovkit/reaction_system.hpp"
#include "markovkit/trajectory.hpp"
#include <limits>
#include <vector>

namespace ssa {

/// Gillespie's exact direct stochastic simulation method.
markovkit::Trajectory gillespie(const markovkit::ReactionSystem& model,
                                const std::vector<double>& rates,
                                const markovkit::State& initial,
                                double t0,
                                double tf,
                                int seed = 42,
                                std::size_t maximum_steps =
                                    std::numeric_limits<std::size_t>::max());

} // namespace ssa
