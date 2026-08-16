/// @file problem.hpp
/// @brief FSPProblem: complete specification of a CME initial value problem.
#pragma once

#include "markovkit/fsp/boundary.hpp"
#include "markovkit/fsp/multi_index.hpp"
#include "markovkit/fsp/reaction_system.hpp"

namespace fsp {

/// @brief Full specification of a Chemical Master Equation problem.
struct FSPProblem {
  ReactionSystem model;      ///< reaction network
  MultiIndex u0;             ///< initial state (deterministic, probability 1)
  double t0;                 ///< start time
  double tf;                 ///< end time
  std::vector<double> rates; ///< rate parameters passed to propensities
  BoundaryCondition bc;      ///< domain boundary
};

} // namespace fsp
