#pragma once
#include "fsp/boundary.hpp"
#include "markovkit/reaction_system.hpp"
#include <vector>

namespace fsp {

/// Chemical master equation initial-value problem on a finite domain.
struct FSPProblem {
  markovkit::ReactionSystem model;
  markovkit::State u0;
  double t0 = 0.0;
  double tf = 0.0;
  std::vector<double> rates;
  BoundaryCondition bc;
};

} // namespace fsp
