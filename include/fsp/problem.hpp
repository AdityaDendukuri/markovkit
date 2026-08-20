#pragma once
#include "fsp/boundary.hpp"
#include "fsp/multi_index.hpp"
#include "fsp/reaction_system.hpp"
#include <vector>

namespace fsp {

// Complete CME initial value problem
struct FSPProblem {
  ReactionSystem model;
  MultiIndex u0;
  double t0 = 0.0;
  double tf = 0.0;
  std::vector<double> rates;
  BoundaryCondition bc;
};

} // namespace fsp
