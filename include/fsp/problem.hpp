#pragma once
#include "cme/boundary.hpp"
#include "cme/reaction_system.hpp"
#include <vector>

namespace fsp {

/// Chemical master equation initial-value problem on a finite domain.
struct FSPProblem {
    cme::ReactionSystem model;
    markovkit::State u0;
    double t0 = 0.0;
    double tf = 0.0;
    std::vector<double> rates;
    cme::BoundaryCondition bc;
};

} // namespace fsp
