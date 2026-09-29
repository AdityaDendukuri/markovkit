#pragma once
#include "cme/boundary.hpp"
#include "cme/reaction_system.hpp"
#include "core/types.hpp"

namespace fsp {

/// Chemical master equation initial-value problem on a finite domain.
struct fsp_problem {
    cme::reaction_system model;
    markovkit::state u0;
    double t0 = 0.0;
    double tf = 0.0;
    num::array<double> rates;
    cme::boundary_condition bc;
};

} // namespace fsp
