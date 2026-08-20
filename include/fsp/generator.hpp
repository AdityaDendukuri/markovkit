#pragma once
#include "fsp/reaction_system.hpp"
#include "fsp/state_space.hpp"
#include <numerics.hpp>
#include <vector>

namespace fsp {

// Result of generator construction
struct GeneratorResult {
  num::SparseMatrix A;                    // CME generator
  std::vector<double> exit_rate_boundary; // outflux to states outside StateSpace
  std::vector<double> diag;               // within-domain outflux = -A(i,i)
};

// Build CME generator restricted to sp
GeneratorResult build_generator(const StateSpace& sp,
                                const ReactionSystem& model,
                                const std::vector<double>& rates,
                                double t,
                                bool absorbing = false);

} // namespace fsp
