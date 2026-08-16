/// @file generator.hpp
/// @brief CME generator matrix construction.
#pragma once

#include "linalg/sparse/sparse.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include "markovkit/fsp/state_space.hpp"
#include <vector>

namespace fsp {

/// @brief Result of build_generator.
struct GeneratorResult {
  num::SparseMatrix A; ///< full generator (in_flow + out_flow)
  std::vector<double>
      exit_rate_boundary;   ///< per-state outflux to states outside domain
  std::vector<double> diag; ///< per-state within-domain outflux = -A(i,i)
};

/// @brief Build the CME generator restricted to sp.
GeneratorResult build_generator(const StateSpace &sp,
                                const ReactionSystem &model,
                                const std::vector<double> &rates, double t,
                                bool absorbing = false);

} // namespace fsp
