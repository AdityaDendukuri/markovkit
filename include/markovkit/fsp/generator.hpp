/// @file generator.hpp
/// @brief CME generator matrix construction.
///
/// build_generator constructs the infinitesimal generator A of the CME
/// restricted to the active state space.  Two modes are supported:
///   - absorbing=false: closed form; probability is conserved (for AdaptiveFSP)
///   - absorbing=true:  outgoing probability that leaves the active set is
///                      absorbed into a sink; required for KrylovFSP mass criterion
///                      (Vo & Sidje 2016).
#pragma once

#include "markovkit/fsp/state_space.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include "linalg/sparse/sparse.hpp"
#include <vector>

namespace fsp {

/// @brief Result of build_generator.
struct GeneratorResult {
    num::SparseMatrix A;                    ///< full generator (in_flow + out_flow)
    std::vector<double> exit_rate_boundary; ///< per-state outflux to states outside domain
    std::vector<double> diag;              ///< per-state within-domain outflux = -A(i,i)
};

/// @brief Build the CME generator restricted to sp.
///
/// @param sp          Active state space
/// @param model       Reaction network
/// @param rates       Rate parameters
/// @param t           Current time
/// @param absorbing   If true, diagonal counts ALL outgoing propensities (including
///                    transitions to states outside sp), so probability leaks.
///                    Required for the KrylovFSP mass-loss criterion.
GeneratorResult build_generator(const StateSpace& sp,
                                const ReactionSystem& model,
                                const std::vector<double>& rates,
                                double t,
                                bool absorbing = false);

} // namespace fsp
