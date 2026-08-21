#pragma once
#include "fsp/state_space.hpp"
#include "markovkit/boundary_transition.hpp"
#include "markovkit/reaction_system.hpp"
#include <numerics.hpp>
#include <vector>

namespace fsp {

/// Restricted column generator and the transitions omitted from it.
struct GeneratorResult {
  num::SparseMatrix matrix;
  std::vector<double> boundary_rates;
  std::vector<double> diagonal_rates;
  std::vector<markovkit::BoundaryTransition> boundary;
};

/// Build the CME column generator on `sp`.
///
/// If `absorbing` is true, rates leaving `sp` remain in the diagonal so
/// probability lost from the finite system measures truncation error.
GeneratorResult build_generator(const StateSpace& sp,
                                const markovkit::ReactionSystem& model,
                                const std::vector<double>& rates,
                                double t,
                                bool absorbing = false);

} // namespace fsp
