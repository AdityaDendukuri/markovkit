#pragma once
#include "cme/state_space.hpp"
#include "container/vector.hpp"
#include "linear/sparse/sparse.hpp"
#include "markovkit/reaction_system.hpp"

namespace cme {

/// Restricted column generator, with the rate at which each state leaves the
/// finite space and its rate to other retained states.
struct generator_result {
    num::spmat matrix;
    num::vec<double> boundary_rates;
    num::vec<double> diagonal_rates;
};

/// Build the CME column generator on `sp`.
///
/// If `absorbing` is true, rates leaving `sp` remain in the diagonal so
/// probability lost from the finite system measures truncation error.
generator_result build_generator(const state_space &sp, const reaction_system &model,
                                 const num::array<double> &rates, double t, bool absorbing = false);

} // namespace cme
