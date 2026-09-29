#pragma once
#include "cme/state_space.hpp"
#include "markovkit/boundary_transition.hpp"
#include "markovkit/reaction_system.hpp"
#include <numerics.hpp>

namespace cme {

/// Restricted column generator and the transitions omitted from it.
struct generator_result {
    num::spmat matrix;
    num::vec boundary_rates;
    num::vec diagonal_rates;
    num::array<markovkit::boundary_transition> boundary;
};

/// Build the CME column generator on `sp`.
///
/// If `absorbing` is true, rates leaving `sp` remain in the diagonal so
/// probability lost from the finite system measures truncation error.
generator_result build_generator(const state_space &sp, const reaction_system &model,
                                 const num::array<double> &rates, double t, bool absorbing = false);

/// A Markov generator restricted to a finite set of states.
struct restricted_generator {
    num::array<markovkit::state> states;
    num::unordered_map<markovkit::state, int> index;
    num::spmat generator{0, 0, {}, {}, {0}};
    num::array<markovkit::boundary_transition> boundary;
    bool symmetric_pattern = false;
};

/// Restrict a reaction generator to an existing finite state space.
[[nodiscard]] restricted_generator restrict_cme(const state_space &states,
                                                const reaction_system &model,
                                                const num::array<double> &rates, double time = 0.0);

} // namespace cme
