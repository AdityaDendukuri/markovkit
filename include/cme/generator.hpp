#pragma once
#include "cme/state_space.hpp"
#include "markovkit/boundary_transition.hpp"
#include "markovkit/reaction_system.hpp"
#include <numerics.hpp>
#include <unordered_map>
#include <vector>

namespace cme {

/// Restricted column generator and the transitions omitted from it.
struct GeneratorResult {
    num::spmat matrix;
    std::vector<double> boundary_rates;
    std::vector<double> diagonal_rates;
    std::vector<markovkit::BoundaryTransition> boundary;
};

/// Build the CME column generator on `sp`.
///
/// If `absorbing` is true, rates leaving `sp` remain in the diagonal so
/// probability lost from the finite system measures truncation error.
GeneratorResult build_generator(const StateSpace &sp, const ReactionSystem &model,
                                const std::vector<double> &rates, double t, bool absorbing = false);

/// A Markov generator restricted to a finite set of states.
struct RestrictedGenerator {
    std::vector<markovkit::State> states;
    std::unordered_map<markovkit::State, int> index;
    num::spmat generator{0, 0, {}, {}, {0}};
    std::vector<markovkit::BoundaryTransition> boundary;
    bool symmetric_pattern = false;
};

/// Restrict a reaction generator to an existing finite state space.
[[nodiscard]] RestrictedGenerator restrict_cme(const StateSpace &states,
                                               const ReactionSystem &model,
                                               const std::vector<double> &rates, double time = 0.0);

} // namespace cme
