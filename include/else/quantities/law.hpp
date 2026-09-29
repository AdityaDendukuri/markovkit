// Entrance laws: occupation quantities shared by trajectories.
#pragma once

#include "else/core/subnetwork.hpp"
#include "else/quantities/factor_update.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include <algorithm>
#include <stdexcept>

namespace else_sim {

// U = EZ and V = EZ^2, one row per entrance: U(k,j) is the expected time
// spent in j before escape, starting from the kth entrance.
struct entrance_law_result {
    num::mat occupation;        // U
    num::mat second_occupation; // V
};

// Update cached U = E Z0 and V = E Z0^2 through the exact Woodbury identity,
// where update represents Z = (A0 + P Q^T)^-1.  The entrance rows E must be
// unchanged; callers solve normally for entrances that are new this step.
template <typename Update>
[[nodiscard]] entrance_law_result update_entrance_law(const entrance_law_result &base_law,
                                                      const Update &update) {
    using namespace num::ops;
    if (base_law.occupation.cols() != update.left.rows() ||
        base_law.second_occupation.rows() != base_law.occupation.rows() ||
        base_law.second_occupation.cols() != base_law.occupation.cols() ||
        update.transpose_inverse_right_squared.rows() != update.left.rows()) {
        throw std::invalid_argument("cached entrance law has incompatible dimensions");
    }

    const num::mat inverse_right = num::transpose(update.transpose_inverse_right);
    const num::mat inverse_right_squared = num::transpose(update.transpose_inverse_right_squared);
    const num::mat coefficients = right_solve_reduced(update, base_law.occupation * update.left);
    const num::mat occupation = base_law.occupation - coefficients * inverse_right;
    const num::mat intermediate = base_law.second_occupation - coefficients * inverse_right_squared;
    const num::mat second_coefficients = right_solve_reduced(update, intermediate * update.left);
    const num::mat second_occupation = intermediate - second_coefficients * inverse_right;
    return {occupation, second_occupation};
}

// Solve U M = E, then V M = U, both as transpose solves against `solver`;
// the columns of E are the entrance indicators.
template <typename State, typename Solver>
[[nodiscard]] entrance_law_result entrance_law(const subnetwork<State> &subnetwork,
                                               view<const idx> entrances, const Solver &solver) {
    if (entrances.empty()) {
        throw std::invalid_argument("entrance_law requires at least one entrance");
    }

    num::mat indicators(size(subnetwork), entrances.size(), 0.0);
    for (idx k = 0; k < entrances.size(); ++k) {
        if (entrances[k] >= size(subnetwork)) {
            throw std::out_of_range("entrance state is outside the subnetwork");
        }
        indicators(entrances[k], k) = 1.0;
    }

    const num::mat occupation = solve_transpose(solver, indicators);
    const num::mat second_occupation = solve_transpose(solver, occupation);
    return {num::transpose(occupation), num::transpose(second_occupation)};
}

template <typename State>
[[nodiscard]] entrance_law_result entrance_law(const subnetwork<State> &subnetwork,
                                               view<const idx> entrances) {
    return entrance_law(subnetwork, entrances, subnetwork);
}

// beta_j = w_j * U(k,j) over the escape states, normalized to remove roundoff.
template <typename State>
[[nodiscard]] num::vec escape_distribution(const subnetwork<State> &subnetwork,
                                           const entrance_law_result &law, idx entrance_row) {
    const auto &escape_states = subnetwork.escape_states;
    const auto &escape_rates = subnetwork.escape_rates;

    num::vec distribution(escape_states.size(), 0.0);
    real mass = 0.0;
    for (idx position = 0; position < escape_states.size(); ++position) {
        const idx j = escape_states[position];
        distribution[position] = std::max(0.0, escape_rates[j] * law.occupation(entrance_row, j));
        mass += distribution[position];
    }
    if (!(mass > 0.0)) {
        throw std::runtime_error("subnetwork has no reachable escape from this entrance");
    }
    for (real &value : distribution) {
        value /= mass;
    }
    return distribution;
}

// mu_kj = V(k,j) / U(k,j):  escape time conditioned on escaping via j.
[[nodiscard]] inline real conditional_escape_time(const entrance_law_result &law, idx entrance_row,
                                                  idx escape_state) {
    const real occupation = law.occupation(entrance_row, escape_state);
    if (!(occupation > 0.0)) {
        return 0.0;
    }
    return law.second_occupation(entrance_row, escape_state) / occupation;
}

// Mean escape time from entrance k: sum_j U(k,j).
[[nodiscard]] inline real mean_escape_time(const entrance_law_result &law, idx entrance_row) {
    real total = 0.0;
    for (idx j = 0; j < law.occupation.cols(); ++j) {
        total += law.occupation(entrance_row, j);
    }
    return total;
}

} // namespace else_sim
