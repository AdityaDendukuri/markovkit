// state-removal scores from Section 3 of the paper
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/factorization/factor.hpp"
#include "linear/factorization/probed_inverse_diagonal.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/solvers/pcg.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "subsweep/subnetwork.hpp"
#include <stdexcept>

namespace subsweep {

using num::idx;
using num::real;
using num::view;

// v_j = R_jj u_j - rho_j
[[nodiscard]] inline num::vec<real>
expected_visit_scores(view<const real> total_rates, view<const real> u, view<const real> rho) {
    if (total_rates.size() != u.size() || u.size() != rho.size()) {
        throw std::invalid_argument("R_jj, u, and rho must have the same size");
    }
    num::vec<real> score(u.size(), 0.0);
    for (idx j = 0; j < u.size(); ++j) {
        score[j] = total_rates[j] * u[j] - rho[j];
    }
    return score;
}

// ell_j = u_j (Z 1)_j / Z_jj
[[nodiscard]] inline num::vec<real> cut_time_scores(view<const real> u, view<const real> Z1,
                                                    view<const real> diagonal_Z) {
    if (u.size() != Z1.size() || u.size() != diagonal_Z.size()) {
        throw std::invalid_argument("u, Z 1, and diag(Z) must have the same size");
    }
    num::vec<real> score(u.size(), 0.0);
    for (idx j = 0; j < u.size(); ++j) {
        score[j] = diagonal_Z[j] > 0.0 ? u[j] * Z1[j] / diagonal_Z[j] : 0.0;
    }
    return score;
}

// Exact cut-time scores from a factorization or updated solver for Z = R^{-1}.
template <typename State, typename Solver>
[[nodiscard]] num::vec<real> exact_cut_time_scores(const subnetwork<State> &D,
                                                   const num::vec<real> &rho, const Solver &Z,
                                                   view<const real> diagonal_Z) {
    if (rho.size() != D.size() || diagonal_Z.size() != D.size()) {
        throw std::invalid_argument("rho and diag(Z) must have one entry per state");
    }
    auto u = num::solve(num::transpose(Z), rho);
    auto Z1 = num::solve(Z, num::vec<real>(D.size(), 1.0));
    return cut_time_scores(u, Z1, diagonal_Z);
}

// One approximate Cholesky factor preconditions the two solves and every
// Lanczos action used to estimate diag(Z).
template <typename State>
[[nodiscard]] num::vec<real>
approximate_cut_time_scores(const subnetwork<State> &D, const num::vec<real> &rho,
                            view<const real> symmetrizer,
                            num::inverse_diagonal_options options = {}) {
    if (rho.size() != D.size() || symmetrizer.size() != D.size()) {
        throw std::invalid_argument("rho and the symmetrizer must have one entry per state");
    }
    const num::spmat &R = D.matrix();
    num::vec<real> inverse_scale(D.size(), 0.0);
    num::vec<real> squared_scale(D.size(), 0.0);
    for (idx j = 0; j < D.size(); ++j) {
        inverse_scale[j] = 1.0 / symmetrizer[j];
        squared_scale[j] = symmetrizer[j] * symmetrizer[j];
    }
    const num::spmat symmetric = num::sparse_diagonal_similarity(R, inverse_scale);
    const num::spmat M = num::sparse_congruence(symmetric, symmetrizer);
    const auto approximate =
        num::grounded_approxchol_factor(M, options.preconditioner, options.seed ^ 0x9e3779b9U);
    const auto M_operator = num::assume_spd(num::operators::sparse_op(M));

    num::vec<real> scaled_u(D.size(), 0.0);
    auto u_solve = num::pcg(M_operator, approximate, rho, scaled_u);
    num::vec<real> Z1(D.size(), 0.0);
    auto Z1_solve = num::pcg(M_operator, approximate, squared_scale, Z1);
    if (!u_solve.converged || !Z1_solve.converged) {
        throw std::runtime_error("AC2-preconditioned cut-time solve did not converge");
    }
    num::vec<real> u(D.size(), 0.0);
    for (idx j = 0; j < D.size(); ++j) {
        u[j] = squared_scale[j] * scaled_u[j];
    }
    num::vec<real> diagonal_Z = num::inverse_diagonal(R, symmetrizer, approximate, options);
    return cut_time_scores(u, Z1, diagonal_Z);
}

// Symmetric generator blocks need no user-supplied similarity weights.
template <typename State>
[[nodiscard]] num::vec<real>
approximate_cut_time_scores(const subnetwork<State> &D, const num::vec<real> &rho,
                            num::inverse_diagonal_options options = {}) {
    return approximate_cut_time_scores(D, rho, num::vec<real>(D.size(), 1.0), options);
}

} // namespace subsweep
