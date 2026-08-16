/// @file src/fsp/solvers/krylov_fsp.cpp
/// @brief Compiled solve_krylov_fsp.
#include "markovkit/fsp/solvers/krylov_fsp.hpp"
#include "markovkit/fsp/generator.hpp"
#include "markovkit/fsp/state_space.hpp"
#include <algorithm>
#include <cmath>
#include <numerics.hpp>
#include <random>

namespace fsp {

std::tuple<FSPSolution, KrylovFSPDiagnostics> solve_krylov_fsp(const FSPProblem& prob,
                                                               const KrylovFSPParams& params) {
  const ReactionSystem& model = prob.model;
  const std::vector<double>& rates = prob.rates;
  const BoundaryCondition& bc = prob.bc;
  double t0 = prob.t0;
  double tf = prob.tf;
  int n_species = (int)prob.u0.size();

  std::mt19937 rng(42);

  StateSpace sp;
  sp.add_state(prob.u0, 1.0);
  sp.expand(model, bc, params.r);

  FSPSolution sol;
  sol.n_species = n_species;
  sol.t.push_back(t0);
  sol.snapshots.push_back({sp.states, sp.probs});
  sol.state_space_sizes.push_back(sp.size());

  KrylovFSPDiagnostics diag;
  diag.reject_count = 0;
  diag.total_iters = 0;

  double t = t0;
  double tau = params.tau_init;
  int iter = 0;

  while (t < tf && iter < params.max_iter) {
    ++iter;

    auto [A, exit_rate_boundary, gen_diag] = build_generator(sp, model, rates, t, true);
    int n = sp.size();

    bool iexpand = false;
    double tau_step = std::min(tau, tf - t);

    num::Vector p_new;
    bool fsp_ok = false;
    int n_tau_reduce = 0;

    while (!fsp_ok && n_tau_reduce < 60) {
      tau_step = std::min(tau, tf - t);

      num::Vector p_in((num::idx)n, 0.0);
      for (int i = 0; i < n; ++i)
        p_in[(num::idx)i] = sp.probs[i];

      num::operators::SparseOp Aop(A);
      p_new = num::expv(tau_step, Aop, p_in, params.krylov_m, params.krylov_tol);

      bool has_bad = false;
      for (int i = 0; i < n; ++i) {
        if (std::isnan(p_new[(num::idx)i]) || std::isinf(p_new[(num::idx)i])) {
          has_bad = true;
          break;
        }
        if (p_new[(num::idx)i] < 0.0)
          p_new[(num::idx)i] = 0.0;
      }

      if (has_bad) {
        tau /= 2.0;
        iexpand = true;
        ++diag.reject_count;
        ++n_tau_reduce;
        continue;
      }

      double mass = 0.0;
      for (int i = 0; i < n; ++i)
        mass += p_new[(num::idx)i];

      double fsp_threshold = 1.0 - params.eps * (t + tau_step) / tf;
      if (mass >= fsp_threshold) {
        fsp_ok = true;
      } else {
        tau /= 2.0;
        iexpand = true;
        ++diag.reject_count;
        ++n_tau_reduce;
      }
    }

    t += tau_step;
    sp.probs.resize(n);
    for (int i = 0; i < n; ++i)
      sp.probs[i] = p_new[(num::idx)i];

    double mass_before = 0.0;
    for (double p : sp.probs)
      mass_before += p;

    num::operators::SparseOp Aop(A);
    num::Vector p_vec((num::idx)n, 0.0);
    for (int i = 0; i < n; ++i)
      p_vec[(num::idx)i] = sp.probs[i];
    num::Vector Ap((num::idx)n, 0.0);
    Aop.apply(p_vec, Ap);

    std::vector<bool> remove_mask(n, false);
    double removed_mass = 0.0;
    for (int i = 0; i < n; ++i) {
      double p_val = sp.probs[i];
      double ap_val = std::abs(Ap[(num::idx)i]);
      if (p_val < params.droptol && ap_val < params.droptol_deriv) {
        remove_mask[i] = true;
        removed_mass += p_val;
      }
    }

    double remaining_budget = (mass_before - (1.0 - params.eps));
    if (removed_mass <= remaining_budget) {
      sp.remove_states(remove_mask);
    }

    double sum_p = 0.0;
    for (double p : sp.probs)
      sum_p += p;
    if (sum_p > 0.0) {
      for (double& p : sp.probs)
        p /= sum_p;
    }

    if (iexpand) {
      sp.expand_ssa(model, rates, t, bc, params.r, rng);
    } else {
      sp.expand(model, bc, params.r);
      tau *= params.growth_factor;
    }

    sol.t.push_back(t);
    sol.snapshots.push_back({sp.states, sp.probs});
    sol.state_space_sizes.push_back(sp.size());
  }

  diag.total_iters = iter;
  return {sol, diag};
}

} // namespace fsp
