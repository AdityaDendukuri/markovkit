#include "fsp/solvers/krylov_fsp.hpp"
#include "fsp/generator.hpp"
#include "fsp/state_space.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fsp {

std::tuple<FSPSolution, KrylovFSPDiagnostics>
solve_krylov_fsp(const FSPProblem& prob, const KrylovFSPParams& params) {
  const ReactionSystem& model = prob.model;
  const std::vector<double>& rates = prob.rates;
  const BoundaryCondition& bc = prob.bc;
  double t0 = prob.t0;
  double tf = prob.tf;
  int n_species = static_cast<int>(prob.u0.size());

  StateSpace sp;
  sp.add_state(prob.u0, 1.0);

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
  double droptol = params.droptol;
  int accepted_steps = 0;

  while (t < tf && diag.total_iters < params.max_iter) {
    ++diag.total_iters;

    tau = std::min(tau, tf - t);

    // 1) Expand state space
    sp.expand(model, bc, params.r);

    // 2) Build absorbing generator
    auto [A, erb, gen_diag] = build_generator(sp, model, rates, t, true);
    int n = sp.size();

    // 3) Propagate via Krylov expv
    num::Vector p_in(static_cast<num::idx>(n), 0.0);
    for (int i = 0; i < n; ++i) p_in[static_cast<num::idx>(i)] = sp.probs[i];

    num::Vector w = num::expv(tau, A, p_in, params.krylov_m, params.krylov_tol);

    // 4) Check Vo & Sidje acceptance condition: sum(w) >= 1 - eps * (t + tau) / tf
    double mass = 0.0;
    for (num::idx i = 0; i < w.size(); ++i) mass += std::max(0.0, w[i]);

    double required_mass = 1.0 - params.eps * ((t + tau - t0) / (tf - t0));

    if (mass < required_mass && tau > 1e-14) {
      tau *= 0.5;
      ++diag.reject_count;
      continue;
    }

    // Step accepted: update probabilities
    sp.probs.resize(n);
    for (int i = 0; i < n; ++i) sp.probs[i] = std::max(0.0, w[static_cast<num::idx>(i)]);

    // 5) Joint probability and derivative pruning
    num::Vector dp_dt(static_cast<num::idx>(n), 0.0);
    num::operators::SparseOp(A).apply(w, dp_dt);

    std::vector<bool> remove_mask(n, false);
    for (int i = 0; i < n; ++i) {
      if (sp.probs[i] < droptol && std::abs(dp_dt[static_cast<num::idx>(i)]) < params.droptol_deriv) {
        remove_mask[i] = true;
      }
    }
    sp.remove_states(remove_mask);
    sp.renormalize();

    t += tau;
    ++accepted_steps;

    diag.tau_log.push_back(tau);
    diag.mass_log.push_back(mass);
    diag.time_log.push_back(t);
    diag.size_log.push_back(sp.size());

    if (params.log_interval > 0 && accepted_steps % params.log_interval == 0) {
      if (params.progress_cb) params.progress_cb(accepted_steps, t, tau, sp.size(), mass);
    }

    if (accepted_steps % params.save_interval == 0 || t >= tf) {
      sol.t.push_back(t);
      sol.snapshots.push_back({sp.states, sp.probs});
      sol.state_space_sizes.push_back(sp.size());
    }

    tau = std::min(tau * params.growth_factor, tf - t);
  }

  if (sol.t.back() < t - 1e-15 * std::abs(t)) {
    sol.t.push_back(t);
    sol.snapshots.push_back({sp.states, sp.probs});
    sol.state_space_sizes.push_back(sp.size());
  }

  return {std::move(sol), std::move(diag)};
}

} // namespace fsp
