#include "fsp/solvers/krylov_fsp.hpp"
#include "fsp/generator.hpp"
#include "fsp/state_space.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fsp {

std::tuple<FSPSolution, KrylovFSPDiagnostics> solve_krylov_fsp(const FSPProblem& prob,
                                                               const KrylovFSPParams& params) {
  const markovkit::ReactionSystem& model = prob.model;
  const std::vector<double>& rates = prob.rates;
  const BoundaryCondition& bc = prob.bc;
  double t0 = prob.t0;
  double tf = prob.tf;
  int n_species = static_cast<int>(prob.u0.size());

  StateSpace sp;
  sp.add_state(prob.u0, 1.0);

  FSPSolution sol;
  sol.n_species = n_species;
  sol.times.push_back(t0);
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

    // Expand before propagation so newly reachable probability has support.
    sp.expand(model, bc, params.r);

    // The absorbing diagonal turns missing probability into an error estimate.
    auto assembled = build_generator(sp, model, rates, t, true);
    auto& A = assembled.matrix;
    int n = sp.size();

    // Propagate on the current finite state space.
    num::Vector p_in(static_cast<num::idx>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      p_in[static_cast<num::idx>(i)] = sp.probs[i];
    }

    num::Vector w = num::expv(tau, A, p_in, params.krylov_m, params.krylov_tol);

    // Accept only if retained mass satisfies the Vo--Sidje error budget.
    double mass = 0.0;
    for (num::idx i = 0; i < w.size(); ++i) {
      mass += std::max(0.0, w[i]);
    }

    double required_mass = 1.0 - (params.eps * ((t + tau - t0) / (tf - t0)));

    if (mass < required_mass && tau > 1e-14) {
      tau *= 0.5;
      ++diag.reject_count;
      continue;
    }

    // Commit the accepted probability vector.
    sp.probs.resize(n);
    for (int i = 0; i < n; ++i) {
      sp.probs[i] = std::max(0.0, w[static_cast<num::idx>(i)]);
    }

    // Remove a state only when both its probability and local derivative are small.
    num::Vector dp_dt(static_cast<num::idx>(n), 0.0);
    num::operators::SparseOp(A).apply(w, dp_dt);

    std::vector<bool> remove_mask(n, false);
    for (int i = 0; i < n; ++i) {
      if (sp.probs[i] < droptol &&
          std::abs(dp_dt[static_cast<num::idx>(i)]) < params.droptol_deriv) {
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
      if (params.progress_cb) {
        params.progress_cb(accepted_steps, t, tau, sp.size(), mass);
      }
    }

    if (accepted_steps % params.save_interval == 0 || t >= tf) {
      sol.times.push_back(t);
      sol.snapshots.push_back({sp.states, sp.probs});
      sol.state_space_sizes.push_back(sp.size());
    }

    tau = std::min(tau * params.growth_factor, tf - t);
  }

  if (sol.times.back() < t - (1e-15 * std::abs(t))) {
    sol.times.push_back(t);
    sol.snapshots.push_back({sp.states, sp.probs});
    sol.state_space_sizes.push_back(sp.size());
  }

  return {std::move(sol), std::move(diag)};
}

} // namespace fsp
