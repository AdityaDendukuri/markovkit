/// @file src/fsp/solvers/adaptive_fsp.cpp
/// @brief Compiled solve_adaptive_fsp.
#include "markovkit/fsp/solvers/adaptive_fsp.hpp"
#include "markovkit/fsp/generator.hpp"
#include "markovkit/fsp/state_space.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <numerics.hpp>
#include <optional>
#include <random>

namespace fsp {

std::tuple<FSPSolution, AdaptiveFSPDiagnostics>
solve_adaptive_fsp(const FSPProblem& prob, const AdaptiveFSPParams& params) {
  const ReactionSystem& model = prob.model;
  const std::vector<double>& rates = prob.rates;
  const BoundaryCondition& bc = prob.bc;
  double t0 = prob.t0;
  double tf = prob.tf;
  int n_species = (int)prob.u0.size();

  StateSpace sp;
  sp.add_state(prob.u0, 1.0);
  sp.expand(model, bc, params.expansion_depth);

  FSPSolution sol;
  sol.n_species = n_species;
  sol.t.push_back(t0);
  sol.snapshots.push_back({sp.states, sp.probs});
  sol.state_space_sizes.push_back(sp.size());

  AdaptiveFSPDiagnostics diag;
  diag.total_iters = 0;

  std::mt19937 rng(params.rng_seed);

  std::optional<num::SparseMatrix> A_cached;
  std::vector<double> erb_cached;
  std::vector<double> diag_cached;
  std::vector<double> exit_cached;
  bool gen_valid = false;

  double t = t0;
  int iter = 0;

  while (t < tf && iter < params.max_iter) {
    ++iter;

    if (params.expand_method == ExpandMethod::ssa)
      sp.expand_ssa(model, rates, t, bc, params.expansion_depth, rng);
    else
      sp.expand(model, bc, params.expansion_depth);

    if (!gen_valid || sp.dirty) {
      auto [A, erb, gd] = build_generator(sp, model, rates, t, true);
      A_cached = std::move(A);
      erb_cached = std::move(erb);
      diag_cached = std::move(gd);

      int n = sp.size();
      exit_cached.resize(n);
      for (int i = 0; i < n; ++i)
        exit_cached[i] = diag_cached[i] + erb_cached[i];

      sp.dirty = false;
      gen_valid = true;
    }

    int n = sp.size();
    double Phi = 0.0;
    for (int i = 0; i < n; ++i)
      Phi += sp.probs[i] * diag_cached[i];

    double dt_flux = (Phi > 0.0) ? params.eps_dt / Phi : (tf - t);
    double dt = std::min(dt_flux, tf - t);

    num::Vector p_in((num::idx)n, 0.0);
    for (int i = 0; i < n; ++i)
      p_in[(num::idx)i] = sp.probs[i];

    num::operators::SparseOp Aop(*A_cached);
    num::Vector p_out = num::expv(dt, Aop, p_in, params.krylov_m, params.krylov_tol);

    sp.probs.resize(n);
    for (int i = 0; i < n; ++i)
      sp.probs[i] = std::max(0.0, p_out[(num::idx)i]);

    sp.compress(model, rates, t, params.prob_quantile, params.flux_tolerance, &exit_cached);
    if (params.abs_threshold > 0.0) {
      int n_before = sp.size();
      std::vector<bool> kill(n_before, false);
      for (int i = 0; i < n_before; ++i)
        kill[i] = sp.probs[i] < params.abs_threshold;
      sp.remove_states(kill);
      gen_valid = false;
    }

    double sum_p = 0.0;
    for (double p : sp.probs)
      sum_p += p;
    if (sum_p > 0.0) {
      for (double& p : sp.probs)
        p /= sum_p;
    }

    t += dt;

    sol.t.push_back(t);
    sol.snapshots.push_back({sp.states, sp.probs});
    sol.state_space_sizes.push_back(sp.size());
  }

  diag.total_iters = iter;
  return {sol, diag};
}

} // namespace fsp
