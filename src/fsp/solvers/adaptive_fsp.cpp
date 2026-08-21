#include "fsp/solvers/adaptive_fsp.hpp"
#include "cme/generator.hpp"
#include "cme/state_space.hpp"
#include "markovkit/statistics.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace fsp {

std::tuple<FSPSolution, AdaptiveFSPDiagnostics>
solve_adaptive_fsp(const FSPProblem& prob, const AdaptiveFSPParams& params) {
  const markovkit::ReactionSystem& model = prob.model;
  const std::vector<double>& rates = prob.rates;
  const cme::BoundaryCondition& bc = prob.bc;
  double t0 = prob.t0;
  double tf = prob.tf;
  int n_species = static_cast<int>(prob.u0.size());

  cme::StateSpace sp;
  sp.add_state(prob.u0, 1.0);

  // Save the initial point mass as the first snapshot.
  FSPSolution sol;
  sol.n_species = n_species;
  sol.times.push_back(t0);
  sol.snapshots.push_back({sp.states, sp.probs});
  sol.state_space_sizes.push_back(sp.size());

  AdaptiveFSPDiagnostics diag;
  std::mt19937 rng(params.rng_seed);

  double t = t0;
  int iter = 0;

  while (t < tf && iter < params.max_iter) {
    ++iter;

    // Grow the active state space before propagation.
    if (params.expand_method == ExpandMethod::ssa) {
      sp.expand_ssa(model, rates, t, bc, params.expansion_depth, rng);
    } else {
      sp.expand(model, bc, params.expansion_depth);
    }

    auto generator = cme::build_generator(sp, model, rates, t, false);
    std::vector<double> reaction_rates(generator.diagonal_rates.size());
    for (std::size_t i = 0; i < reaction_rates.size(); ++i) {
      reaction_rates[i] = generator.diagonal_rates[i] + generator.boundary_rates[i];
    }

    int n = sp.size();
    // Expected internal activity sets the next propagation interval.
    const double phi = num::dot(std::span<const double>(sp.probs),
                                std::span<const double>(generator.diagonal_rates));

    double dt_flux = (phi > 0.0) ? params.eps_dt / phi : (tf - t);
    double dt = std::min(dt_flux, tf - t);

    num::Vector p_in(sp.probs);

    // Apply the CME exponential over the selected interval.
    num::Vector p_out = num::expv(dt, generator.matrix, p_in, params.krylov_m, params.krylov_tol);

    num::kernel::array::map(p_out, [](double value) { return std::max(0.0, value); });
    sp.probs.resize(n);
    num::copy_to(p_out, sp.probs);

    // Prune after propagation, while retaining states carrying appreciable flux.
    sp.compress(model, rates, t, params.prob_quantile, params.flux_tolerance, &reaction_rates);
    if (params.abs_threshold > 0.0) {
      // Apply an optional absolute cutoff after flux pruning.
      int n_before = sp.size();
      std::vector<bool> kill(n_before, false);
      for (int i = 0; i < n_before; ++i) {
        kill[i] = sp.probs[i] < params.abs_threshold;
      }
      sp.remove_states(kill);
    }
    double mass_before_norm = sp.renormalize();
    double pruned_mass = 1.0 - mass_before_norm;

    t += dt;

    diag.dt_log.push_back(dt);
    diag.t_log.push_back(t);
    diag.size_log.push_back(sp.size());
    diag.flux_log.push_back(phi);

    if (params.log_interval > 0 && iter % params.log_interval == 0) {
      // Materialize diagnostics only when requested.
      if (params.progress_cb) {
        const auto means = markovkit::species_means(sp.states, sp.probs);
        params.progress_cb(iter, t, dt, sp.size(), means, pruned_mass);
      }
      if (params.dist_cb) {
        params.dist_cb(t, sp.states, sp.probs);
      }
    }

    if (iter % params.save_interval == 0 || t >= tf) {
      // Save only configured checkpoints and the final state.
      sol.times.push_back(t);
      sol.snapshots.push_back({sp.states, sp.probs});
      sol.state_space_sizes.push_back(sp.size());
    }
  }

  if (sol.times.back() < t - (1e-15 * std::abs(t))) {
    sol.times.push_back(t);
    sol.snapshots.push_back({sp.states, sp.probs});
    sol.state_space_sizes.push_back(sp.size());
  }

  diag.total_iters = iter;
  return {std::move(sol), std::move(diag)};
}

} // namespace fsp
