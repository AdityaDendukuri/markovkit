/// @file solvers/adaptive_fsp.hpp
/// @brief Adaptive FSP solver: flux-driven time stepping with state space management.
#pragma once

#include "markovkit/fsp/problem.hpp"
#include "markovkit/fsp/solution.hpp"
#include "markovkit/fsp/state_space.hpp"
#include "markovkit/fsp/generator.hpp"
#include <numerics.hpp>
#include <vector>
#include <tuple>
#include <functional>
#include <random>
#include <optional>
#include <cmath>
#include <algorithm>

namespace fsp {

enum class ExpandMethod { stoich, ssa };

/// @brief Parameters for the AdaptiveFSP solver.
struct AdaptiveFSPParams {
    double eps_dt          = 1.0;     ///< time-step control: dt = eps_dt / Phi
    double prob_quantile   = 0.1;     ///< fraction of low-prob states considered for pruning
    double flux_tolerance  = 1e-6;    ///< relative flux threshold for pruning
    int          expansion_depth = 1;          ///< shells of neighbors added per step
    ExpandMethod expand_method  = ExpandMethod::stoich; ///< stoich=full, ssa=SSA-guided
    int          rng_seed       = 42;          ///< seed for SSA expansion RNG
    int          save_interval  = 1000;        ///< save snapshot every N steps
    int    max_iter        = 1 << 28; ///< iteration limit
    int    krylov_m        = 30;      ///< max Krylov dimension for expv
    double krylov_tol      = 1e-8;    ///< Krylov convergence tolerance
    int    log_interval    = 0;       ///< print progress every N iters (0 = off)
    double abs_threshold   = 0.0;    ///< absolute probability threshold for pruning (0 = off)

    std::function<void(int, double, double, int, const std::vector<double>&, double)> progress_cb;
    std::function<void(double, const std::vector<MultiIndex>&, const std::vector<double>&)> dist_cb;
};

/// @brief Per-step diagnostics from an AdaptiveFSP solve.
struct AdaptiveFSPDiagnostics {
    std::vector<double> dt_log;    ///< step sizes taken
    std::vector<double> t_log;     ///< simulation time after each step
    std::vector<int>    size_log;  ///< |S| after each step
    std::vector<double> flux_log;  ///< Phi = -dot(diag(A), p) per step
    int                 total_iters;
};

/// @brief Solve the CME with the adaptive FSP algorithm.
inline std::tuple<FSPSolution, AdaptiveFSPDiagnostics>
solve_adaptive_fsp(const FSPProblem& prob, const AdaptiveFSPParams& params = {})
{
    const ReactionSystem&      model  = prob.model;
    const std::vector<double>& rates  = prob.rates;
    const BoundaryCondition&   bc     = prob.bc;
    double t0 = prob.t0;
    double tf  = prob.tf;
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
    std::vector<double>       erb_cached;
    std::vector<double>       diag_cached;
    std::vector<double>       exit_cached;
    bool                      gen_valid = false;

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
            A_cached    = std::move(A);
            erb_cached  = std::move(erb);
            diag_cached = std::move(gd);

            int n = sp.size();
            exit_cached.resize(n);
            for (int i = 0; i < n; ++i)
                exit_cached[i] = diag_cached[i] + erb_cached[i];

            sp.dirty  = false;
            gen_valid = true;
        }

        int n = sp.size();
        double Phi = 0.0;
        for (int i = 0; i < n; ++i)
            Phi += sp.probs[i] * diag_cached[i];

        double dt_flux = (Phi > 0.0) ? params.eps_dt / Phi : (tf - t);
        double dt = std::min(dt_flux, tf - t);

        num::Vector p_in((num::idx)n, 0.0);
        for (int i = 0; i < n; ++i) p_in[(num::idx)i] = sp.probs[i];

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
        for (double p : sp.probs) sum_p += p;
        if (sum_p > 0.0) {
            for (double& p : sp.probs) p /= sum_p;
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
