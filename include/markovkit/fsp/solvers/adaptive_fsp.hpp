/// @file solvers/adaptive_fsp.hpp
/// @brief Adaptive FSP solver: flux-driven time stepping with state space management.
///
/// Implements the adaptive Finite State Projection algorithm:
///   1. Expand state space (stoichiometric neighbors)
///   2. Build CME generator
///   3. Compute adaptive step: dt = eps_dt / Phi_total
///   4. Propagate probability: p <- exp(dt*A) * p  (via Krylov expv)
///   5. Flux-aware pruning + renormalization
///   6. Repeat until t >= tf
#pragma once

#include "markovkit/fsp/problem.hpp"
#include "markovkit/fsp/solution.hpp"
#include <vector>
#include <tuple>
#include <functional>
#include <random>

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

    /// @brief Optional callback invoked every log_interval steps.
    /// Signature: (iter, t, dt, state_space_size, per_species_means, pruned_mass)
    std::function<void(int, double, double, int, const std::vector<double>&, double)> progress_cb;

    /// @brief Optional callback invoked every log_interval steps with full distribution.
    /// Signature: (t, states, probs)
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
///
/// @return (solution, diagnostics)
std::tuple<FSPSolution, AdaptiveFSPDiagnostics>
solve_adaptive_fsp(const FSPProblem& prob, const AdaptiveFSPParams& params = {});

} // namespace fsp
