/// @file solvers/krylov_fsp.hpp
/// @brief Krylov-FSP solver: Algorithm 2 of Vo & Sidje 2016.
///
/// Features:
///   - Absorbing-form generator (probability leaks out at boundary)
///   - Time-scaled FSP acceptance criterion: sum(w) >= 1 - eps*(t+tau)/tf
///   - Adaptive step: grow by growth_factor on success, halve on rejection
///   - Joint probability + derivative pruning with auto-droptol reduction
///   - SSA-guided + r-step reachability expansion
#pragma once

#include "markovkit/fsp/problem.hpp"
#include "markovkit/fsp/solution.hpp"
#include <vector>
#include <tuple>
#include <functional>

namespace fsp {

/// @brief Parameters for the KrylovFSP solver.
struct KrylovFSPParams {
    double eps           = 1e-6;   ///< FSP tolerance eps_FSP
    double tau_init      = 1.0;    ///< initial time step
    double droptol       = 1e-10;  ///< initial probability threshold for pruning
    double droptol_deriv = 1e-16;  ///< derivative threshold for joint filter
    double growth_factor = 2.0;    ///< tau multiplier after successful step
    int    r             = 1;      ///< reachability expansion depth
    int    save_interval = 1000;   ///< save snapshot every N accepted steps
    int    max_iter      = 1 << 28;
    int    krylov_m      = 30;     ///< max Krylov dimension for expv
    double krylov_tol    = 1e-8;   ///< Krylov convergence tolerance
    int    log_interval  = 0;      ///< print progress every N accepted steps (0 = off)

    /// @brief Optional callback invoked every log_interval accepted steps.
    /// Signature: (accepted_steps, t, tau, state_space_size, mass)
    std::function<void(int, double, double, int, double)> progress_cb;
};

/// @brief Per-step diagnostics from a KrylovFSP solve.
struct KrylovFSPDiagnostics {
    std::vector<double> tau_log;      ///< time steps (including rejected)
    std::vector<double> mass_log;     ///< probability mass after each step
    std::vector<double> time_log;     ///< simulation time after each step
    std::vector<int>    size_log;     ///< |S| after each step
    int                 reject_count;
    int                 total_iters;
};

/// @brief Solve the CME with the Krylov-FSP algorithm (Vo & Sidje 2016).
///
/// @return (solution, diagnostics)
std::tuple<FSPSolution, KrylovFSPDiagnostics>
solve_krylov_fsp(const FSPProblem& prob, const KrylovFSPParams& params = {});

} // namespace fsp
