#pragma once
#include "fsp/problem.hpp"
#include "fsp/solution.hpp"
#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

namespace fsp {

enum class ExpandMethod : std::uint8_t { stoich, ssa };

/// Controls flux-based time steps and adaptive support management.
struct AdaptiveFSPParams {
    double eps_dt = 1.0;          // dt = eps_dt / Phi
    double prob_quantile = 0.1;   // fraction of low-prob states checked for pruning
    double flux_tolerance = 1e-6; // relative flux threshold for pruning
    int expansion_depth = 1;      // neighbor shells per step
    ExpandMethod expand_method = ExpandMethod::stoich;
    int rng_seed = 42;
    int save_interval = 1000;
    int max_iter = 1 << 28;
    int krylov_m = 30;
    double krylov_tol = 1e-8;
    int log_interval = 0;
    double abs_threshold = 0.0; // absolute probability threshold (0 = off)

    // (iter, t, dt, size, per_species_means, pruned_mass)
    std::function<void(int, double, double, int, const std::vector<double> &, double)> progress_cb;

    // (t, states, probs)
    std::function<void(double, const std::vector<markovkit::State> &, const std::vector<double> &)>
        dist_cb;
};

/// Per-step data useful for checking support size and time-step selection.
struct AdaptiveFSPDiagnostics {
    std::vector<double> dt_log;
    std::vector<double> t_log;
    std::vector<int> size_log;
    std::vector<double> flux_log;
    int total_iters = 0;
};

/// Evolve the CME while expanding and pruning its finite state space.
std::tuple<FSPSolution, AdaptiveFSPDiagnostics>
solve_adaptive_fsp(const FSPProblem &prob, const AdaptiveFSPParams &params = {});

} // namespace fsp
