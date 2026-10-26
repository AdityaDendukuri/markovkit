#pragma once
#include "core/types.hpp"
#include "fsp/problem.hpp"
#include "fsp/solution.hpp"
#include <cstdint>
#include <functional>
#include <tuple>

namespace fsp {

enum class expand_method : std::uint8_t { stoich, ssa };

/// Controls flux-based time steps and adaptive support management.
struct adaptive_fsp_params {
    double eps_dt = 1.0;          // dt = eps_dt / Phi
    double prob_quantile = 0.1;   // fraction of low-prob states checked for pruning
    double flux_tolerance = 1e-6; // relative flux threshold for pruning
    int expansion_depth = 1;      // neighbor shells per step
    expand_method expand_method = expand_method::stoich;
    int rng_seed = 42;
    int save_interval = 1000;
    int max_iter = 1 << 28;
    int krylov_m = 30;
    double krylov_tol = 1e-8;
    int log_interval = 0;
    double abs_threshold = 0.0; // absolute probability threshold (0 = off)

    // (iter, t, dt, size, per_species_means, pruned_mass)
    std::function<void(int, double, double, int, const num::vec<double> &, double)> progress_cb;

    // (t, states, probs)
    std::function<void(double, const num::array<markovkit::state> &, const num::array<double> &)>
        dist_cb;
};

/// Per-step data useful for checking support size and time-step selection.
struct adaptive_fsp_diagnostics {
    num::array<double> dt_log;
    num::array<double> t_log;
    num::array<int> size_log;
    num::array<double> flux_log;
    int total_iters = 0;
};

/// Evolve the CME while expanding and pruning its finite state space.
std::tuple<fsp_solution, adaptive_fsp_diagnostics>
solve_adaptive_fsp(const fsp_problem &prob, const adaptive_fsp_params &params = {});

} // namespace fsp
