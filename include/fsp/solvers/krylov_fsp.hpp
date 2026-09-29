#pragma once
#include "core/types.hpp"
#include "fsp/problem.hpp"
#include "fsp/solution.hpp"
#include <functional>
#include <tuple>

namespace fsp {

/// Parameters for the mass-controlled Krylov-FSP method of Vo and Sidje.
struct krylov_fsp_params {
    double eps = 1e-6;            // error tolerance
    double tau_init = 1.0;        // initial step
    double droptol = 1e-10;       // probability pruning threshold
    double droptol_deriv = 1e-16; // derivative pruning threshold
    double growth_factor = 2.0;   // tau multiplier upon acceptance
    int r = 1;                    // expansion depth
    int save_interval = 1000;
    int max_iter = 1 << 28;
    int krylov_m = 30;
    double krylov_tol = 1e-8;
    int log_interval = 0;

    // (accepted_steps, t, tau, size, mass)
    std::function<void(int, double, double, int, double)> progress_cb;
};

/// Accepted step sizes, retained mass, and rejected-step count.
struct krylov_fsp_diagnostics {
    num::array<double> tau_log;
    num::array<double> mass_log;
    num::array<double> time_log;
    num::array<int> size_log;
    int reject_count = 0;
    int total_iters = 0;
};

/// Solve the CME with adaptive support and Krylov exponential propagation.
std::tuple<fsp_solution, krylov_fsp_diagnostics>
solve_krylov_fsp(const fsp_problem &prob, const krylov_fsp_params &params = {});

} // namespace fsp
