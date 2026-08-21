#pragma once
#include "fsp/problem.hpp"
#include "fsp/solution.hpp"
#include <functional>
#include <tuple>
#include <vector>

namespace fsp {

/// Parameters for the mass-controlled Krylov-FSP method of Vo and Sidje.
struct KrylovFSPParams {
  double eps = 1e-6;              // error tolerance
  double tau_init = 1.0;          // initial step
  double droptol = 1e-10;         // probability pruning threshold
  double droptol_deriv = 1e-16;   // derivative pruning threshold
  double growth_factor = 2.0;     // tau multiplier upon acceptance
  int r = 1;                      // expansion depth
  int save_interval = 1000;
  int max_iter = 1 << 28;
  int krylov_m = 30;
  double krylov_tol = 1e-8;
  int log_interval = 0;

  // (accepted_steps, t, tau, size, mass)
  std::function<void(int, double, double, int, double)> progress_cb;
};

/// Accepted step sizes, retained mass, and rejected-step count.
struct KrylovFSPDiagnostics {
  std::vector<double> tau_log;
  std::vector<double> mass_log;
  std::vector<double> time_log;
  std::vector<int> size_log;
  int reject_count = 0;
  int total_iters = 0;
};

/// Solve the CME with adaptive support and Krylov exponential propagation.
std::tuple<FSPSolution, KrylovFSPDiagnostics>
solve_krylov_fsp(const FSPProblem& prob, const KrylovFSPParams& params = {});

} // namespace fsp
