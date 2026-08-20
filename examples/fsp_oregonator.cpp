#include "markovkit.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

int main() {
  const double y1s = 500.0, y2s = 1000.0, y3s = 2000.0;
  const double mu1s = 2000.0, mu2s = 50000.0;

  const std::vector<double> rates = {
      mu1s / y2s,                  // k1 = 2.0
      mu2s / (y1s * y2s),          // k2 = 0.1
      (mu1s + mu2s) / y1s,         // k3 = 104.0
      2.0 * mu1s / (y1s * y1s),    // k4 = 0.016
      (mu1s + mu2s) / y3s,         // k5 = 26.0
  };

  fsp::ReactionSystem model;
  model.stoich_vecs = {
      {1, -1, 0},   // R1
      {-1, -1, 0},  // R2
      {1, 0, 1},    // R3
      {-2, 0, 0},   // R4
      {0, 1, -1},   // R5
  };
  model.propensities = {
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[0] * x[1]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[1] * x[0] * x[1]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[2] * x[0]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        return x[0] < 2 ? 0.0 : r[3] * x[0] * (x[0] - 1) * 0.5;
      },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[4] * x[2]; },
  };

  fsp::FSPProblem prob;
  prob.model = model;
  prob.u0 = {500, 1000, 2000};
  prob.t0 = 0.0;
  prob.tf = 2.0;
  prob.rates = rates;
  prob.bc = fsp::rect_boundary(50000);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Oregonator (Adaptive FSP) tf=" << prob.tf << "\n";

  fsp::AdaptiveFSPParams params;
  params.eps_dt = 1.0;
  params.prob_quantile = 0.1;
  params.flux_tolerance = 1.0;
  params.expansion_depth = 1;
  params.expand_method = fsp::ExpandMethod::ssa;
  params.save_interval = 5000;

  auto t0 = Clock::now();
  auto [sol, diag] = fsp::solve_adaptive_fsp(prob, params);
  double wall = std::chrono::duration<double>(Clock::now() - t0).count();

  auto means = sol.mean_trajectory();
  int fi = sol.n_snapshots() - 1;

  std::cout << "  iters=" << diag.total_iters << "  wall=" << wall << "s\n"
            << "  final |S|=" << sol.state_space_sizes[fi] << "\n"
            << "  t_reached=" << sol.t[fi] << "\n"
            << "  final means: <X>=" << means[fi][0] << " <Y>=" << means[fi][1]
            << " <Z>=" << means[fi][2] << "\n";

  return 0;
}
