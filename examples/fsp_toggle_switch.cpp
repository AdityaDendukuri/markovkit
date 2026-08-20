#include "markovkit.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

int main() {
  const double alpha = 20.0;
  const double beta = 400.0;
  const double K = 100.0;
  const double d_U = 1.0 + 0.1 / 1.1;
  const double d_V = 1.0;
  const double K3 = K * K * K;

  const std::vector<double> rates = {alpha, beta, K3, d_U, alpha, beta, K3, d_V};

  fsp::ReactionSystem model;
  model.stoich_vecs = {
      {1, 0},   // R1: 0 -> U
      {-1, 0},  // R2: U -> 0
      {0, 1},   // R3: 0 -> V
      {0, -1},  // R4: V -> 0
  };
  model.propensities = {
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        double v3 = static_cast<double>(x[1]) * x[1] * x[1];
        return r[0] + r[1] * r[2] / (r[2] + v3);
      },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        return r[3] * x[0];
      },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        double u3 = static_cast<double>(x[0]) * x[0] * x[0];
        return r[4] + r[5] * r[6] / (r[6] + u3);
      },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        return r[7] * x[1];
      },
  };

  fsp::FSPProblem prob;
  prob.model = model;
  prob.u0 = {100, 100};
  prob.t0 = 0.0;
  prob.tf = 10.0;
  prob.rates = rates;
  prob.bc = fsp::rect_boundary(500);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Toggle switch (Adaptive FSP) tf=" << prob.tf << "\n";

  fsp::AdaptiveFSPParams params;
  params.eps_dt = 10.0;
  params.prob_quantile = 0.0;
  params.flux_tolerance = 0.0;
  params.abs_threshold = 1e-6;
  params.expansion_depth = 5;
  params.save_interval = 200;

  auto t0 = Clock::now();
  auto [sol, diag] = fsp::solve_adaptive_fsp(prob, params);
  double wall = std::chrono::duration<double>(Clock::now() - t0).count();

  auto means = sol.mean_trajectory();
  int fi = sol.n_snapshots() - 1;

  std::cout << "  iters=" << diag.total_iters << "  wall=" << wall << "s\n"
            << "  final |S|=" << sol.state_space_sizes[fi] << "\n"
            << "  final means: <U>=" << means[fi][0] << " <V>=" << means[fi][1] << "\n";

  return 0;
}
