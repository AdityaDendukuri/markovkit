#include "markovkit.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

int main() {
  const double y1s = 500.0, y2s = 1000.0, y3s = 2000.0;
  const double mu1s = 2000.0, mu2s = 50000.0;

  const std::vector<double> rates = {
      mu1s / y2s,
      mu2s / (y1s * y2s),
      (mu1s + mu2s) / y1s,
      2.0 * mu1s / (y1s * y1s),
      (mu1s + mu2s) / y3s,
  };

  fsp::ReactionSystem model;
  model.stoich_vecs = {
      {1, -1, 0},
      {-1, -1, 0},
      {1, 0, 1},
      {-2, 0, 0},
      {0, 1, -1},
  };
  model.propensities = {
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[0] * x[1]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        return r[1] * x[0] * x[1];
      },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[2] * x[0]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        return x[0] < 2 ? 0.0 : r[3] * x[0] * (x[0] - 1) * 0.5;
      },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[4] * x[2]; },
  };

  const fsp::MultiIndex initial{500, 1000, 2000};
  std::size_t num_trajectories = 50;
  double tf = 2.0;

  std::cout << "Oregonator (ELSE Shared-Subnetwork Ensemble: " << num_trajectories << " paths)\n";

  auto t0 = Clock::now();
  auto ensemble =
      else_sim::else_ensemble(model, rates, initial, num_trajectories, 0.0, tf, {.capacity = 60});
  double wall = std::chrono::duration<double>(Clock::now() - t0).count();

  std::cout << "  wall=" << wall << "s\n";
  std::cout << "  trajectories simulated=" << ensemble.paths.size() << "\n";

  return 0;
}
