#include <cmath>
#include <iostream>
#include <markovkit.hpp>
#include <plot/plot.hpp>
#include <vector>

int main() {
  std::cout << "=== MarkovKit: Oregonator Stochastic CME Simulation & Plotting "
               "===\n\n";

  using namespace fsp;

  // 1. Define Oregonator Reaction System:
  // Reactions:
  //   1) Y -> X            (rate c1 = mu1 / y2)
  //   2) X + Y -> 0        (rate c2 = mu2 / (y1 * y2))
  //   3) X -> 2X + Z       (rate c3 = (mu1 + mu2) / y1)
  //   4) 2X -> 0           (rate c4 = 2 * mu1 / (y1^2))
  //   5) Z -> Y            (rate c5 = (mu2 - mu1) / y3)
  ReactionSystem sys;
  sys.stoich_vecs = {MultiIndex{1, -1, 0}, MultiIndex{-1, -1, 0},
                     MultiIndex{1, 0, 1}, MultiIndex{-2, 0, 0},
                     MultiIndex{0, 1, -1}};

  sys.propensities = {
      [](const MultiIndex &x, const std::vector<double> &rates, double t) {
        return rates[0] * x[1];
      },
      [](const MultiIndex &x, const std::vector<double> &rates, double t) {
        return rates[1] * x[0] * x[1];
      },
      [](const MultiIndex &x, const std::vector<double> &rates, double t) {
        return rates[2] * x[0];
      },
      [](const MultiIndex &x, const std::vector<double> &rates, double t) {
        return rates[3] * x[0] * (x[0] - 1.0) / 2.0;
      },
      [](const MultiIndex &x, const std::vector<double> &rates, double t) {
        return rates[4] * x[2];
      }};

  double y1 = 500.0, y2 = 1000.0, y3 = 2000.0;
  double mu1 = 2000.0, mu2 = 50000.0;
  std::vector<double> rates = {mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1,
                               2.0 * mu1 / (y1 * y1), (mu2 - mu1) / y3};

  // 2. Set Up FSP Problem
  FSPProblem prob;
  prob.model = sys;
  prob.rates = rates;
  prob.u0 = MultiIndex{500, 1000, 2000};
  prob.t0 = 0.0;
  prob.tf = 0.05;
  prob.bc = rect_boundary({1500, 2500, 4000});

  AdaptiveFSPParams params;
  params.eps_dt = 0.1;
  params.expansion_depth = 2;
  params.krylov_m = 30;
  params.krylov_tol = 1e-8;
  params.max_iter = 500;

  std::cout << "Solving Oregonator FSP from t=0.0 to t=" << prob.tf << "...\n";
  auto [sol, diag] = solve_adaptive_fsp(prob, params);
  std::cout << "Adaptive FSP completed in " << sol.t.size() << " steps.\n";

  // 3. Extract time series for plotting: dt(t), |S|(t), <X>(t), <Y>(t), <Z>(t)
  std::vector<double> time_points = sol.t;
  std::vector<double> state_sizes;
  for (int sz : sol.state_space_sizes)
    state_sizes.push_back(static_cast<double>(sz));

  std::vector<double> mean_X, mean_Y, mean_Z;
  for (const auto &snap : sol.snapshots) {
    double mx = 0.0, my = 0.0, mz = 0.0;
    for (std::size_t i = 0; i < snap.states.size(); ++i) {
      mx += snap.states[i][0] * snap.probs[i];
      my += snap.states[i][1] * snap.probs[i];
      mz += snap.states[i][2] * snap.probs[i];
    }
    mean_X.push_back(mx);
    mean_Y.push_back(my);
    mean_Z.push_back(mz);
  }

  // 4. Generate 6-panel figure using num::plt (gnuplot wrapper)
  num::plt::subplot(2, 3);

  // Panel 1: State space size |S|(t)
  num::plt::plot(time_points, state_sizes, "|S|", "lines lw 2");
  num::plt::title("(a) State Space Size |S|");
  num::plt::xlabel("Time");
  num::plt::ylabel("|S|");
  num::plt::next();

  // Panel 2: Mean X(t)
  num::plt::plot(time_points, mean_X, "<X>", "lines lw 2");
  num::plt::title("(b) Mean X Population");
  num::plt::xlabel("Time");
  num::plt::ylabel("<X>");
  num::plt::next();

  // Panel 3: Mean Y(t)
  num::plt::plot(time_points, mean_Y, "<Y>", "lines lw 2");
  num::plt::title("(c) Mean Y Population");
  num::plt::xlabel("Time");
  num::plt::ylabel("<Y>");
  num::plt::next();

  // Panel 4: X-Y Phase Projection
  num::plt::plot(mean_X, mean_Y, "X-Y", "lines lw 2");
  num::plt::title("(d) X-Y Phase Projection");
  num::plt::xlabel("<X>");
  num::plt::ylabel("<Y>");
  num::plt::next();

  // Panel 5: X-Z Phase Projection
  num::plt::plot(mean_X, mean_Z, "X-Z", "lines lw 2");
  num::plt::title("(e) X-Z Phase Projection");
  num::plt::xlabel("<X>");
  num::plt::ylabel("<Z>");
  num::plt::next();

  // Panel 6: Y-Z Phase Projection
  num::plt::plot(mean_Y, mean_Z, "Y-Z", "lines lw 2");
  num::plt::title("(f) Y-Z Phase Projection");
  num::plt::xlabel("<Y>");
  num::plt::ylabel("<Z>");

  std::cout
      << "Saving Oregonator simulation plot to 'oregonator_results.png'...\n";
  num::plt::savefig("oregonator_results.png");

  std::cout
      << "Oregonator simulation and plot generation completed successfully!\n";
  return 0;
}
