/// @file toggle_switch_example.cpp
/// @brief Reproduction of the Tian-Burrage Genetic Toggle Switch Rare-Event Trajectory Figure:
/// Direct SSA (Trapped in initial basin) vs Multiple ELSE (Visiting low-probability V-high basin).
#include <cmath>
#include <iostream>
#include <markovkit.hpp>
#include <plot/plot.hpp>
#include <vector>

int main() {
  std::cout << "=== MarkovKit: Tian-Burrage Genetic Toggle Switch Trajectory Comparison ===\n";
  std::cout << "Comparing Direct SSA vs Multiple ELSE in 2D State Space (U, V)...\n\n";

  using namespace fsp;
  using namespace markovkit::else_solver;
  using namespace markovkit::ssa;

  // 1. Define Tian-Burrage Genetic Toggle Switch Reaction System:
  // Reactions:
  //   1) 0 -> U  (propensity: eta * (alpha1 + beta1 * K1^3 / (K1^3 + V^3)))
  //   2) U -> 0  (propensity: (d1 + dU) * U)
  //   3) 0 -> V  (propensity: eta * (alpha2 + beta2 * K2^3 / (K2^3 + U^3)))
  //   4) V -> 0  (propensity: d2 * V)
  ReactionSystem sys;
  sys.stoich_vecs = {MultiIndex{1, 0}, MultiIndex{-1, 0}, MultiIndex{0, 1}, MultiIndex{0, -1}};

  double scale_factor = 0.2;
  std::vector<double> rates = {
      1.0,                    // eta
      20.0 * scale_factor,    // alpha1 = 4.0
      400.0 * scale_factor,   // beta1  = 80.0
      100.0 * scale_factor,   // K1     = 20.0
      1.0,                    // d1
      0.1 / 1.1,              // dU
      20.0 * scale_factor,    // alpha2 = 4.0
      400.0 * scale_factor,   // beta2  = 80.0
      100.0 * scale_factor,   // K2     = 20.0
      1.0                     // d2
  };

  sys.propensities = {
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        double v = static_cast<double>(x[1]);
        double k3 = r[3] * r[3] * r[3];
        return r[0] * (r[1] + r[2] * k3 / (k3 + v * v * v));
      },
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        return (r[4] + r[5]) * static_cast<double>(x[0]);
      },
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        double u = static_cast<double>(x[0]);
        double k8 = r[8] * r[8] * r[8];
        return r[0] * (r[6] + r[7] * k8 / (k8 + u * u * u));
      },
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        return r[9] * static_cast<double>(x[1]);
      }};

  MultiIndex initial_state{17, 1};
  double tf = 500.0;

  // 2. Direct SSA Simulation
  std::cout << "1. Simulating Direct SSA (Gillespie) trajectory to tf=" << tf << "...\n";
  SSATrajectory ssa_traj = gillespie_direct_simulate(sys, rates, initial_state, 0.0, tf, 66);
  std::cout << "   Direct SSA completed in " << ssa_traj.t.size() << " steps.\n";

  std::vector<double> ssa_U, ssa_V;
  for (const auto &state : ssa_traj.x) {
    ssa_U.push_back(static_cast<double>(state[0]));
    ssa_V.push_back(static_cast<double>(state[1]));
  }

  // 3. Multiple ELSE (Expected-Visits Low-Probability State Extrapolation) Simulation
  std::cout << "2. Simulating Multiple ELSE trajectory across rare saddle region to tf=" << tf << "...\n";
  ELSETrajectory else_traj = else_trajectory_simulate(sys, rates, initial_state, 0.0, tf, 3, 42);
  std::cout << "   Multiple ELSE completed in " << else_traj.number_of_else_steps << " ELSE steps ("
            << else_traj.t.size() << " trajectory points).\n";

  std::vector<double> else_U, else_V;
  for (const auto &state : else_traj.x) {
    else_U.push_back(static_cast<double>(state[0]));
    else_V.push_back(static_cast<double>(state[1]));
  }

  // Basin attractors for reference
  std::vector<double> start_U{17.0}, start_V{1.0};
  std::vector<double> basin_U_high_U{76.0}, basin_U_high_V{5.0};
  std::vector<double> basin_V_high_U{5.0}, basin_V_high_V{83.0};

  auto build_figure = [&]() {
    num::plt::subplot(1, 2);

    // Panel (a): Direct SSA
    num::plt::plot(ssa_U, ssa_V, "SSA Path", "lines lw 1");
    num::plt::plot(start_U, start_V, "Start (17,1)", "points pt 7 ps 1.2");
    num::plt::plot(basin_U_high_U, basin_U_high_V, "U-high (76,5)", "points pt 9 ps 1.2");
    num::plt::plot(basin_V_high_U, basin_V_high_V, "V-high (5,83)", "points pt 13 ps 1.2");
    num::plt::title("(a) Direct SSA: Trapped in Initial Basin");
    num::plt::xlabel("U Molecule Count");
    num::plt::ylabel("V Molecule Count");
    num::plt::xlim(0, 105);
    num::plt::ylim(0, 105);
    num::plt::legend();
    num::plt::next();

    // Panel (b): Shared Expected-Visits ELSE
    num::plt::plot(else_U, else_V, "ELSE Path", "lines lw 1");
    num::plt::plot(start_U, start_V, "Start (17,1)", "points pt 7 ps 1.2");
    num::plt::plot(basin_U_high_U, basin_U_high_V, "U-high (76,5)", "points pt 9 ps 1.2");
    num::plt::plot(basin_V_high_U, basin_V_high_V, "V-high (5,83)", "points pt 13 ps 1.2");
    num::plt::title("(b) Expected-Visits ELSE: Visits Low-Prob Basin");
    num::plt::xlabel("U Molecule Count");
    num::plt::ylabel("V Molecule Count");
    num::plt::xlim(0, 105);
    num::plt::ylim(0, 105);
    num::plt::legend();
  };

  // Save real high-resolution PNG figure
  std::cout << "Saving high-resolution figure to 'toggle_switch_results.png'...\n";
  build_figure();
  num::plt::savefig("toggle_switch_results.png");

  // Render ASCII terminal view
  std::cout << "\nRendering 2D State Space Trajectory Comparison in ASCII (140x35)...\n\n";
  build_figure();
  num::plt::show_dumb(140, 35);

  std::cout << "\nTian-Burrage Toggle Switch Rare-Event Trajectory Reproduction Completed Successfully!\n";
  return 0;
}
