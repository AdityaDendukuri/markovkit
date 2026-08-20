#include "markovkit.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

int main() {
  fsp::ReactionSystem model;
  model.stoich_vecs = {
      {-1, -1, 1, 0}, // S + E -> C
      {1, 1, -1, 0},  // C -> S + E
      {0, 1, -1, 1},  // C -> P + E
  };
  model.propensities = {
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) {
        return r[0] * x[0] * x[1];
      },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[1] * x[2]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[2] * x[2]; },
  };

  const std::vector<double> rates = {0.01, 0.1, 0.1};
  const fsp::MultiIndex initial{50, 10, 0, 0};
  const double comparison_time = 10.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Michaelis-Menten: S(0)=50, E(0)=10, C(0)=0, P(0)=0 at t=" << comparison_time
            << "\n";

  // 1. Adaptive FSP Solve
  fsp::FSPProblem prob;
  prob.model = model;
  prob.u0 = initial;
  prob.t0 = 0.0;
  prob.tf = comparison_time;
  prob.rates = rates;
  prob.bc = fsp::rect_boundary({50, 50, 50, 50});

  fsp::AdaptiveFSPParams fsp_params;
  fsp_params.eps_dt = 0.1;
  fsp_params.flux_tolerance = 1e-10;
  auto [fsp_sol, fsp_diag] = fsp::solve_adaptive_fsp(prob, fsp_params);

  int fi = fsp_sol.n_snapshots() - 1;
  std::vector<std::map<int, double>> fsp_margs(4);
  for (int s = 0; s < 4; ++s) {
    fsp_margs[s] = fsp_sol.marginal(fi, s);
  }

  // 2. CME-specific step construction, followed by the shared Talbot solver.
  else_sim::ELSEOptions else_opts;
  else_opts.capacity = 60;
  else_opts.expansion_depth = 0;
  auto density_steps = else_sim::else_density(model, rates, initial, 30, else_opts);
  else_sim::TalbotDensitySolver density_solver(std::move(density_steps));
  const auto else_solution = density_solver.solve(initial, comparison_time, 32);

  std::vector<std::map<int, double>> else_margs(4);
  for (std::size_t i = 0; i < else_solution.states.size(); ++i) {
    for (int s = 0; s < 4; ++s) {
      else_margs[s][else_solution.states[i][s]] += else_solution.probability[i];
    }
  }

  // Print comparison means
  const char* species_names[] = {"S", "E", "C", "P"};
  for (int s = 0; s < 4; ++s) {
    double fsp_mean = 0.0, else_mean = 0.0;
    for (const auto& [k, v] : fsp_margs[s])
      fsp_mean += k * v;
    for (const auto& [k, v] : else_margs[s])
      else_mean += k * v;
    std::cout << "  <" << species_names[s] << ">: FSP=" << fsp_mean << " | ELSE=" << else_mean
              << "\n";
  }

  // Export marginals to CSV
  std::ofstream out("michaelis_menten_marginals.csv");
  out << "species,coord,count,fsp_prob,else_prob\n";
  for (int s = 0; s < 4; ++s) {
    std::set<int> all_counts;
    for (const auto& [k, v] : fsp_margs[s])
      all_counts.insert(k);
    for (const auto& [k, v] : else_margs[s])
      all_counts.insert(k);
    for (int count : all_counts) {
      double fsp_p = fsp_margs[s].count(count) ? fsp_margs[s][count] : 0.0;
      double else_p = else_margs[s].count(count) ? else_margs[s][count] : 0.0;
      out << species_names[s] << "," << s << "," << count << "," << fsp_p << "," << else_p << "\n";
    }
  }
  out.close();
  std::cout << "Saved marginals to michaelis_menten_marginals.csv\n";

  // Generate 2x2 comparison figure matching Pluto notebook
  FILE* gp = popen("gnuplot", "w");
  if (gp) {
    std::fprintf(gp, "set terminal pngcairo size 950,700 enhanced font 'Arial,10'\n");
    std::fprintf(gp, "set output 'michaelis_menten_comparison.png'\n");
    std::fprintf(gp, "set multiplot layout 2,2 title 'FSP vs ELSE at t=10.0 (Michaelis-Menten)'\n");
    std::fprintf(gp, "set datafile separator ','\n");
    std::fprintf(gp, "set ylabel 'probability'\n");
    std::fprintf(gp, "set boxwidth 0.8\n");
    std::fprintf(gp, "set style fill solid 0.45\n");

    for (int s = 0; s < 4; ++s) {
      std::fprintf(gp, "set xlabel '%s'\n", species_names[s]);
      std::fprintf(gp, "set title 'Marginal %s'\n", species_names[s]);
      std::fprintf(gp,
                   "plot 'michaelis_menten_marginals.csv' every ::1 using ($2==%d?$3:1/0):4 with "
                   "lines lw 2 lc rgb 'blue' title 'FSP', \\\n",
                   s);
      std::fprintf(gp,
                   "     '' every ::1 using ($2==%d?$3:1/0):5 with boxes lc rgb 'red' title "
                   "'inverted ELSE'\n",
                   s);
    }
    std::fprintf(gp, "unset multiplot\n");
    pclose(gp);
    std::cout << "Rendered michaelis_menten_comparison.png\n";
  }

  return 0;
}
