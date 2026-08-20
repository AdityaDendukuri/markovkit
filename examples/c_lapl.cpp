#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "markovkit.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <class T> std::vector<T> json_array(const boost::json::value& value) {
  if (!value.is_array()) {
    throw std::invalid_argument("expected a JSON array");
  }
  std::vector<T> result;
  result.reserve(value.as_array().size());
  for (const auto& item : value.as_array()) {
    result.push_back(item.to_number<T>());
  }
  return result;
}

std::vector<std::vector<double>> json_matrix(const boost::json::value& value) {
  if (!value.is_array()) {
    throw std::invalid_argument("expected a JSON matrix");
  }
  std::vector<std::vector<double>> result;
  result.reserve(value.as_array().size());
  for (const auto& row : value.as_array()) {
    result.push_back(json_array<double>(row));
  }
  return result;
}

} // namespace

int main(int argc, char** argv) {
  std::string json_path = (argc > 1) ? argv[1] : "laplacians/cs-medium.json";

  const auto document = num::io::read_json(json_path);
  if (!document.is_object()) {
    throw std::invalid_argument("Laplacian document must be a JSON object");
  }
  const auto& object = document.as_object();
  const auto* laplacian_value = object.if_contains("L");
  const auto* h_value = object.if_contains("h");
  const auto* committors_value = object.if_contains("C");
  if (laplacian_value == nullptr || h_value == nullptr || committors_value == nullptr) {
    throw std::invalid_argument("Laplacian document requires L, h, and C fields");
  }

  const auto L = num::io::sparse_matrix(*laplacian_value);
  const num::idx full_n = L.n_rows();
  const auto h = json_array<double>(*h_value);
  const auto C_rows = json_matrix(*committors_value);
  std::cout << "Loaded " << json_path << " (n=" << full_n << ", nnz=" << L.nnz() << ")\n";

  // Keep the full Laplacian. Julia's `subnetwork_capacity` limits the
  // selected neighborhood; it does not truncate the input matrix.
  const auto& full_L = L;

  // Find start states: starts[j] = argmax_i C(i, j)
  const std::size_t n_committors = 5;
  std::vector<num::idx> starts(n_committors, 0);
  for (std::size_t j = 0; j < n_committors; ++j) {
    double max_c = -1.0;
    num::idx best_i = 0;
    for (num::idx i = 0; i < full_n; ++i) {
      if (C_rows[i][j] > max_c) {
        max_c = C_rows[i][j];
        best_i = i;
      }
    }
    starts[j] = best_i;
    std::cout << "  start[" << j << "] = " << best_i << " (max C=" << max_c << ")\n";
  }

  // Match the Julia Fourier/Talbot ELSE notebook.
  const int n_times = 41;
  const double t_min = 1e-12;
  const double t_max = 1e-4;
  std::vector<double> times(n_times);
  for (int i = 0; i < n_times; ++i) {
    double alpha = static_cast<double>(i) / (n_times - 1);
    times[i] = std::pow(10.0, std::log10(t_min) + alpha * (std::log10(t_max) - std::log10(t_min)));
  }

  const num::idx subnetwork_capacity =
      argc > 2 ? std::min<num::idx>(std::stoull(argv[2]), full_n) : 500;
  constexpr int number_of_steps = 2;

  // CSV output file for reproducing plots.
  std::ofstream out("c_lapl_curves.csv");
  out << "start_panel,time,C0,C1,C2,C3,C4\n";

  const bool full_laplacian = argc > 3 && std::string(argv[3]) == "full";
  if (full_laplacian) {
    std::vector<num::idx> all_states(full_n);
    std::iota(all_states.begin(), all_states.end(), 0);
    std::vector<else_sim::DensityStep> steps;
    steps.push_back(else_sim::laplacian_density_step(L, h, all_states));
    else_sim::TalbotDensitySolver solver(std::move(steps));
    std::vector<else_sim::MultiIndex> initial_states;
    initial_states.reserve(starts.size());
    for (num::idx start : starts) {
      initial_states.push_back(else_sim::MultiIndex{static_cast<int>(start)});
    }

    for (std::size_t time_index = 0; time_index < times.size(); ++time_index) {
      const auto solutions = solver.solve(initial_states, times[time_index], 14);
      for (std::size_t panel = 0; panel < solutions.size(); ++panel) {
        std::array<double, 5> values{};
        for (num::idx i = 0; i < solutions[panel].states.size(); ++i) {
          const auto global = static_cast<num::idx>(solutions[panel].states[i][0]);
          for (std::size_t j = 0; j < n_committors; ++j) {
            values[j] += C_rows[global][j] * solutions[panel].probability[i];
          }
        }
        out << panel << "," << times[time_index];
        for (double value : values) {
          out << "," << value;
        }
        out << "\n";
      }
    }
  } else {
    for (std::size_t panel = 0; panel < starts.size(); ++panel) {
      auto steps = else_sim::laplacian_density_steps(
          L, h, starts[panel], subnetwork_capacity, number_of_steps);
      else_sim::TalbotDensitySolver solver(std::move(steps));
      const else_sim::MultiIndex initial{static_cast<int>(starts[panel])};

      for (double t : times) {
        const auto solution = solver.solve(initial, t, 14);
        std::vector<double> c_vals(n_committors, 0.0);
        for (num::idx i = 0; i < solution.probability.size(); ++i) {
          const num::idx global = static_cast<num::idx>(solution.states[i][0]);
          for (std::size_t j = 0; j < n_committors; ++j) {
            c_vals[j] += C_rows[global][j] * solution.probability[i];
          }
        }
        out << panel << "," << t;
        for (std::size_t j = 0; j < n_committors; ++j) {
          out << "," << c_vals[j];
        }
        out << "\n";
      }
    }
  }
  out.close();
  std::cout << "Saved committor curves to c_lapl_curves.csv\n";

  // Generate PNG figure via gnuplot
  FILE* gp = popen("gnuplot", "w");
  if (gp) {
    std::fprintf(gp, "set terminal pngcairo size 1100,650 enhanced font 'Arial,10'\n");
    std::fprintf(gp, "set output 'c_lapl_evolution.png'\n");
    std::fprintf(gp,
                 "set multiplot layout 2,3 title 'Laplacian Committor Evolution (cs-medium)'\n");
    std::fprintf(gp, "set datafile separator ','\n");
    std::fprintf(gp, "set logscale x\n");
    std::fprintf(gp, "set xrange [1e-12:1e-4]\n");
    std::fprintf(gp, "set yrange [-0.02:1.02]\n");
    std::fprintf(gp, "set xlabel 't'\n");
    std::fprintf(gp, "set ylabel 'C^T p(t)'\n");

    for (int panel = 0; panel < 5; ++panel) {
      std::fprintf(gp, "set title 'start = argmax(C%d)'\n", panel + 1);
      if (panel == 0) {
        std::fprintf(gp, "set key right top\n");
      } else {
        std::fprintf(gp, "unset key\n");
      }
      std::fprintf(gp,
                   "plot 'c_lapl_curves.csv' every ::1 using ($1==%d?$2:1/0):3 with lines lw 2 "
                   "title 'C1', \\\n",
                   panel);
      std::fprintf(
          gp, "     '' every ::1 using ($1==%d?$2:1/0):4 with lines lw 2 title 'C2', \\\n", panel);
      std::fprintf(
          gp, "     '' every ::1 using ($1==%d?$2:1/0):5 with lines lw 2 title 'C3', \\\n", panel);
      std::fprintf(
          gp, "     '' every ::1 using ($1==%d?$2:1/0):6 with lines lw 2 title 'C4', \\\n", panel);
      std::fprintf(
          gp, "     '' every ::1 using ($1==%d?$2:1/0):7 with lines lw 2 title 'C5'\n", panel);
    }
    std::fprintf(gp, "unset multiplot\n");
    pclose(gp);
    std::cout << "Rendered c_lapl_evolution.png\n";
  }

  return 0;
}
