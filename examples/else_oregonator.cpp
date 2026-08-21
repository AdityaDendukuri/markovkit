#include "markovkit.hpp"
#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t samples = 1000;
constexpr std::size_t simulation_steps = 100000;
constexpr num::idx subnetwork_capacity = 60;
constexpr double final_time = std::numeric_limits<double>::infinity();
constexpr std::array<const char*, 3> colors = {"#2980b9", "#c0392b", "#27ae60"};
constexpr std::array<const char*, 3> light_colors = {"#9ecae1", "#f4a6a0", "#a1d99b"};

using Curves = std::array<std::vector<double>, 3>;

std::vector<double> sample_times(double horizon) {
  std::vector<double> times(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    times[i] = horizon * static_cast<double>(i) / static_cast<double>(samples - 1);
  }
  return times;
}

Curves sample(const markovkit::Trajectory& path, const std::vector<double>& times) {
  Curves values;
  for (auto& species : values) {
    species.resize(times.size());
  }

  std::size_t state = 0;
  for (std::size_t i = 0; i < times.size(); ++i) {
    while (state + 1 < path.times.size() && path.times[state + 1] <= times[i]) {
      ++state;
    }
    for (std::size_t species = 0; species < values.size(); ++species) {
      values[species][i] = path.states[state][species];
    }
  }
  return values;
}

num::Series band(const std::vector<double>& times,
                 const std::vector<double>& lower,
                 const std::vector<double>& upper) {
  num::Series polygon;
  polygon.reserve(2 * times.size());
  for (std::size_t i = 0; i < times.size(); ++i) {
    polygon.emplace_back(times[i], upper[i]);
  }
  for (std::size_t i = times.size(); i-- > 0;) {
    polygon.emplace_back(times[i], lower[i]);
  }
  return polygon;
}

void plot_paths(const std::vector<markovkit::Trajectory>& paths,
                const std::string& title) {
  double first_end = std::numeric_limits<double>::infinity();
  double last_end = 0.0;
  for (const auto& path : paths) {
    first_end = std::min(first_end, path.times.back());
    last_end = std::max(last_end, path.times.back());
  }
  const auto times = sample_times(first_end);

  std::array<std::vector<double>, 3> lower;
  std::array<std::vector<double>, 3> upper;
  std::array<std::vector<double>, 3> mean;
  for (std::size_t species = 0; species < 3; ++species) {
    lower[species].assign(times.size(), std::numeric_limits<double>::max());
    upper[species].assign(times.size(), std::numeric_limits<double>::lowest());
    mean[species].assign(times.size(), 0.0);
  }

  for (const auto& path : paths) {
    const auto values = sample(path, times);
    for (std::size_t species = 0; species < 3; ++species) {
      for (std::size_t i = 0; i < times.size(); ++i) {
        lower[species][i] = std::min(lower[species][i], values[species][i]);
        upper[species][i] = std::max(upper[species][i], values[species][i]);
        mean[species][i] += values[species][i] / static_cast<double>(paths.size());
      }
    }
  }

  if (paths.size() > 1) {
    for (std::size_t species = 0; species < 3; ++species) {
      num::plt::plot(band(times, lower[species], upper[species]),
                     std::string{},
                     "filledcurves closed fs transparent solid 0.18 noborder lc rgb '" +
                         std::string(light_colors[species]) + "'");
    }
  }
  for (std::size_t species = 0; species < 3; ++species) {
    num::plt::plot(times,
                   mean[species],
                   std::string(1, static_cast<char>('X' + species)),
                   "lines lw 2 lc rgb '" + std::string(colors[species]) + "'");
  }
  std::ostringstream endpoint;
  endpoint << std::setprecision(3) << "  endpoint=[" << first_end << ", " << last_end << "]";
  num::plt::title(title + endpoint.str());
  num::plt::xlabel("t");
  num::plt::ylabel("count");
  num::plt::legend();
}

} // namespace

int main() {
  const double y1 = 500.0;
  const double y2 = 1000.0;
  const double y3 = 2000.0;
  const double mu1 = 2000.0;
  const double mu2 = 50000.0;
  const std::vector<double> rates = {
      mu1 / y2,
      mu2 / (y1 * y2),
      (mu1 + mu2) / y1,
      2.0 * mu1 / (y1 * y1),
      (mu1 + mu2) / y3,
  };

  markovkit::ReactionSystem model{
      .changes = {{1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-2, 0, 0}, {0, 1, -1}},
      .propensities = {
          [](const markovkit::State& x, const auto& r, double) { return r[0] * x[1]; },
          [](const markovkit::State& x, const auto& r, double) { return r[1] * x[0] * x[1]; },
          [](const markovkit::State& x, const auto& r, double) { return r[2] * x[0]; },
          [](const markovkit::State& x, const auto& r, double) {
            return x[0] < 2 ? 0.0 : 0.5 * r[3] * x[0] * (x[0] - 1);
          },
          [](const markovkit::State& x, const auto& r, double) { return r[4] * x[2]; },
      }};

  const markovkit::State initial{500, 1000, 2000};
  const auto one_else = else_sim::else_ensemble(model, rates, initial, 1, 0.0, final_time,
                                                 {.capacity = subnetwork_capacity,
                                                  .maximum_steps = simulation_steps},
                                                 42);
  const std::vector<markovkit::Trajectory> one_ssa{
      ssa::gillespie(model, rates, initial, 0.0, final_time, 42, simulation_steps)};
  const auto many_else = else_sim::else_ensemble(model, rates, initial, 20, 0.0, final_time,
                                                  {.capacity = subnetwork_capacity,
                                                   .maximum_steps = simulation_steps},
                                                  42);
  std::vector<markovkit::Trajectory> many_ssa;
  many_ssa.reserve(20);
  for (int seed = 42; seed < 62; ++seed) {
    many_ssa.push_back(
        ssa::gillespie(model, rates, initial, 0.0, final_time, seed, simulation_steps));
  }

  num::plt::subplot(2, 2);
  plot_paths(one_else, "ELSE: 1 trajectory");
  num::plt::next();
  plot_paths(one_ssa, "SSA: 1 trajectory");
  num::plt::next();
  plot_paths(many_else, "ELSE: 20 trajectories");
  num::plt::next();
  plot_paths(many_ssa, "SSA: 20 trajectories");
  num::plt::savefig("oregonator_else_ssa.png");
}
