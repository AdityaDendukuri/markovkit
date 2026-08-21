#include "markovkit.hpp"
#include <vector>

int main() {
  markovkit::ReactionSystem model;
  model.changes = {{1}, {-1}};
  model.propensities = {
      [](const markovkit::State&, const std::vector<double>& r, double) { return r[0]; },
      [](const markovkit::State& x, const std::vector<double>& r, double) { return r[1] * x[0]; }};

  const std::vector<double> rates = {5.0, 0.1};
  const markovkit::State initial{0};

  const auto path = else_sim::else_trajectory(model, rates, initial, 0.0, 20.0, {.capacity = 12});

  std::vector<double> counts;
  counts.reserve(path.states.size());
  for (const auto& state : path.states) {
    counts.push_back(state[0]);
  }

  num::plt::plot(path.times, counts, "ELSE", "steps lw 2");
  num::plt::xlabel("time");
  num::plt::ylabel("count");
  num::plt::savefig("else_birth_death.png");
}
