#include "markovkit.hpp"
#include <iostream>

int main() {
  fsp::ReactionSystem model;
  model.stoich_vecs = {{1}, {-1}};
  model.propensities = {
      [](const fsp::MultiIndex&, const std::vector<double>& r, double) { return r[0]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[1] * x[0]; }};

  const std::vector<double> rates = {5.0, 0.1};
  const fsp::MultiIndex initial{0};

  auto traj = else_sim::else_trajectory(model, rates, initial, 0.0, 20.0, {.capacity = 12});

  std::cout << "Birth-Death (ELSE Trajectory): steps=" << traj.states.size() << "\n";
  for (std::size_t i = 0; i < std::min(traj.states.size(), static_cast<std::size_t>(10)); ++i) {
    std::cout << "  t=" << traj.t[i] << " count=" << traj.states[i][0] << "\n";
  }

  return 0;
}
