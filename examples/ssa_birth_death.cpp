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

  auto exact_path = ssa::gillespie(model, rates, initial, 0.0, 10.0, 42);
  auto leap_path = ssa::tau_leaping(model, rates, initial, 0.0, 10.0, {}, 42);

  std::cout << "SSA Birth-Death:\n"
            << "  Gillespie exact events: " << exact_path.steps()
            << ", final count: " << exact_path.states.back()[0] << "\n"
            << "  Tau-leaping steps: " << leap_path.states.size()
            << ", final count: " << leap_path.states.back()[0] << "\n";

  return 0;
}
