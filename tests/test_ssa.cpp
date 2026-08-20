#include "markovkit.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_gillespie_birth_death() {
  fsp::ReactionSystem model;
  model.stoich_vecs = {{1}, {-1}};
  model.propensities = {
      [](const fsp::MultiIndex&, const std::vector<double>& r, double) { return r[0]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[1] * x[0]; }};

  std::vector<double> rates = {5.0, 0.1};
  fsp::MultiIndex initial{0};

  const int paths = 2000;
  double mean = 0.0;
  for (int i = 0; i < paths; ++i) {
    auto path = ssa::gillespie(model, rates, initial, 0.0, 2.0, 100 + i);
    mean += path.states.back()[0];
  }
  mean /= paths;

  double expected = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
  assert(std::abs(mean - expected) < 0.5);
  std::cout << "[PASS] test_gillespie_birth_death (mean=" << mean << ", expected=" << expected << ")\n";
}

int main() {
  test_gillespie_birth_death();
  std::cout << "All SSA tests passed.\n";
  return 0;
}
