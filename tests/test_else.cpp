#include "markovkit.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_subnetwork_moments_and_exit_law() {
  fsp::ReactionSystem model;
  model.stoich_vecs = {{1}, {-1}};
  model.propensities = {
      [](const fsp::MultiIndex&, const std::vector<double>& r, double) { return r[0]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[1] * x[0]; }};

  fsp::StateSpace sp;
  sp.add_state({0});
  sp.add_state({1});
  sp.add_state({2});

  std::vector<double> rates = {1.0, 1.0};
  else_sim::Subnetwork sub(sp, model, rates);

  num::Vector start(3, 0.0);
  start[0] = 1.0;
  auto u = sub.occupation(start);
  assert(u.size() == 3);
  assert(u[0] > 0.0);

  num::Matrix starts(3, 1, 0.0);
  starts(0, 0) = 1.0;
  auto m = sub.moments(starts);
  assert(m.size() == 3);

  auto p_exit = sub.escape_probabilities(m, 0);
  double sum_p = 0.0;
  for (double p : p_exit)
    sum_p += p;
  assert(std::abs(sum_p - 1.0) < 1e-10);

  // Conditional mean time is positive
  for (num::idx b : sub.boundary_states()) {
    double tau = sub.conditional_mean_time(m, b, 0);
    assert(tau >= 0.0);
  }

  std::cout << "[PASS] test_subnetwork_moments_and_exit_law\n";
}

void test_fourier_inversion() {
  const int modes = 24;
  const double period = 8.0;
  const double shift = 0.5;
  const double decay = 1.5;

  else_sim::FourierCoefficients coeffs(1, modes, period, shift);
  const auto shifts = else_sim::fourier_shifts(modes, period, shift);
  for (int m = 0; m <= modes; ++m) {
    coeffs(0, m) = 1.0 / (shifts[m] + decay);
  }

  auto inv = else_sim::invert_fourier(coeffs, 1.0);
  assert(inv.size() == 1);
  assert(std::abs(inv[0] - 1.0) < 1e-12);
  std::cout << "[PASS] test_fourier_inversion\n";
}

void test_common_talbot_density() {
  const std::vector<num::idx> rows{0, 0, 1, 1};
  const std::vector<num::idx> columns{0, 1, 0, 1};
  const std::vector<double> laplacian_values{1.0, -1.0, -1.0, 1.0};
  const auto laplacian = num::SparseMatrix::from_triplets(2, 2, rows, columns, laplacian_values);
  const std::vector<double> stationary_sqrt{1.0, 1.0};
  const std::vector<num::idx> states{0, 1};

  auto laplacian_step = else_sim::laplacian_density_step(laplacian, stationary_sqrt, states);

  else_sim::DensityStep cme_step;
  cme_step.states = {{0}, {1}};
  cme_step.index[cme_step.states[0]] = 0;
  cme_step.index[cme_step.states[1]] = 1;
  cme_step.generator =
      num::SparseMatrix::from_triplets(2, 2, rows, columns, {-1.0, 1.0, 1.0, -1.0});

  else_sim::TalbotDensitySolver laplacian_solver({std::move(laplacian_step)});
  else_sim::TalbotDensitySolver cme_solver({std::move(cme_step)});
  const auto from_laplacian = laplacian_solver.solve(else_sim::MultiIndex{0}, 0.5, 14);
  const auto from_cme = cme_solver.solve(else_sim::MultiIndex{0}, 0.5, 14);

  assert(from_laplacian.probability.size() == 2);
  assert(from_cme.probability.size() == 2);
  for (std::size_t state = 0; state < 2; ++state)
    assert(std::abs(from_laplacian.probability[state] - from_cme.probability[state]) < 1e-10);

  const double expected = 0.5 * (1.0 + std::exp(-1.0));
  assert(std::abs(from_laplacian.probability[0] - expected) < 1e-5);
  std::cout << "[PASS] test_common_talbot_density\n";
}

int main() {
  test_subnetwork_moments_and_exit_law();
  test_fourier_inversion();
  test_common_talbot_density();
  std::cout << "All ELSE tests passed.\n";
  return 0;
}
