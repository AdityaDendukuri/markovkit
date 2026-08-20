#include "markovkit.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_state_space_expand_compress() {
  fsp::ReactionSystem model;
  model.stoich_vecs = {{1}, {-1}};
  model.propensities = {
      [](const fsp::MultiIndex&, const std::vector<double>& r, double) { return r[0]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[1] * x[0]; }};

  fsp::StateSpace sp;
  sp.add_state({0}, 1.0);
  assert(sp.size() == 1);

  sp.expand(model, fsp::rect_boundary(10), 2);
  assert(sp.size() == 3); // 0, 1, 2

  std::vector<double> rates = {5.0, 0.1};
  auto [A, erb, diag] = fsp::build_generator(sp, model, rates, 0.0, false);
  assert(A.n_rows() == 3);
  assert(diag[0] == 5.0);

  auto [flux, n_removed] = sp.compress(model, rates, 0.0, 0.5, 1e-6);
  assert(sp.size() <= 3);

  double mass = sp.renormalize();
  assert(std::abs(mass - 1.0) < 1e-12);
  std::cout << "[PASS] test_state_space_expand_compress\n";
}

void test_adaptive_fsp_birth_death() {
  fsp::ReactionSystem model;
  model.stoich_vecs = {{1}, {-1}};
  model.propensities = {
      [](const fsp::MultiIndex&, const std::vector<double>& r, double) { return r[0]; },
      [](const fsp::MultiIndex& x, const std::vector<double>& r, double) { return r[1] * x[0]; }};

  fsp::FSPProblem prob;
  prob.model = model;
  prob.u0 = {0};
  prob.t0 = 0.0;
  prob.tf = 2.0;
  prob.rates = {5.0, 0.1};
  prob.bc = fsp::rect_boundary(100);

  fsp::AdaptiveFSPParams params;
  params.eps_dt = 0.1;
  params.flux_tolerance = 1e-6;

  auto [sol, diag] = fsp::solve_adaptive_fsp(prob, params);
  assert(sol.n_snapshots() > 1);

  auto means = sol.mean_trajectory();
  double final_mean = means.back()[0];
  // Analytical mean for birth-death with p0=0: (k_b / k_d) * (1 - exp(-k_d * t))
  // = (5.0 / 0.1) * (1 - exp(-0.1 * 2.0)) = 50 * (1 - 0.81873) = 9.063
  double expected_mean = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
  assert(std::abs(final_mean - expected_mean) < 0.1);
  std::cout << "[PASS] test_adaptive_fsp_birth_death (mean=" << final_mean
            << ", expected=" << expected_mean << ")\n";
}

int main() {
  test_state_space_expand_compress();
  test_adaptive_fsp_birth_death();
  std::cout << "All FSP tests passed.\n";
  return 0;
}
