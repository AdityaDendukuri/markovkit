#include "markovkit.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

void test_subnetwork_moments_and_exit_law() {
  markovkit::ReactionSystem model;
  model.changes = {{1}, {-1}};
  model.propensities = {
      [](const markovkit::State&, const std::vector<double>& r, double) { return r[0]; },
      [](const markovkit::State& x, const std::vector<double>& r, double) { return r[1] * x[0]; }};

  fsp::StateSpace sp;
  sp.add_state({0});
  sp.add_state({1});
  sp.add_state({2});

  std::vector<double> rates = {1.0, 1.0};
  else_sim::Subnetwork sub(else_sim::restrict_cme(sp, model, rates));
  const std::vector<double> h{1.0, 1.0, std::sqrt(0.5)};
  else_sim::Subnetwork reversible(
      else_sim::reversible_generator(else_sim::restrict_cme(sp, model, rates), h));

  num::Vector start(3, 0.0);
  start[0] = 1.0;
  auto u = sub.occupation(start);
  assert(u.size() == 3);
  assert(u[0] > 0.0);
  const auto reversible_u = reversible.occupation(start);
  for (num::idx state = 0; state < u.size(); ++state) {
    assert(std::abs(reversible_u[state] - u[state]) < 1e-12);
  }

  num::Matrix starts(3, 2, 0.0);
  starts(0, 0) = 1.0;
  starts(1, 1) = 1.0;
  auto m = sub.occupation_moments(starts);
  const auto reversible_m = reversible.occupation_moments(starts);
  assert(m.size() == 3);
  for (num::idx row = 0; row < m.size(); ++row) {
    for (num::idx column = 0; column < m.starts(); ++column) {
      assert(std::abs(reversible_m.first(row, column) - m.first(row, column)) < 1e-12);
      assert(std::abs(reversible_m.second(row, column) - m.second(row, column)) < 1e-12);
    }
  }

  auto p_exit = sub.exit_probabilities(m, 0);
  double sum_p = 0.0;
  for (double p : p_exit)
    sum_p += p;
  assert(std::abs(sum_p - 1.0) < 1e-10);

  // Conditional mean time is positive
  for (num::idx b : sub.boundary_states()) {
    double tau = sub.conditional_exit_time(m, b, 0);
    assert(tau >= 0.0);
  }

  std::cout << "[PASS] test_subnetwork_moments_and_exit_law\n";
}

void test_exact_cut_time_losses() {
  // Reference 2-state system:
  // R = [-2  1;  1 -3], p0 = e1
  // -R = [2 -1; -1  3], det = 5
  // Z = (-R)^-1 = [0.6 0.2; 0.2 0.4]
  // u = Z * e1 = [0.6, 0.2]
  // Z^T 1 = [0.8, 0.6]
  // Delta T = [0.6 * (0.8 / 0.6), 0.2 * (0.6 / 0.4)] = [0.8, 0.3]
  fsp::StateSpace sp;
  sp.add_state({0});
  sp.add_state({1});

  markovkit::ReactionSystem model;
  model.changes = {{1}, {-1}, {100}}; // 100 is an exit transition
  model.propensities = {
      [](const markovkit::State& x, const std::vector<double>&, double) {
        return (x[0] == 0) ? 1.0 : 0.0; // 0 -> 1 rate 1
      },
      [](const markovkit::State& x, const std::vector<double>&, double) {
        return (x[0] == 1) ? 1.0 : 0.0; // 1 -> 0 rate 1
      },
      [](const markovkit::State& x, const std::vector<double>&, double) {
        return (x[0] == 0) ? 1.0 : 2.0; // 0 -> exit rate 1, 1 -> exit rate 2
      },
  };

  else_sim::Subnetwork sub(else_sim::restrict_cme(sp, model, {}));
  num::Vector p0(2, 0.0);
  p0[0] = 1.0;
  auto u = sub.occupation(p0);
  assert(std::abs(u[0] - 0.6) < 1e-12);
  assert(std::abs(u[1] - 0.2) < 1e-12);

  auto delta_t = sub.cut_time_losses(u);
  assert(delta_t.size() == 2);
  assert(std::abs(delta_t[0] - 0.8) < 1e-12);
  assert(std::abs(delta_t[1] - 0.3) < 1e-12);

  std::cout << "[PASS] test_exact_cut_time_losses (Delta T = [0.8, 0.3])\n";
}

void test_laplacian_column_sum_property() {
  // Non-uniform test case:
  // h = [1, 2], L = [4 -2; -2 1], L*h = 0
  // R = -diag(h) * L * diag(1/h) = [-4 1; 4 -1]
  const std::vector<num::idx> rows{0, 0, 1, 1};
  const std::vector<num::idx> cols{0, 1, 0, 1};
  const std::vector<double> vals{4.0, -2.0, -2.0, 1.0};
  const auto L = num::SparseMatrix::from_triplets(2, 2, rows, cols, vals);
  const std::vector<double> h{1.0, 2.0};

  auto step = else_sim::restrict_laplacian(L, h, {0, 1});
  const auto& R = step.generator;

  assert(std::abs(R(0, 0) - (-4.0)) < 1e-12);
  assert(std::abs(R(0, 1) - 1.0) < 1e-12);
  assert(std::abs(R(1, 0) - 4.0) < 1e-12);
  assert(std::abs(R(1, 1) - (-1.0)) < 1e-12);

  // Column sums are zero
  assert(std::abs(R(0, 0) + R(1, 0)) < 1e-12);
  assert(std::abs(R(0, 1) + R(1, 1)) < 1e-12);

  std::cout << "[PASS] test_laplacian_column_sum_property (non-uniform h=[1, 2])\n";
}

void test_laplacian_trajectory_uses_reversible_solver() {
  const auto laplacian = num::SparseMatrix::from_triplets(
      3,
      3,
      {0, 0, 1, 1, 1, 2, 2},
      {0, 1, 0, 1, 2, 1, 2},
      {1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0});
  const std::vector<double> h{1.0, 1.0, 1.0};
  const else_sim::ReversibleLaplacian generator(laplacian, h);
  const auto path = else_sim::laplacian_else_trajectory(
      generator,
      0,
      0.0,
      std::numeric_limits<double>::infinity(),
      {.capacity = 2, .maximum_steps = 3},
      7);

  assert(path.states.size() == 4);
  assert(path.times.size() == path.states.size());
  assert(path.times.back() > path.times.front());
}

void test_two_step_talbot_density() {
  // Two-step chain:
  // Step 0: State {0}, generator R = [-2], exit edge to {1} with rate 2.0
  // Step 1: State {1}, generator R = [-3], exit edge to {2} with rate 3.0
  else_sim::RestrictedGenerator step0;
  step0.states = {{0}};
  step0.index[step0.states[0]] = 0;
  step0.generator = num::SparseMatrix::from_triplets(1, 1, {0}, {0}, {-2.0});
  step0.boundary.push_back({0, {1}, 2.0});

  else_sim::RestrictedGenerator step1;
  step1.states = {{1}};
  step1.index[step1.states[0]] = 0;
  step1.generator = num::SparseMatrix::from_triplets(1, 1, {0}, {0}, {-3.0});
  step1.boundary.push_back({0, {2}, 3.0});

  else_sim::TalbotDensitySolver solver({std::move(step0), std::move(step1)});
  assert(solver.states().size() == 2);

  double t = 1.0;
  auto sol = solver.solve(markovkit::State{0}, t, 20);

  // Analytical solution for sequential transition 0 -> 1 -> 2:
  // p0(t) = exp(-2*t)
  // p1(t) = 2*(exp(-2*t) - exp(-3*t))
  double expected_p0 = std::exp(-2.0 * t);
  double expected_p1 = 2.0 * (std::exp(-2.0 * t) - std::exp(-3.0 * t));
  double total_mass = expected_p0 + expected_p1;
  double norm_p0 = expected_p0 / total_mass;
  double norm_p1 = expected_p1 / total_mass;

  assert(std::abs(sol.probability[0] - norm_p0) < 1e-4);
  assert(std::abs(sol.probability[1] - norm_p1) < 1e-4);

  std::cout << "[PASS] test_two_step_talbot_density\n";
}

markovkit::ReactionSystem birth_death_model() {
  return {
      .changes = {{1}, {-1}},
      .propensities = {
          [](const markovkit::State&, const std::vector<double>& rates, double) {
            return rates[0];
          },
          [](const markovkit::State& state, const std::vector<double>& rates, double) {
            return rates[1] * state[0];
          },
      },
  };
}

void test_single_trajectory_uses_ensemble() {
  const auto model = birth_death_model();
  const std::vector<double> rates{5.0, 0.1};
  const else_sim::ELSEOptions options{.capacity = 12};

  const auto single = else_sim::else_trajectory(model, rates, {0}, 2.0, 5.0, options, 17);
  const auto ensemble = else_sim::else_ensemble(model, rates, {0}, 1, 2.0, 5.0, options, 17);

  assert(single.times == ensemble[0].times);
  assert(single.states == ensemble[0].states);
  assert(single.times.front() == 2.0);
  assert(single.times.back() == 5.0);
  assert(single.times.size() == single.states.size());
}

void test_density_respects_capacity() {
  const auto subnetworks = else_sim::density_subnetworks(
      birth_death_model(), {5.0, 0.1}, {0}, 3, {.capacity = 6, .expansion_depth = 3});

  assert(!subnetworks.empty());
  for (const auto& subnetwork : subnetworks) {
    assert(subnetwork.states.size() <= 6);
  }
}

void test_trajectory_step_limit() {
  const auto path = else_sim::else_trajectory(birth_death_model(),
                                               {5.0, 0.1},
                                               {0},
                                               0.0,
                                               100.0,
                                               {.capacity = 12, .maximum_steps = 5},
                                               17);
  assert(path.states.size() == 6);
  assert(path.times.back() < 100.0);
}

int main() {
  test_subnetwork_moments_and_exit_law();
  test_exact_cut_time_losses();
  test_laplacian_column_sum_property();
  test_laplacian_trajectory_uses_reversible_solver();
  test_two_step_talbot_density();
  test_single_trajectory_uses_ensemble();
  test_density_respects_capacity();
  test_trajectory_step_limit();
  std::cout << "All ELSE tests passed.\n";
  return 0;
}
