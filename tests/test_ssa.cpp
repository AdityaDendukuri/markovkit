#include <cassert>
#include <iostream>
#include <markovkit.hpp>

void test_gillespie_direct() {
  using namespace fsp;
  using namespace markovkit::ssa;

  // Birth-death process: 0 -> X (rate 10), X -> 0 (rate 1 * X)
  ReactionSystem sys;
  sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
  sys.propensities = {
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        return r[0];
      },
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        return r[1] * static_cast<double>(x[0]);
      }};
  std::vector<double> rates = {10.0, 1.0};

  auto traj =
      gillespie_direct_simulate(sys, rates, MultiIndex{0}, 0.0, 5.0, 42);

  assert(traj.t.size() > 5);
  assert(traj.x.size() == traj.t.size());
  assert(traj.t.back() == 5.0);

  std::cout << "[PASS] Gillespie Direct Method SSA (" << traj.t.size()
            << " steps, final X = " << traj.x.back()[0] << ")\n";
}

void test_tau_leaping() {
  using namespace fsp;
  using namespace markovkit::ssa;

  ReactionSystem sys;
  sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
  sys.propensities = {
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        return r[0];
      },
      [](const MultiIndex &x, const std::vector<double> &r, double t) {
        return r[1] * static_cast<double>(x[0]);
      }};
  std::vector<double> rates = {10.0, 1.0};

  TauLeapingParams params;
  params.tau_fixed = 0.1;

  auto traj =
      tau_leaping_simulate(sys, rates, MultiIndex{0}, 0.0, 5.0, params, 42);

  assert(traj.t.size() >= 50);
  assert(traj.t.back() == 5.0);

  std::cout << "[PASS] Explicit Poisson Tau-Leaping (" << traj.t.size()
            << " steps, final X = " << traj.x.back()[0] << ")\n";
}

int main() {
  std::cout << "--- Running SSA & Tau-Leaping Suite Tests ---\n";
  test_gillespie_direct();
  test_tau_leaping();
  std::cout << "All SSA suite tests passed successfully!\n";
  return 0;
}
