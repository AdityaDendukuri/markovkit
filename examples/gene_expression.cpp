#include <iomanip>
#include <iostream>
#include <markovkit.hpp>

int main() {
  std::cout << "=== MarkovKit: Constitutive Gene Expression FSP Solve ===\n\n";

  using namespace fsp;

  // 1. Define Reaction System: DNA -> DNA + mRNA (k=5.0), mRNA -> 0 (gamma=0.1)
  ReactionSystem sys;
  sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
  sys.propensities = {[](const MultiIndex &x, const std::vector<double> &rates,
                         double t) { return rates[0]; },
                      [](const MultiIndex &x, const std::vector<double> &rates,
                         double t) { return rates[1] * x[0]; }};
  std::vector<double> rates = {5.0, 0.1};

  // 2. Set Up State Space
  StateSpace states;
  states.add_state(MultiIndex{0}, 1.0);

  BoundaryCondition bc = rect_boundary(50);

  states.expand(sys, bc, 20);
  std::cout << "Truncated State Space Size: " << states.size() << " states\n";

  // 3. Assemble Generator Matrix Q
  auto [Q, erb, gd] = build_generator(states, sys, rates, 0.0, true);
  std::cout << "Generator Matrix Assembled (" << Q.n_rows() << "x" << Q.n_cols()
            << " CSR)\n";

  // 4. Time Evolution via Krylov Arnoldi (num::expv)
  num::Vector p0((num::idx)states.size(), 0.0);
  p0[0] = 1.0; // Initial state mRNA = 0 with probability 1.0

  num::operators::SparseOp Qop(Q);
  double t_final = 10.0;
  num::Vector p_t = num::expv(t_final, Qop, p0, 30, 1e-8);

  double sum_p = 0.0;
  for (num::idx i = 0; i < p_t.size(); ++i)
    sum_p += p_t[i];

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Probability Sum at t=" << t_final << ": " << sum_p << "\n";
  std::cout << "P(mRNA = 0) at t=" << t_final << ": " << p_t[0] << "\n";

  std::cout << "\nMarkovKit FSP Simulation Completed Successfully.\n";
  return 0;
}
