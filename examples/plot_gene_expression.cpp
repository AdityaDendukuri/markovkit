#include <iostream>
#include <markovkit.hpp>
#include <plot/plot.hpp>
#include <vector>

int main() {
  std::cout << "=== MarkovKit: Plotting FSP Probability Distribution ===\n\n";

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
  BoundaryCondition bc = rect_boundary(100);
  states.expand(sys, bc, 40);

  // 3. Assemble Generator Matrix Q
  auto [Q, erb, gd] = build_generator(states, sys, rates, 0.0, true);

  // 4. Solve FSP via Krylov expv
  num::Vector p0((num::idx)states.size(), 0.0);
  p0[0] = 1.0;

  num::operators::SparseOp Qop(Q);
  double t_final = 20.0;
  num::Vector p_t = num::expv(t_final, Qop, p0, 30, 1e-8);

  // 5. Extract (mRNA_count, probability) series for plotting
  std::vector<double> mrna_counts;
  std::vector<double> probs;

  for (std::size_t i = 0; i < states.size(); ++i) {
    mrna_counts.push_back(states.states[i][0]);
    probs.push_back(p_t[(num::idx)i]);
  }

  // 6. Plot using num::plt (Matplotlib / NumPy style gnuplot wrapper)
  num::plt::plot(mrna_counts, probs, "P(mRNA, t=20)", "boxes fill solid 0.5");
  num::plt::title("FSP mRNA Probability Distribution (t = 20.0)");
  num::plt::xlabel("mRNA Copy Number");
  num::plt::ylabel("Probability");
  num::plt::legend();

  std::cout << "Saving plot to 'fsp_gene_expression.png'...\n";
  num::plt::savefig("fsp_gene_expression.png");

  std::cout << "Simulation & Plotting Completed Successfully.\n";
  return 0;
}
