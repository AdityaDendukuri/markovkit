#include "markovkit.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  // A 5-state 1D reversible line chain with uniform ground state h
  const num::idx n = 5;
  std::vector<double> h = {1.0, 1.0, 1.0, 1.0, 1.0};

  // Symmetric positive semi-definite graph Laplacian L (L*1 = 0)
  std::vector<num::idx> rows, cols;
  std::vector<num::real> vals;
  for (num::idx i = 0; i < n; ++i) {
    double deg = 0.0;
    if (i > 0) {
      rows.push_back(i);
      cols.push_back(i - 1);
      vals.push_back(-1.0);
      deg += 1.0;
    }
    if (i + 1 < n) {
      rows.push_back(i);
      cols.push_back(i + 1);
      vals.push_back(-1.0);
      deg += 1.0;
    }
    rows.push_back(i);
    cols.push_back(i);
    vals.push_back(deg);
  }
  auto L = num::SparseMatrix::from_triplets(n, n, rows, cols, vals);

  // Markov generator Q = -diag(h) * L * diag(1/h)
  std::vector<num::idx> q_rows, q_cols;
  std::vector<num::real> q_vals;
  for (num::idx i = 0; i < n; ++i) {
    for (num::idx k = L.row_ptr()[i]; k < L.row_ptr()[i + 1]; ++k) {
      num::idx j = L.col_idx()[k];
      double val = -h[i] * L.values()[k] * (1.0 / h[j]);
      q_rows.push_back(i);
      q_cols.push_back(j);
      q_vals.push_back(val);
    }
  }
  auto Q = num::SparseMatrix::from_triplets(n, n, q_rows, q_cols, q_vals);

  // Committor vector C (e.g. probability of reaching right boundary)
  num::Vector C(n, 0.0);
  for (num::idx i = 0; i < n; ++i) {
    C[i] = static_cast<double>(i) / static_cast<double>(n - 1);
  }

  // Propagate distribution from state 0 over time
  num::Vector p0(n, 0.0);
  p0[0] = 1.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Laplacian Committor Evolution (5 states):\n";
  std::cout << "  time   | committor C^T p(t)\n";
  std::cout << "  ---------------------------\n";

  for (double t : {0.0, 0.5, 1.0, 2.0, 5.0, 10.0}) {
    num::Vector p = (t == 0.0) ? p0 : num::expv(t, Q, p0);
    double c_val = 0.0;
    for (num::idx i = 0; i < n; ++i) c_val += C[i] * p[i];
    std::cout << "  t=" << std::setw(4) << t << " | E[C] = " << c_val << "\n";
  }

  return 0;
}
