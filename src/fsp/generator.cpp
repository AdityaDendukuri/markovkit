#include "fsp/generator.hpp"
#include <vector>

namespace fsp {

GeneratorResult build_generator(const StateSpace& sp,
                                const ReactionSystem& model,
                                const std::vector<double>& rates,
                                double t,
                                bool absorbing) {
  int n = sp.size();

  std::vector<num::idx> rows, cols;
  std::vector<num::real> vals;
  rows.reserve(static_cast<std::size_t>(n) * (model.n_reactions() + 1));
  cols.reserve(rows.capacity());
  vals.reserve(rows.capacity());

  std::vector<double> exit_rate_boundary(n, 0.0);
  std::vector<double> diag(n, 0.0);

  for (int j = 0; j < n; ++j) {
    const MultiIndex& x = sp.states[j];
    double col_sum = 0.0;

    for (std::size_t k = 0; k < model.n_reactions(); ++k) {
      double alpha = model.propensities[k](x, rates, t);
      if (alpha <= 0.0) continue;

      MultiIndex y = x + model.stoich_vecs[k];
      auto it = sp.index.find(y);
      if (it != sp.index.end()) {
        int i = it->second;
        rows.push_back(static_cast<num::idx>(i));
        cols.push_back(static_cast<num::idx>(j));
        vals.push_back(alpha);
        col_sum += alpha;
      } else {
        exit_rate_boundary[j] += alpha;
        if (absorbing) col_sum += alpha;
      }
    }

    diag[j] = col_sum;
    if (col_sum > 0.0) {
      rows.push_back(static_cast<num::idx>(j));
      cols.push_back(static_cast<num::idx>(j));
      vals.push_back(-col_sum);
    }
  }

  num::SparseMatrix A = num::SparseMatrix::from_triplets(
      static_cast<num::idx>(n), static_cast<num::idx>(n), rows, cols, vals);

  return {std::move(A), std::move(exit_rate_boundary), std::move(diag)};
}

} // namespace fsp
