#include "cme/generator.hpp"
#include <vector>

namespace cme {

GeneratorResult build_generator(const StateSpace& sp,
                                const ReactionSystem& model,
                                const std::vector<double>& rates,
                                double t,
                                bool absorbing) {
  int n = sp.size();
  model.validate();

  std::vector<num::idx> rows, cols;
  std::vector<num::real> vals;
  rows.reserve(static_cast<std::size_t>(n) * (model.size() + 1));
  cols.reserve(rows.capacity());
  vals.reserve(rows.capacity());

  std::vector<double> exit_rate_boundary(n, 0.0);
  std::vector<double> diag(n, 0.0);
  std::vector<markovkit::BoundaryTransition> boundary;

  // Assemble column-oriented jumps and record exits.
  for (int j = 0; j < n; ++j) {
    const markovkit::State& x = sp.states[j];
    double col_sum = 0.0;

    // Each enabled reaction contributes one jump from this source state.
    for (std::size_t k = 0; k < model.size(); ++k) {
      double alpha = model.propensities[k](x, rates, t);
      if (alpha <= 0.0) {
        continue;
      }

      markovkit::State y = x + model.changes[k];
      auto it = sp.index.find(y);
      if (it != sp.index.end()) {
        int i = it->second;
        rows.push_back(static_cast<num::idx>(i));
        cols.push_back(static_cast<num::idx>(j));
        vals.push_back(alpha);
        col_sum += alpha;
      } else {
        exit_rate_boundary[j] += alpha;
        boundary.push_back({static_cast<num::idx>(j), y, alpha});
        // Absorbing FSP keeps outward rates in the diagonal, so mass can leave.
        if (absorbing) {
          col_sum += alpha;
        }
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

  // Boundary metadata supports truncation control and ELSE exits.
  return {std::move(A), std::move(exit_rate_boundary), std::move(diag), std::move(boundary)};
}

RestrictedGenerator restrict_cme(const StateSpace& states,
                                 const ReactionSystem& model,
                                 const std::vector<double>& rates,
                                 double time) {
  // Keep boundary loss in the restricted generator diagonal.
  auto assembled = build_generator(states, model, rates, time, true);
  return {states.states,
          states.index,
          std::move(assembled.matrix),
          std::move(assembled.boundary),
          false};
}

} // namespace cme
