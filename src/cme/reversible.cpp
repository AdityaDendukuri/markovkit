#include "cme/reversible.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cme {
namespace {

markovkit::State graph_state(num::idx state) {
  return markovkit::State{static_cast<int>(state)};
}

void validate_laplacian(const num::SparseMatrix& laplacian,
                        const std::vector<double>& stationary_sqrt) {
  if (laplacian.n_rows() != laplacian.n_cols()) {
    throw std::invalid_argument("Laplacian must be square");
  }
  if (stationary_sqrt.size() != laplacian.n_rows() ||
      !std::all_of(stationary_sqrt.begin(), stationary_sqrt.end(), [](double value) {
        return value > 0.0;
      })) {
    throw std::invalid_argument("positive stationary weights must match the Laplacian size");
  }
}

} // namespace

ReversibleGenerator reversible_generator(RestrictedGenerator restricted,
                                         std::span<const double> stationary_sqrt) {
  // Similarity-transform -Q into a symmetric operator.
  std::vector<double> weights(stationary_sqrt.begin(), stationary_sqrt.end());
  auto symmetric = num::diagonal_similarity(num::scaled(restricted.generator, -1.0), weights);
  // Scale the symmetry tolerance to the largest coupling.
  double scale = 0.0;
  for (num::idx row = 0; row < symmetric.rows(); ++row) {
    for (num::idx column = 0; column < row; ++column) {
      scale = std::max(scale, std::abs(symmetric(row, column)));
      scale = std::max(scale, std::abs(symmetric(column, row)));
    }
  }
  if (num::linalg::symmetry_error(symmetric) > 1e-10 * std::max(1.0, scale)) {
    throw std::invalid_argument("stationary weights do not symmetrize the generator");
  }
  return {std::move(restricted), std::move(weights), std::move(symmetric)};
}

ReversibleLaplacian::ReversibleLaplacian(const num::SparseMatrix& laplacian,
                                         const std::vector<double>& stationary_sqrt)
    : laplacian_(&laplacian),
      stationary_sqrt_(&stationary_sqrt) {
  validate_laplacian(laplacian, stationary_sqrt);
}

RestrictedGenerator restrict_laplacian(const num::SparseMatrix& laplacian,
                                       const std::vector<double>& stationary_sqrt,
                                       const std::vector<num::idx>& states) {
  validate_laplacian(laplacian, stationary_sqrt);
  if (states.empty()) {
    throw std::invalid_argument("a Laplacian restriction requires states");
  }

  RestrictedGenerator restricted;
  // Map selected global states to compact local indices.
  restricted.states.reserve(states.size());
  for (num::idx state : states) {
    if (state >= laplacian.n_rows()) {
      throw std::out_of_range("Laplacian state is out of range");
    }
    const auto id = graph_state(state);
    if (restricted.index.contains(id)) {
      throw std::invalid_argument("Laplacian restriction states must be unique");
    }
    restricted.index[id] = static_cast<int>(restricted.states.size());
    restricted.states.push_back(id);
  }

  std::vector<num::idx> rows;
  std::vector<num::idx> columns;
  std::vector<double> values;
  // Convert retained Laplacian edges back to generator rates.
  for (num::idx local_column = 0; local_column < states.size(); ++local_column) {
    const num::idx global_column = states[local_column];
    bool diagonal_found = false;
    for (num::idx entry = laplacian.row_ptr()[global_column];
         entry < laplacian.row_ptr()[global_column + 1];
         ++entry) {
      const num::idx global_row = laplacian.col_idx()[entry];
      if (global_row == global_column) {
        // The Laplacian diagonal becomes the negative generator diagonal.
        rows.push_back(local_column);
        columns.push_back(local_column);
        values.push_back(-laplacian.values()[entry]);
        diagonal_found = true;
        continue;
      }
      const double rate =
          -laplacian.values()[entry] * stationary_sqrt[global_row] / stationary_sqrt[global_column];
      if (rate < 0.0) {
        throw std::invalid_argument("Laplacian has a positive off-diagonal entry");
      }
      if (rate == 0.0) {
        continue;
      }
      const auto local = restricted.index.find(graph_state(global_row));
      if (local != restricted.index.end()) {
        // Retained neighbors become internal transitions.
        rows.push_back(static_cast<num::idx>(local->second));
        columns.push_back(local_column);
        values.push_back(rate);
      } else {
        // Omitted neighbors remain boundary transitions.
        restricted.boundary.push_back({local_column, graph_state(global_row), rate});
      }
    }
    if (!diagonal_found) {
      throw std::invalid_argument("Laplacian requires an explicit diagonal");
    }
  }
  restricted.generator =
      num::SparseMatrix::from_triplets(states.size(), states.size(), rows, columns, values);
  restricted.symmetric_pattern = true;
  return restricted;
}

} // namespace cme
