#include "else/subnetwork.hpp"
#include "fsp/generator.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace else_sim {
namespace {

constexpr num::idx sparse_threshold = 32;

num::SparseMatrix negative(const num::SparseMatrix& matrix) {
  std::vector<double> values(matrix.values(), matrix.values() + matrix.nnz());
  for (double& value : values) {
    value = -value;
  }
  return {matrix.n_rows(),
          matrix.n_cols(),
          std::move(values),
          std::vector<num::idx>(matrix.col_idx(), matrix.col_idx() + matrix.nnz()),
          std::vector<num::idx>(matrix.row_ptr(), matrix.row_ptr() + matrix.n_rows() + 1)};
}

num::SparseMatrix transpose(const num::SparseMatrix& matrix) {
  std::vector<num::idx> rows;
  std::vector<num::idx> columns;
  std::vector<double> values;
  rows.reserve(matrix.nnz());
  columns.reserve(matrix.nnz());
  values.reserve(matrix.nnz());
  for (num::idx row = 0; row < matrix.n_rows(); ++row) {
    for (num::idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
      rows.push_back(matrix.col_idx()[entry]);
      columns.push_back(row);
      values.push_back(matrix.values()[entry]);
    }
  }
  return num::SparseMatrix::from_triplets(
      matrix.n_cols(), matrix.n_rows(), rows, columns, values);
}

num::Matrix dense(const num::SparseMatrix& matrix) {
  num::Matrix result(matrix.n_rows(), matrix.n_cols(), 0.0);
  for (num::idx row = 0; row < matrix.n_rows(); ++row) {
    for (num::idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
      result(row, matrix.col_idx()[entry]) = matrix.values()[entry];
    }
  }
  return result;
}

num::Matrix symmetric_operator(const num::SparseMatrix& matrix,
                               std::span<const double> stationary_sqrt) {
  if (stationary_sqrt.size() != matrix.n_rows() ||
      !std::all_of(stationary_sqrt.begin(), stationary_sqrt.end(), [](double value) {
        return value > 0.0;
      })) {
    throw std::invalid_argument("reversible weights must be positive and match the subnetwork");
  }

  num::Matrix result(matrix.n_rows(), matrix.n_cols(), 0.0);
  for (num::idx row = 0; row < matrix.n_rows(); ++row) {
    for (num::idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
      const num::idx column = matrix.col_idx()[entry];
      result(row, column) =
          matrix.values()[entry] * stationary_sqrt[column] / stationary_sqrt[row];
    }
  }
  double scale = 0.0;
  double symmetry_error = 0.0;
  for (num::idx row = 0; row < result.rows(); ++row) {
    for (num::idx column = 0; column < row; ++column) {
      scale = std::max(scale, std::abs(result(row, column)));
      scale = std::max(scale, std::abs(result(column, row)));
      symmetry_error =
          std::max(symmetry_error, std::abs(result(row, column) - result(column, row)));
    }
  }
  if (symmetry_error > 1e-10 * std::max(1.0, scale)) {
    throw std::invalid_argument("stationary weights do not symmetrize the subnetwork generator");
  }
  return result;
}

} // namespace

struct Subnetwork::SolverImpl {
  explicit SolverImpl(num::SparseMatrix operator_matrix)
      : matrix(std::move(operator_matrix)) {
    if (matrix.n_rows() >= sparse_threshold && num::klu_available()) {
      sparse_factor = std::make_unique<num::KLUFactor>(matrix);
    } else {
      dense_factor = num::lu(dense(matrix));
    }
  }

  SolverImpl(num::SparseMatrix operator_matrix, std::span<const double> stationary_sqrt)
      : matrix(std::move(operator_matrix)),
        weights(stationary_sqrt.begin(), stationary_sqrt.end()) {
    auto symmetric = symmetric_operator(matrix, stationary_sqrt);
    cholesky_factor = num::cholesky(num::linalg::assume_spd(std::move(symmetric)));
    if (!cholesky_factor->success) {
      throw std::invalid_argument("reversible subnetwork operator is not positive definite");
    }
  }

  num::SparseMatrix matrix;
  std::optional<num::LUResult> dense_factor;
  std::optional<num::CholeskyResult> cholesky_factor;
  std::unique_ptr<num::KLUFactor> sparse_factor;
  std::optional<num::LUResult> transpose_factor;
  std::unique_ptr<num::KLUFactor> sparse_transpose_factor;
  std::vector<double> weights;

  void solve(const num::Vector& b, num::Vector& x) const {
    if (cholesky_factor) {
      num::Vector transformed_b(b.size(), 0.0);
      num::Vector transformed_x(b.size(), 0.0);
      for (num::idx state = 0; state < b.size(); ++state) {
        transformed_b[state] = b[state] / weights[state];
      }
      num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
      for (num::idx state = 0; state < b.size(); ++state) {
        x[state] = weights[state] * transformed_x[state];
      }
      return;
    }
    if (sparse_factor) {
      sparse_factor->solve(b, x);
    } else {
      if (!dense_factor) {
        throw std::logic_error("missing dense subnetwork factorization");
      }
      num::lu_solve(*dense_factor, b, x);
    }
  }

  void solve(const num::Matrix& right_hand_side, num::Matrix& solution) const {
    if (cholesky_factor) {
      num::Matrix transformed = right_hand_side;
      for (num::idx row = 0; row < transformed.rows(); ++row) {
        for (num::idx column = 0; column < transformed.cols(); ++column) {
          transformed(row, column) /= weights[row];
        }
      }
      num::cholesky_solve(*cholesky_factor, transformed, solution);
      for (num::idx row = 0; row < solution.rows(); ++row) {
        for (num::idx column = 0; column < solution.cols(); ++column) {
          solution(row, column) *= weights[row];
        }
      }
      return;
    }
    if (sparse_factor) {
      sparse_factor->solve(right_hand_side, solution);
    } else {
      if (!dense_factor) {
        throw std::logic_error("missing dense subnetwork factorization");
      }
      num::lu_solve(*dense_factor, right_hand_side, solution);
    }
  }

  void solve_transpose(const num::Vector& b, num::Vector& x) {
    if (cholesky_factor) {
      num::Vector transformed_b(b.size(), 0.0);
      num::Vector transformed_x(b.size(), 0.0);
      for (num::idx state = 0; state < b.size(); ++state) {
        transformed_b[state] = weights[state] * b[state];
      }
      num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
      for (num::idx state = 0; state < b.size(); ++state) {
        x[state] = transformed_x[state] / weights[state];
      }
      return;
    }
    // Only cut-time losses need the transpose, so factor it lazily.
    if (sparse_factor) {
      if (!sparse_transpose_factor) {
        sparse_transpose_factor = std::make_unique<num::KLUFactor>(transpose(matrix));
      }
      sparse_transpose_factor->solve(b, x);
      return;
    }
    if (!transpose_factor) {
      transpose_factor = num::lu(dense(transpose(matrix)));
    }
    if (!transpose_factor) {
      throw std::logic_error("missing transposed subnetwork factorization");
    }
    num::lu_solve(*transpose_factor, b, x);
  }

  void inverse_diagonal(std::vector<double>& diagonal) const {
    const num::idx n = diagonal.size();
    constexpr num::idx block_size = 64;

    // Solve blocks of basis vectors; never materialize the full inverse.
    for (num::idx start = 0; start < n; start += block_size) {
      const num::idx current_block = std::min(block_size, n - start);
      num::Matrix right_hand_side(n, current_block, 0.0);
      for (num::idx column = 0; column < current_block; ++column) {
        right_hand_side(start + column, column) = 1.0;
      }
      num::Matrix solution(n, current_block, 0.0);
      solve(right_hand_side, solution);
      for (num::idx column = 0; column < current_block; ++column) {
        diagonal[start + column] = solution(start + column, column);
      }
    }
  }
};

Subnetwork::Subnetwork(const fsp::StateSpace& sp,
                       const markovkit::ReactionSystem& model,
                       const std::vector<double>& rates,
                       double t)
    : Subnetwork(sp, model, rates, t, nullptr) {}

Subnetwork::Subnetwork(const fsp::StateSpace& sp,
                       const markovkit::ReactionSystem& model,
                       const std::vector<double>& rates,
                       Reversible reversible,
                       double t)
    : Subnetwork(sp, model, rates, t, &reversible) {}

Subnetwork::Subnetwork(const fsp::StateSpace& sp,
                       const markovkit::ReactionSystem& model,
                       const std::vector<double>& rates,
                       double t,
                       const Reversible* reversible)
    : states_(sp.states),
      index_(sp.index) {
  const num::idx n = states_.size();
  auto assembled = fsp::build_generator(sp, model, rates, t, true);
  R_ = std::move(assembled.matrix);
  exit_rates_ = std::move(assembled.boundary_rates);
  total_rates_ = std::move(assembled.diagonal_rates);
  boundary_ = std::move(assembled.boundary);
  for (num::idx state = 0; state < n; ++state) {
    if (exit_rates_[state] > 0.0) {
      boundary_states_.push_back(state);
    }
  }

  // Occupation uses the positive transient operator -R.
  auto operator_matrix = negative(R_);
  solver_ = reversible == nullptr
                ? std::make_unique<SolverImpl>(std::move(operator_matrix))
                : std::make_unique<SolverImpl>(std::move(operator_matrix),
                                               reversible->stationary_sqrt);
}

Subnetwork::Subnetwork(std::vector<markovkit::State> states,
                       num::SparseMatrix generator,
                       std::vector<markovkit::BoundaryTransition> boundary,
                       Reversible reversible)
    : states_(std::move(states)),
      R_(std::move(generator)),
      total_rates_(states_.size(), 0.0),
      exit_rates_(states_.size(), 0.0),
      boundary_(std::move(boundary)) {
  if (R_.n_rows() != states_.size() || R_.n_cols() != states_.size()) {
    throw std::invalid_argument("subnetwork generator dimensions do not match its states");
  }
  for (num::idx state = 0; state < states_.size(); ++state) {
    index_[states_[state]] = static_cast<int>(state);
    total_rates_[state] = -R_(state, state);
  }
  for (const auto& transition : boundary_) {
    if (transition.source >= states_.size()) {
      throw std::invalid_argument("subnetwork boundary source is out of range");
    }
    exit_rates_[transition.source] += transition.rate;
  }
  for (num::idx state = 0; state < states_.size(); ++state) {
    if (exit_rates_[state] > 0.0) {
      boundary_states_.push_back(state);
    }
  }
  solver_ = std::make_unique<SolverImpl>(negative(R_), reversible.stationary_sqrt);
}

Subnetwork::~Subnetwork() = default;
Subnetwork::Subnetwork(Subnetwork&&) noexcept = default;
Subnetwork& Subnetwork::operator=(Subnetwork&&) noexcept = default;

int Subnetwork::find(const markovkit::State& x) const {
  auto it = index_.find(x);
  return it != index_.end() ? it->second : -1;
}

num::Vector Subnetwork::occupation(const num::Vector& start) const {
  num::Vector u(size(), 0.0);
  solver_->solve(start, u);
  return u;
}

OccupationMoments Subnetwork::occupation_moments(const num::Matrix& starts) const {
  OccupationMoments m;
  m.first = num::Matrix(size(), starts.cols(), 0.0);
  m.second = num::Matrix(size(), starts.cols(), 0.0);
  solver_->solve(starts, m.first);
  // A second solve gives (-R)^-2 starts without forming (-R)^-1.
  solver_->solve(m.first, m.second);
  return m;
}

std::vector<double> Subnetwork::exit_probabilities(const num::Vector& occupancy) const {
  std::vector<double> probs;
  probs.reserve(boundary_states_.size());
  double total = 0.0;
  for (num::idx b : boundary_states_) {
    double p = exit_rates_[b] * std::max(0.0, occupancy[b]);
    probs.push_back(p);
    total += p;
  }
  if (total > 0.0) {
    for (double& p : probs) {
      p /= total;
    }
  }
  return probs;
}

std::vector<double> Subnetwork::exit_probabilities(const OccupationMoments& m, num::idx col) const {
  std::vector<double> probs;
  probs.reserve(boundary_states_.size());
  double total = 0.0;
  for (num::idx b : boundary_states_) {
    double p = exit_rates_[b] * std::max(0.0, m.first(b, col));
    probs.push_back(p);
    total += p;
  }
  if (total > 0.0) {
    for (double& p : probs) {
      p /= total;
    }
  }
  return probs;
}

double Subnetwork::conditional_exit_time(const OccupationMoments& m,
                                         num::idx exit_state_idx,
                                         num::idx col) const {
  double u = m.first(exit_state_idx, col);
  if (!(u > 0.0)) {
    return 0.0;
  }
  return std::max(0.0, m.second(exit_state_idx, col)) / u;
}

std::vector<double> Subnetwork::expected_entries(const num::Vector& occupancy,
                                                 const num::Vector& start) const {
  std::vector<double> counts(size(), 0.0);
  for (num::idx j = 0; j < size(); ++j) {
    // Departures equal rate times occupation; subtract the initial entrance.
    counts[j] = std::max(0.0, (total_rates_[j] * occupancy[j]) - start[j]);
  }
  return counts;
}

std::vector<double> Subnetwork::cut_time_losses(const num::Vector& occupancy) const {
  const num::idx n = size();
  num::Vector ones(n, 1.0);
  num::Vector z_col_sums(n, 0.0); // Z^T 1
  solver_->solve_transpose(ones, z_col_sums);

  std::vector<double> z_diag(n, 0.0); // Z_kk
  solver_->inverse_diagonal(z_diag);

  std::vector<double> losses(n, 0.0);
  for (num::idx k = 0; k < n; ++k) {
    if (z_diag[k] > 1e-15) {
      losses[k] = occupancy[k] * (z_col_sums[k] / z_diag[k]);
    }
  }
  return losses;
}

} // namespace else_sim
