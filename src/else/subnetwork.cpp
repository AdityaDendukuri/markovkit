#include "else/subnetwork.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace else_sim {

struct Subnetwork::SolverImpl {
  // General subnetworks use an automatically selected dense or sparse solver.
  explicit SolverImpl(const num::SparseMatrix& operator_matrix)
      : general_solver(std::make_unique<num::AutoLinearSolver>(operator_matrix)) {}

  SolverImpl(num::Matrix symmetric, std::vector<double> stationary_sqrt)
      : weights(std::move(stationary_sqrt)) {
    // Reversible subnetworks reduce to a positive-definite Cholesky solve.
    cholesky_factor = num::cholesky(num::linalg::assume_spd(std::move(symmetric)));
    if (!cholesky_factor->success) {
      throw std::invalid_argument("reversible subnetwork operator is not positive definite");
    }
  }

  std::unique_ptr<num::AutoLinearSolver> general_solver;
  std::optional<num::CholeskyResult> cholesky_factor;
  std::vector<double> weights;
  mutable num::InverseDiagonalWorkspace inverse_workspace;

  void solve(const num::Vector& b, num::Vector& x) const {
    if (cholesky_factor) {
      // Convert between generator and symmetric coordinates around the solve.
      num::Vector transformed_b = b;
      num::divide_elements(transformed_b, weights);
      num::Vector transformed_x(b.size(), 0.0);
      num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
      x = std::move(transformed_x);
      num::scale_elements(x, weights);
      return;
    }
    general_solver->solve(b, x);
  }

  void solve(const num::Matrix& right_hand_side, num::Matrix& solution) const {
    if (cholesky_factor) {
      num::Matrix transformed = right_hand_side;
      num::divide_rows(transformed, weights);
      num::cholesky_solve(*cholesky_factor, transformed, solution);
      num::scale_rows(solution, weights);
      return;
    }
    general_solver->solve(right_hand_side, solution);
  }

  void solve_transpose(const num::Vector& b, num::Vector& x) {
    if (cholesky_factor) {
      // The transpose solve reverses the similarity scaling.
      num::Vector transformed_b = b;
      num::scale_elements(transformed_b, weights);
      num::Vector transformed_x(b.size(), 0.0);
      num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
      x = std::move(transformed_x);
      num::divide_elements(x, weights);
      return;
    }
    general_solver->solve_transpose(b, x);
  }

  void inverse_diagonal(std::vector<double>& diagonal) const {
    // Reuse workspace while extracting the fundamental matrix diagonal.
    num::Vector result(diagonal.size(), 0.0);
    if (cholesky_factor) {
      num::inverse_diagonal(*cholesky_factor, result, inverse_workspace);
    } else {
      num::inverse_diagonal(*general_solver, result, inverse_workspace);
    }
    num::copy_to(result, diagonal);
  }
};

Subnetwork::Subnetwork(cme::RestrictedGenerator restricted) {
  initialize(std::move(restricted));
  solver_ = std::make_unique<SolverImpl>(num::scaled(R_, -1.0));
}

Subnetwork::Subnetwork(cme::ReversibleGenerator reversible) {
  auto weights = std::move(reversible.stationary_sqrt);
  auto symmetric = std::move(reversible.symmetric_operator);
  initialize(std::move(reversible.restricted));
  solver_ = std::make_unique<SolverImpl>(std::move(symmetric), std::move(weights));
}

void Subnetwork::initialize(cme::RestrictedGenerator restricted) {
  // Cache the restricted generator and its boundary exit rates.
  states_ = std::move(restricted.states);
  index_ = std::move(restricted.index);
  R_ = std::move(restricted.generator);
  total_rates_.assign(states_.size(), 0.0);
  exit_rates_.assign(states_.size(), 0.0);
  boundary_ = std::move(restricted.boundary);
  if (R_.n_rows() != states_.size() || R_.n_cols() != states_.size()) {
    throw std::invalid_argument("subnetwork generator dimensions do not match its states");
  }
  for (num::idx state = 0; state < states_.size(); ++state) {
    if (!index_.contains(states_[state])) {
      index_[states_[state]] = static_cast<int>(state);
    }
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
}

Subnetwork::~Subnetwork() = default;
Subnetwork::Subnetwork(Subnetwork&&) noexcept = default;
Subnetwork& Subnetwork::operator=(Subnetwork&&) noexcept = default;

int Subnetwork::find(const markovkit::State& x) const {
  auto it = index_.find(x);
  return it != index_.end() ? it->second : -1;
}

num::Vector Subnetwork::occupation(const num::Vector& start) const {
  // Solving -R u = p0 integrates the transient state probabilities.
  num::Vector u(size(), 0.0);
  solver_->solve(start, u);
  return u;
}

OccupationIntegrals Subnetwork::occupation_integrals(const num::Matrix& starts) const {
  OccupationIntegrals integrals;
  integrals.occupation = num::Matrix(size(), starts.cols(), 0.0);
  integrals.time_weighted_occupation = num::Matrix(size(), starts.cols(), 0.0);
  solver_->solve(starts, integrals.occupation);
  // Applying (-R)^-1 twice gives integral t p(t) dt without forming the inverse.
  solver_->solve(integrals.occupation, integrals.time_weighted_occupation);
  return integrals;
}

std::vector<double> Subnetwork::exit_probabilities(const num::Vector& occupancy) const {
  // Integrated boundary flux gives the exit-state distribution.
  std::vector<double> probs;
  probs.reserve(boundary_states_.size());
  for (num::idx b : boundary_states_) {
    double p = exit_rates_[b] * std::max(0.0, occupancy[b]);
    probs.push_back(p);
  }
  num::clip_and_normalize_nonnegative(probs);
  return probs;
}

std::vector<double> Subnetwork::exit_probabilities(const OccupationIntegrals& integrals,
                                                   num::idx col) const {
  std::vector<double> probs;
  probs.reserve(boundary_states_.size());
  for (num::idx b : boundary_states_) {
    double p = exit_rates_[b] * std::max(0.0, integrals.occupation(b, col));
    probs.push_back(p);
  }
  num::clip_and_normalize_nonnegative(probs);
  return probs;
}

double Subnetwork::conditional_exit_time(const OccupationIntegrals& integrals,
                                         num::idx exit_state_idx,
                                         num::idx col) const {
  // Divide the first time moment by the exit state's occupation.
  double u = integrals.occupation(exit_state_idx, col);
  if (!(u > 0.0)) {
    return 0.0;
  }
  return std::max(0.0, integrals.time_weighted_occupation(exit_state_idx, col)) / u;
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
  // Estimate the occupation time lost by removing each state.
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
