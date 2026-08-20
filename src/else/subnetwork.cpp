#include "else/subnetwork.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace else_sim {

struct Subnetwork::SolverImpl {
  bool is_sparse = false;
  num::Matrix dense_R;
  num::LUResult dense_lu;
  std::optional<num::SparseMatrix> sparse_negated;
  std::optional<num::SparseMatrix> sparse_R;
  std::optional<num::KLUFactor> klu_factor;
  std::optional<num::SparseResolventSolver> sparse_resolvent;

  void solve(const num::Vector& b, num::Vector& x) const {
    if (is_sparse && klu_factor) {
      klu_factor->solve(b, x);
    } else {
      num::lu_solve(dense_lu, b, x);
    }
  }

  void solve(const num::Matrix& B, num::Matrix& X) const {
    if (is_sparse && klu_factor) {
      klu_factor->solve(B, X);
    } else {
      num::lu_solve(dense_lu, B, X);
    }
  }

  void solve_shifted(num::cplx s,
                     const std::vector<num::cplx>& b,
                     std::vector<num::cplx>& x) {
    if (is_sparse && sparse_negated) {
      if (!sparse_resolvent) {
        sparse_resolvent.emplace(*sparse_negated);
      }
      sparse_resolvent->factorize(s);
      x = sparse_resolvent->solve(b);
    } else {
      x = num::ResolventFactor(s, dense_R).solve(b);
    }
  }
};

Subnetwork::Subnetwork(const fsp::StateSpace& sp,
                       const fsp::ReactionSystem& model,
                       const std::vector<double>& rates,
                       double t)
    : states_(sp.states), index_(sp.index) {
  const num::idx n = states_.size();
  total_rates_.assign(n, 0.0);
  exit_rates_.assign(n, 0.0);

  std::vector<num::idx> rows, cols;
  std::vector<num::real> vals;
  rows.reserve(n * (model.n_reactions() + 1));
  cols.reserve(rows.capacity());
  vals.reserve(rows.capacity());

  fsp::FlatHashMap<MultiIndex, num::idx> dest_map;

  for (num::idx j = 0; j < n; ++j) {
    const MultiIndex& x = states_[j];
    double total = 0.0;

    for (std::size_t k = 0; k < model.n_reactions(); ++k) {
      double rate = model.propensities[k](x, rates, t);
      if (rate <= 0.0) continue;
      total += rate;

      MultiIndex y = x + model.stoich_vecs[k];
      auto it = index_.find(y);
      if (it != index_.end()) {
        rows.push_back(static_cast<num::idx>(it->second));
        cols.push_back(j);
        vals.push_back(rate);
      } else {
        exit_rates_[j] += rate;
        auto dit = dest_map.find(y);
        num::idx dest_id;
        if (dit == dest_map.end()) {
          dest_id = static_cast<num::idx>(destinations_.size());
          dest_map[y] = dest_id;
          destinations_.push_back(y);
        } else {
          dest_id = dit->second;
        }
        edges_.push_back({j, dest_id, rate});
      }
    }

    total_rates_[j] = total;
    rows.push_back(j);
    cols.push_back(j);
    vals.push_back(-total);

    if (exit_rates_[j] > 0.0) boundary_states_.push_back(j);
  }

  R_ = num::SparseMatrix::from_triplets(n, n, rows, cols, vals);

  // Initialize solver
  solver_ = std::make_unique<SolverImpl>();
  bool use_sparse = num::klu_available() && n > 200;

  if (use_sparse) {
    solver_->is_sparse = true;
    std::vector<num::idx> neg_rows, neg_cols;
    std::vector<num::real> neg_vals;
    for (num::idx i = 0; i < R_.n_rows(); ++i) {
      for (num::idx k = R_.row_ptr()[i]; k < R_.row_ptr()[i + 1]; ++k) {
        neg_rows.push_back(i);
        neg_cols.push_back(R_.col_idx()[k]);
        neg_vals.push_back(-R_.values()[k]);
      }
    }
    solver_->sparse_negated =
        num::SparseMatrix::from_triplets(n, n, neg_rows, neg_cols, neg_vals);
    solver_->sparse_R = R_;
    solver_->klu_factor.emplace(*solver_->sparse_negated);
  } else {
    solver_->is_sparse = false;
    solver_->dense_R = num::Matrix(n, n, 0.0);
    for (num::idx i = 0; i < R_.n_rows(); ++i) {
      for (num::idx k = R_.row_ptr()[i]; k < R_.row_ptr()[i + 1]; ++k) {
        solver_->dense_R(i, R_.col_idx()[k]) = R_.values()[k];
      }
    }
    // Factor -R
    num::Matrix neg_dense(n, n, 0.0);
    for (num::idx i = 0; i < n; ++i)
      for (num::idx j = 0; j < n; ++j) neg_dense(i, j) = -solver_->dense_R(i, j);
    solver_->dense_lu = num::lu(neg_dense);
  }
}

Subnetwork::~Subnetwork() = default;
Subnetwork::Subnetwork(Subnetwork&&) noexcept = default;
Subnetwork& Subnetwork::operator=(Subnetwork&&) noexcept = default;

int Subnetwork::find(const MultiIndex& x) const {
  auto it = index_.find(x);
  return it != index_.end() ? it->second : -1;
}

num::Vector Subnetwork::occupation(const num::Vector& start) const {
  num::Vector u(size(), 0.0);
  solver_->solve(start, u);
  return u;
}

OccupancyMoments Subnetwork::moments(const num::Matrix& starts) const {
  OccupancyMoments m;
  m.first = num::Matrix(size(), starts.cols(), 0.0);
  m.second = num::Matrix(size(), starts.cols(), 0.0);
  solver_->solve(starts, m.first);
  solver_->solve(m.first, m.second);
  return m;
}

std::vector<double> Subnetwork::escape_probabilities(const num::Vector& occupancy) const {
  std::vector<double> probs;
  probs.reserve(boundary_states_.size());
  double total = 0.0;
  for (num::idx b : boundary_states_) {
    double p = exit_rates_[b] * std::max(0.0, occupancy[b]);
    probs.push_back(p);
    total += p;
  }
  if (total > 0.0) {
    for (double& p : probs) p /= total;
  }
  return probs;
}

std::vector<double> Subnetwork::escape_probabilities(const OccupancyMoments& m, num::idx col) const {
  std::vector<double> probs;
  probs.reserve(boundary_states_.size());
  double total = 0.0;
  for (num::idx b : boundary_states_) {
    double p = exit_rates_[b] * std::max(0.0, m.first(b, col));
    probs.push_back(p);
    total += p;
  }
  if (total > 0.0) {
    for (double& p : probs) p /= total;
  }
  return probs;
}

double Subnetwork::conditional_mean_time(const OccupancyMoments& m,
                                        num::idx exit_state_idx,
                                        num::idx col) const {
  double u = m.first(exit_state_idx, col);
  if (!(u > 0.0)) return 0.0;
  return std::max(0.0, m.second(exit_state_idx, col)) / u;
}

std::vector<double> Subnetwork::entry_counts(const num::Vector& occupancy,
                                            const num::Vector& start) const {
  std::vector<double> counts(size(), 0.0);
  for (num::idx j = 0; j < size(); ++j) {
    counts[j] = std::max(0.0, (total_rates_[j] * occupancy[j]) - start[j]);
  }
  return counts;
}

std::optional<std::vector<double>> Subnetwork::cut_time_losses(const num::Vector& occupancy) const {
  return std::nullopt;
}

std::vector<double> Subnetwork::exit_flux(const num::Vector& occupancy) const {
  std::vector<double> flux(destinations_.size(), 0.0);
  for (const auto& edge : edges_) {
    flux[edge.destination] += edge.rate * occupancy[edge.source];
  }
  return flux;
}

std::vector<num::cplx> Subnetwork::exit_flux(const std::vector<num::cplx>& occupancy) const {
  std::vector<num::cplx> flux(destinations_.size(), num::cplx(0.0, 0.0));
  for (const auto& edge : edges_) {
    flux[edge.destination] += edge.rate * occupancy[edge.source];
  }
  return flux;
}

std::vector<num::cplx> Subnetwork::solve_resolvent(num::cplx s,
                                                  const std::vector<num::cplx>& start) const {
  std::vector<num::cplx> x;
  solver_->solve_shifted(s, start, x);
  return x;
}

} // namespace else_sim
