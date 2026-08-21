#pragma once
#include "fsp/state_space.hpp"
#include "markovkit/boundary_transition.hpp"
#include <memory>
#include <numerics.hpp>
#include <span>
#include <unordered_map>
#include <vector>

namespace else_sim {

/// Stationary square-root weights for a reversible CME.
struct Reversible {
  std::span<const double> stationary_sqrt;
};

/// First two occupation moments from a shared factorization of -R.
struct OccupationMoments {
  num::Matrix first;  // (-R)^-1 * starts
  num::Matrix second; // (-R)^-2 * starts

  [[nodiscard]] num::idx size() const {
    return first.rows();
  }
  [[nodiscard]] num::idx starts() const {
    return first.cols();
  }
};

/// Restricted generator, boundary transitions, and occupation solver.
class Subnetwork {
public:
  Subnetwork(const fsp::StateSpace& sp,
             const markovkit::ReactionSystem& model,
             const std::vector<double>& rates,
             double t = 0.0);
  Subnetwork(const fsp::StateSpace& sp,
             const markovkit::ReactionSystem& model,
             const std::vector<double>& rates,
             Reversible reversible,
             double t = 0.0);
  Subnetwork(std::vector<markovkit::State> states,
             num::SparseMatrix generator,
             std::vector<markovkit::BoundaryTransition> boundary,
             Reversible reversible);
  ~Subnetwork();
  Subnetwork(Subnetwork&&) noexcept;
  Subnetwork& operator=(Subnetwork&&) noexcept;

  [[nodiscard]] num::idx size() const {
    return states_.size();
  }
  [[nodiscard]] const std::vector<markovkit::State>& states() const {
    return states_;
  }
  [[nodiscard]] const std::vector<num::idx>& boundary_states() const {
    return boundary_states_;
  }
  [[nodiscard]] const std::vector<markovkit::BoundaryTransition>& boundary() const {
    return boundary_;
  }
  [[nodiscard]] const num::SparseMatrix& generator() const {
    return R_;
  }

  [[nodiscard]] int find(const markovkit::State& x) const;

  // Solve (-R)u = start; u is the expected time spent in each state.
  [[nodiscard]] num::Vector occupation(const num::Vector& start) const;

  // Compute (-R)^-1 starts and (-R)^-2 starts using one factorization.
  [[nodiscard]] OccupationMoments occupation_moments(const num::Matrix& starts) const;

  [[nodiscard]] std::vector<double>
  exit_probabilities(const num::Vector& occupation) const;
  [[nodiscard]] std::vector<double>
  exit_probabilities(const OccupationMoments& occupation, num::idx column = 0) const;

  [[nodiscard]] double conditional_exit_time(const OccupationMoments& occupation,
                                             num::idx boundary_state,
                                             num::idx column = 0) const;

  [[nodiscard]] std::vector<double> expected_entries(const num::Vector& occupation,
                                                     const num::Vector& start) const;

  // Exact mean-exit-time loss from removing each state.
  [[nodiscard]] std::vector<double> cut_time_losses(const num::Vector& occupation) const;

private:
  Subnetwork(const fsp::StateSpace& sp,
             const markovkit::ReactionSystem& model,
             const std::vector<double>& rates,
             double t,
             const Reversible* reversible);

  std::vector<markovkit::State> states_;
  std::unordered_map<markovkit::State, int> index_;
  num::SparseMatrix R_{0, 0, {}, {}, {0}};
  std::vector<double> total_rates_;
  std::vector<double> exit_rates_;
  std::vector<num::idx> boundary_states_;
  std::vector<markovkit::BoundaryTransition> boundary_;

  struct SolverImpl;
  std::unique_ptr<SolverImpl> solver_;
};

} // namespace else_sim
