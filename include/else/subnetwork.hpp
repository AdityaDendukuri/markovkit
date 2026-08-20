#pragma once
#include "fsp/multi_index.hpp"
#include "fsp/reaction_system.hpp"
#include "fsp/state_space.hpp"
#include <memory>
#include <numerics.hpp>
#include <optional>
#include <vector>

namespace else_sim {

using MultiIndex = fsp::MultiIndex;
using State = fsp::MultiIndex;

// Transition crossing subnetwork boundary to an outside state
struct ExitEdge {
  num::idx source;       // source state index in subnetwork
  num::idx destination;  // destination index in destinations list
  double rate;
};

// First and second occupancy moments: u = (-R)^-1 p and u^(2) = (-R)^-2 p
struct OccupancyMoments {
  num::Matrix first;   // (-R)^-1 * starts
  num::Matrix second;  // (-R)^-2 * starts

  [[nodiscard]] num::idx size() const { return first.rows(); }
  [[nodiscard]] num::idx starts() const { return first.cols(); }
};

// Restricted generator R and exit boundary for a discrete state subset
class Subnetwork {
public:
  Subnetwork(const fsp::StateSpace& sp,
             const fsp::ReactionSystem& model,
             const std::vector<double>& rates,
             double t = 0.0);
  ~Subnetwork();
  Subnetwork(Subnetwork&&) noexcept;
  Subnetwork& operator=(Subnetwork&&) noexcept;

  [[nodiscard]] num::idx size() const { return states_.size(); }
  [[nodiscard]] const std::vector<MultiIndex>& states() const { return states_; }
  [[nodiscard]] const std::vector<MultiIndex>& destinations() const { return destinations_; }
  [[nodiscard]] const std::vector<num::idx>& boundary_states() const { return boundary_states_; }
  [[nodiscard]] const std::vector<double>& exit_rates() const { return exit_rates_; }
  [[nodiscard]] const std::vector<double>& total_rates() const { return total_rates_; }
  [[nodiscard]] const std::vector<ExitEdge>& exit_edges() const { return edges_; }
  [[nodiscard]] const num::SparseMatrix& generator() const { return R_; }

  int find(const MultiIndex& x) const;

  // Solves (-R) u = start  ->  expected time spent in each state
  num::Vector occupation(const num::Vector& start) const;

  // Solves (-R) U = starts and (-R) U2 = U  ->  1st and 2nd moments
  OccupancyMoments moments(const num::Matrix& starts) const;

  // Escape probabilities from boundary states, normalized
  std::vector<double> escape_probabilities(const num::Vector& occupancy) const;
  std::vector<double> escape_probabilities(const OccupancyMoments& moments, num::idx col = 0) const;

  // Mean time until exit given exit occurs at exit_state_idx: u^(2)[b] / u[b]
  double conditional_mean_time(const OccupancyMoments& moments,
                               num::idx exit_state_idx,
                               num::idx col = 0) const;

  // Expected entry counts: total_rate[j] * u[j] - start[j]
  std::vector<double> entry_counts(const num::Vector& occupancy,
                                   const num::Vector& start) const;

  // Exact loss in mean exit time if state is cut
  std::optional<std::vector<double>> cut_time_losses(const num::Vector& occupancy) const;

  // Flux of probability entering each outside destination state
  std::vector<double> exit_flux(const num::Vector& occupancy) const;
  std::vector<num::cplx> exit_flux(const std::vector<num::cplx>& occupancy) const;

  // Resolvent solve (sI - R)^-1 start for complex shift s
  std::vector<num::cplx> solve_resolvent(num::cplx s,
                                         const std::vector<num::cplx>& start) const;

private:
  std::vector<MultiIndex> states_;
  fsp::FlatHashMap<MultiIndex, int> index_;
  num::SparseMatrix R_{0, 0, {}, {}, {0}};
  std::vector<double> total_rates_;
  std::vector<double> exit_rates_;
  std::vector<num::idx> boundary_states_;
  std::vector<MultiIndex> destinations_;
  std::vector<ExitEdge> edges_;

  struct SolverImpl;
  std::unique_ptr<SolverImpl> solver_;
};

} // namespace else_sim
