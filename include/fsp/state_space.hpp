#pragma once
#include "fsp/boundary.hpp"
#include "fsp/detail/flat_hash_map.hpp"
#include "fsp/multi_index.hpp"
#include "fsp/reaction_system.hpp"
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace fsp {

// Discrete active state space with indexed lookups
struct StateSpace {
  std::vector<MultiIndex> states;
  std::vector<double> probs;
  fsp::FlatHashMap<MultiIndex, int> index;
  bool dirty = true;

  [[nodiscard]] int size() const { return static_cast<int>(states.size()); }
  [[nodiscard]] bool empty() const { return states.empty(); }

  // Position of state, or -1 if absent
  [[nodiscard]] int find(const MultiIndex& x) const {
    auto it = index.find(x);
    return it != index.end() ? it->second : -1;
  }

  // Add state if absent
  void add_state(const MultiIndex& x, double prob = 0.0);

  // Remove states flagged true in remove_mask
  void remove_states(const std::vector<bool>& remove_mask);

  // Expand state space by stoichiometric neighbors inside boundary
  void expand(const ReactionSystem& model, const BoundaryCondition& bc, int depth = 1);

  // SSA-guided expansion
  void expand_ssa(const ReactionSystem& model,
                  const std::vector<double>& rates,
                  double t,
                  const BoundaryCondition& bc,
                  int depth,
                  std::mt19937& rng);

  // Flux-weighted pruning: returns (total_flux, n_removed)
  std::pair<double, int> compress(const ReactionSystem& model,
                                  const std::vector<double>& rates,
                                  double t,
                                  double prob_quantile,
                                  double flux_tolerance = 1e-6,
                                  const std::vector<double>* precomputed_exit_rates = nullptr);

  // Normalize probs to sum to 1.0; returns previous sum
  double renormalize();
};

} // namespace fsp
