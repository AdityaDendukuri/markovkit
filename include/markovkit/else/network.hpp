/// @file else/network.hpp
/// @brief FESSNetwork: dynamic multi-index state space augmentation and
/// indexing.
#pragma once

#include "markovkit/fsp/detail/flat_hash_map.hpp"
#include "markovkit/fsp/multi_index.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include <vector>

namespace markovkit::else_solver {

struct FESSNetwork {
  std::vector<fsp::MultiIndex> states;
  fsp::FlatHashMap<fsp::MultiIndex, int> index;

  FESSNetwork() = default;
  explicit FESSNetwork(fsp::MultiIndex initial_state);
  explicit FESSNetwork(std::vector<fsp::MultiIndex> initial_states);

  int size() const {
    return static_cast<int>(states.size());
  }
  bool empty() const {
    return states.empty();
  }

  /// @brief Augment network with single state. Returns index.
  int augment(const fsp::MultiIndex& state);

  /// @brief Add depth reaction-graph shells around entrance states
  void augment(const std::vector<fsp::MultiIndex>& entrances,
               const fsp::ReactionSystem& model,
               const std::vector<double>& rates,
               int depth = 1);
};

} // namespace markovkit::else_solver
