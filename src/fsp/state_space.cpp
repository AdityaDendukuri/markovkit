/// @file src/fsp/state_space.cpp
/// @brief Compiled implementations for StateSpace methods.
#include "markovkit/fsp/state_space.hpp"

namespace fsp {

void StateSpace::add_state(const MultiIndex &x, double prob) {
  auto it = index.find(x);
  if (it != index.end()) {
    probs[it->second] += prob;
    return;
  }
  int new_idx = (int)states.size();
  states.push_back(x);
  probs.push_back(prob);
  index[x] = new_idx;

  auto git = global_id.find(x);
  if (git == global_id.end()) {
    global_id[x] = next_id++;
    id_to_state.push_back(x);
  }
  dirty = true;
}

void StateSpace::remove_states(const std::vector<bool> &remove_mask) {
  int n = (int)states.size();
  std::vector<MultiIndex> new_states;
  std::vector<double> new_probs;
  new_states.reserve(n);
  new_probs.reserve(n);

  index.clear();
  for (int i = 0; i < n; ++i) {
    if (!remove_mask[i]) {
      int new_idx = (int)new_states.size();
      index[states[i]] = new_idx;
      new_states.push_back(states[i]);
      new_probs.push_back(probs[i]);
    }
  }
  states = std::move(new_states);
  probs = std::move(new_probs);
  dirty = true;
}

void StateSpace::expand(const ReactionSystem &model,
                        const BoundaryCondition &bc, int depth) {
  for (int d = 0; d < depth; ++d) {
    std::vector<MultiIndex> current_states = states;
    for (const auto &s : current_states) {
      for (const auto &v : model.stoich_vecs) {
        MultiIndex nxt = s;
        for (std::size_t i = 0; i < s.size(); ++i)
          nxt[i] += v[i];
        if (!bc || bc(nxt)) {
          add_state(nxt, 0.0);
        }
      }
    }
  }
}

void StateSpace::expand_ssa(const ReactionSystem &model,
                            const std::vector<double> &rates, double t,
                            const BoundaryCondition &bc, int depth,
                            std::mt19937 &rng) {
  expand(model, bc, depth);
}

std::pair<double, int>
StateSpace::compress(const ReactionSystem &model,
                     const std::vector<double> &rates, double t,
                     double prob_quantile, double flux_tolerance,
                     const std::vector<double> *precomputed_exit_rates) {
  return {0.0, 0};
}

double StateSpace::renormalize() {
  double s = 0.0;
  for (double p : probs)
    s += p;
  if (s > 0.0) {
    for (double &p : probs)
      p /= s;
  }
  return s;
}

std::vector<int> StateSpace::get_global_ids() const {
  std::vector<int> gids(states.size());
  for (std::size_t i = 0; i < states.size(); ++i) {
    gids[i] = global_id.at(states[i]);
  }
  return gids;
}

} // namespace fsp
