#include "fsp/state_space.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace fsp {

void StateSpace::add_state(const MultiIndex& x, double prob) {
  if (index.count(x)) return;
  dirty = true;
  int i = static_cast<int>(states.size());
  states.push_back(x);
  probs.push_back(prob);
  index[x] = i;
}

void StateSpace::remove_states(const std::vector<bool>& remove_mask) {
  if (std::none_of(remove_mask.begin(), remove_mask.end(), [](bool b) { return b; })) return;
  dirty = true;
  int n = static_cast<int>(states.size());
  std::vector<MultiIndex> new_states;
  std::vector<double> new_probs;
  new_states.reserve(n);
  new_probs.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (!remove_mask[i]) {
      new_states.push_back(states[i]);
      new_probs.push_back(probs[i]);
    }
  }
  states = std::move(new_states);
  probs = std::move(new_probs);
  index.clear();
  for (int i = 0; i < static_cast<int>(states.size()); ++i) index[states[i]] = i;
}

void StateSpace::expand(const ReactionSystem& model, const BoundaryCondition& bc, int depth) {
  for (int d = 0; d < depth; ++d) {
    int n_before = static_cast<int>(states.size());
    for (int i = 0; i < n_before; ++i) {
      MultiIndex x = states[i];
      for (const MultiIndex& nu : model.stoich_vecs) {
        MultiIndex y = x + nu;
        if (!index.count(y) && bc(y)) add_state(y, 0.0);
      }
    }
  }
}

void StateSpace::expand_ssa(const ReactionSystem& model,
                            const std::vector<double>& rates,
                            double t,
                            const BoundaryCondition& bc,
                            int depth,
                            std::mt19937& rng) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  int nk = static_cast<int>(model.n_reactions());
  std::vector<double> props(nk);
  for (int d = 0; d < depth; ++d) {
    int n_before = static_cast<int>(states.size());
    for (int i = 0; i < n_before; ++i) {
      MultiIndex x = states[i];
      double total = 0.0;
      for (int k = 0; k < nk; ++k) {
        double a = std::max(0.0, model.propensities[k](x, rates, t));
        props[k] = a;
        total += a;
      }
      if (total <= 0.0) continue;

      double threshold = uniform(rng) * total;
      double csum = 0.0;
      int chosen = -1;
      for (int k = 0; k < nk; ++k) {
        csum += props[k];
        if (threshold <= csum) {
          chosen = k;
          break;
        }
      }
      if (chosen < 0) continue;

      MultiIndex y = x + model.stoich_vecs[chosen];
      if (!index.count(y) && bc(y)) add_state(y, 0.0);
    }
  }
}

std::pair<double, int> StateSpace::compress(const ReactionSystem& model,
                                            const std::vector<double>& rates,
                                            double t,
                                            double prob_quantile,
                                            double flux_tolerance,
                                            const std::vector<double>* precomputed_exit_rates) {
  int n = size();
  if (n <= 1) return {0.0, 0};

  std::vector<double> local_rates;
  const std::vector<double>* exit_rates_ptr = precomputed_exit_rates;
  if (!exit_rates_ptr) {
    local_rates.resize(n, 0.0);
    int nk = static_cast<int>(model.n_reactions());
    for (int i = 0; i < n; ++i) {
      const MultiIndex& x = states[i];
      double total = 0.0;
      for (int k = 0; k < nk; ++k) {
        total += std::max(0.0, model.propensities[k](x, rates, t));
      }
      local_rates[i] = total;
    }
    exit_rates_ptr = &local_rates;
  }

  std::vector<double> flux(n, 0.0);
  double total_flux = 0.0;
  for (int i = 0; i < n; ++i) {
    flux[i] = probs[i] * (*exit_rates_ptr)[i];
    total_flux += flux[i];
  }

  int n_candidates = std::min(static_cast<int>(std::round(prob_quantile * n)), n - 1);
  if (n_candidates <= 0) return {total_flux, 0};

  std::vector<int> candidate_idx(n);
  std::iota(candidate_idx.begin(), candidate_idx.end(), 0);
  std::nth_element(candidate_idx.begin(), candidate_idx.begin() + n_candidates, candidate_idx.end(),
                   [this](int a, int b) { return probs[a] < probs[b]; });

  double threshold = flux_tolerance * total_flux;
  std::vector<bool> remove_mask(n, false);
  int n_removed = 0;
  for (int i = 0; i < n_candidates; ++i) {
    int idx = candidate_idx[i];
    if (flux[idx] < threshold) {
      remove_mask[idx] = true;
      ++n_removed;
    }
  }

  if (n_removed > 0) remove_states(remove_mask);
  return {total_flux, n_removed};
}

double StateSpace::renormalize() {
  double total = 0.0;
  for (double p : probs) total += p;
  if (total > 0.0) {
    double inv = 1.0 / total;
    for (double& p : probs) p *= inv;
  }
  return total;
}

} // namespace fsp
