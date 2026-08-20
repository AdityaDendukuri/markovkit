#pragma once
#include "fsp/multi_index.hpp"
#include <functional>
#include <vector>

namespace fsp {

// Propensity rate: a_k(x, rates, t)
using PropensityFn = std::function<double(const MultiIndex&, const std::vector<double>&, double)>;

// Discrete reaction network: stoichiometric vectors + propensities
struct ReactionSystem {
  std::vector<MultiIndex> stoich_vecs;
  std::vector<PropensityFn> propensities;

  [[nodiscard]] std::size_t n_reactions() const { return stoich_vecs.size(); }
};

} // namespace fsp
