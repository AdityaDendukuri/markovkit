/// @file else/subnetwork.hpp
/// @brief Exact Linear Subnetwork Elimination (ELSE) subnetwork & Schur
/// shedding.
#pragma once

#include "markovkit/else/options.hpp"
#include <algorithm>
#include <cmath>
#include <numerics.hpp>
#include <stdexcept>
#include <vector>

namespace markovkit::else_solver {

/// @brief Subnetwork state representation and fundamental matrix Z = -R^{-1}.
struct Subnetwork {
  num::Matrix R; ///< Killed subnetwork generator matrix R (negative diagonal)
  num::Matrix Z; ///< Fundamental matrix Z = -R^{-1}

  explicit Subnetwork(num::Matrix generator);

  /// @brief Compute occupation vector u = Z * entrance
  num::Vector occupation(const num::Vector &entrance) const;

  /// @brief Compute exact time loss vector L_k for each candidate state k
  std::vector<double> cut_time_losses(const num::Vector &entrance) const;

  /// @brief Compute Schur loss vector S_k = u_k / Z(k,k)
  std::vector<double> schur_losses(const num::Vector &entrance) const;

  /// @brief Compute symmetrized cut_norm loss vector
  std::vector<double> cut_norm_losses(const num::Vector &entrance) const;

  /// @brief O(N^2) Schur rank-1 state shedding update: Z_new = Z - Z_{:,k}
  /// Z_{k,:} / Z_{k,k}
  void shed_state(num::idx k);
};

} // namespace markovkit::else_solver
