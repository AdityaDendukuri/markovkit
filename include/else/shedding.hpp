/// @file else/shedding.hpp
/// @brief State space shedding, exit-time loss evaluation, and adaptive capacity control for ELSE.
#pragma once

#include "cme/state_space.hpp"
#include "else/subnetwork.hpp"
#include <cstdint>
#include <numerics.hpp>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace else_sim {

/// @brief Algorithmic strategy used to evaluate exit-time losses for state shedding.
enum class SheddingMethod : std::uint8_t {
    CholeskyWoodbury, ///< Method 1: Dense Cholesky Woodbury with floating-point error tracking &
                      ///< fallback
    NormalizedSymmetrized, ///< Method 2: Symmetrization and exact inverse-diagonal extraction
    ExpectedVisits,        ///< Method 3: Fast O(n) expected visits / escape rate approximation
    Auto,                  ///< Exact cut time when reversible; expected visits otherwise
};

/// @brief Configuration parameters for state shedding and truncation.
struct SheddingOptions {
    SheddingMethod method = SheddingMethod::Auto;
    num::idx target_capacity = 0; ///< Desired target capacity (0 = tolerance-driven only)
    double tolerance = 1e-4; ///< Maximum allowable relative exit-time loss \f$\Delta \tau / \tau\f$
    bool fallback_to_stable =
        true; ///< Enable graceful fallback if Woodbury downdates lose stability
};

/// @brief Diagnostics recorded during the state shedding process.
struct SheddingDiagnostics {
    SheddingMethod requested_method = SheddingMethod::Auto;
    SheddingMethod effective_method = SheddingMethod::Auto;
    num::idx initial_states = 0;
    num::idx shed_states = 0;
    num::idx remaining_states = 0;
    double initial_mean_exit_time = 0.0;
    double total_estimated_loss = 0.0;
    double relative_loss = 0.0;
    bool fallback_triggered = false;
    std::string fallback_reason;
};

/// @brief Result of a state space shedding operation.
struct SheddingResult {
    std::vector<num::idx> kept_indices;
    std::vector<num::idx> shed_indices;
    std::vector<double> state_losses;
    SheddingDiagnostics diagnostics;
};

/// @brief Computes candidate-by-candidate mean-exit-time loss for all states in a subnetwork.
[[nodiscard]] std::vector<double>
compute_shedding_losses(const Subnetwork &subnetwork, const num::Vector &occupation,
                        const SheddingOptions &options = {},
                        SheddingDiagnostics *diagnostics = nullptr);

/// @brief Performs state space shedding on a subnetwork given an initial occupation profile.
[[nodiscard]] SheddingResult shed_states(const Subnetwork &subnetwork,
                                         const num::Vector &occupation,
                                         const std::vector<num::idx> &protected_states = {},
                                         const SheddingOptions &options = {});

/// @brief Prunes a CME StateSpace in-place based on the specified shedding options.
void prune_state_space(cme::StateSpace &states, const Subnetwork &subnetwork,
                       const num::Vector &entrance,
                       const std::unordered_set<markovkit::State> &protected_states,
                       const SheddingOptions &options = {},
                       SheddingDiagnostics *diagnostics = nullptr);

} // namespace else_sim
