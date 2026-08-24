/// @file src/else/shedding.cpp
/// @brief Clean mathematical implementation of state space shedding for ELSE.
#include "else/shedding.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace else_sim {

namespace {

// Method 3: Fast O(n) Expected Entries / Visits Flow Balance
// Score: s_j = max(0, w_j * u_j) where w_j = -R_jj
std::vector<double> compute_expected_visits_losses(const Subnetwork &subnetwork,
                                                   const num::Vector &occupation) {
    const num::idx n = subnetwork.size();
    const auto &R = subnetwork.generator();
    std::vector<double> losses(n, 0.0);

    for (num::idx i = 0; i < n; ++i) {
        const double w_i = -R(i, i);
        losses[i] = (w_i > 0.0) ? std::max(0.0, occupation[i] * w_i) : 0.0;
    }
    return losses;
}

// Method 2: Steady-State Normalized Symmetrized Operator
// S_tilde = 0.5 * (H (-R) H^-1 + H^-1 (-R)^T H) with H = diag(sqrt(pi))
std::vector<double> compute_symmetrized_losses(const Subnetwork &subnetwork,
                                               const num::Vector &occupation) {
    const num::idx n = subnetwork.size();
    const auto &column_sums = subnetwork.inverse_column_sums();
    const auto &R = subnetwork.generator();
    const auto weights = subnetwork.stationary_weights();

    num::Matrix S(n, n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        const double h_i = (!weights.empty() && i < weights.size()) ? weights[i] : 1.0;
        for (num::idx j = 0; j < n; ++j) {
            const double h_j = (!weights.empty() && j < weights.size()) ? weights[j] : 1.0;
            const double r_ij = (h_i / std::max(1e-12, h_j)) * (-R(i, j));
            const double r_ji = (h_j / std::max(1e-12, h_i)) * (-R(j, i));
            S(i, j) = 0.5 * (r_ij + r_ji);
        }
    }
    for (num::idx i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (num::idx j = 0; j < n; ++j) {
            if (i != j) {
                row_sum += std::abs(S(i, j));
            }
        }
        S(i, i) = std::max(S(i, i), row_sum + 1e-6);
    }

    auto chol = num::cholesky(num::assume_spd(std::move(S)));
    if (!chol.success) {
        return compute_expected_visits_losses(subnetwork, occupation);
    }

    std::vector<double> losses(n, 0.0);
    num::Vector e(n, 0.0);
    num::Vector z(n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        e[i] = 1.0;
        num::cholesky_solve(chol, e, z);
        e[i] = 0.0;
        const double diag = std::max(z[i], 1e-12);
        const double w_i = (i < column_sums.size()) ? column_sums[i] : 1.0;
        losses[i] = std::max(0.0, occupation[i] * (w_i / diag));
    }
    return losses;
}

} // namespace

std::vector<double> compute_shedding_losses(const Subnetwork &subnetwork,
                                            const num::Vector &occupation,
                                            const SheddingOptions &options,
                                            SheddingDiagnostics *diagnostics) {
    const num::idx n = subnetwork.size();
    if (occupation.size() != n) {
        throw std::invalid_argument("occupation size must match subnetwork size");
    }

    SheddingMethod effective = options.method;
    if (effective == SheddingMethod::Auto) {
        effective = subnetwork.is_reversible() ? SheddingMethod::CholeskyWoodbury
                                               : SheddingMethod::ExpectedVisits;
    }

    if (diagnostics) {
        diagnostics->requested_method = options.method;
        diagnostics->effective_method = effective;
        diagnostics->initial_states = n;
        diagnostics->initial_mean_exit_time = 0.0;
        for (num::idx i = 0; i < n; ++i) {
            diagnostics->initial_mean_exit_time += std::max(0.0, occupation[i]);
        }
    }

    // Method 1: Exact Woodbury cut-time loss via selected inverse diagonal
    if (effective == SheddingMethod::CholeskyWoodbury) {
        try {
            auto losses = subnetwork.cut_time_losses(occupation);
            bool has_invalid = false;
            for (num::idx i = 0; i < n; ++i) {
                if (!std::isfinite(losses[i]) || losses[i] < 0.0) {
                    has_invalid = true;
                    break;
                }
            }
            if (!has_invalid) {
                return losses;
            }
            if (options.fallback_to_stable) {
                if (diagnostics) {
                    diagnostics->fallback_triggered = true;
                    diagnostics->fallback_reason = "Non-finite values in Woodbury cut-time loss";
                    diagnostics->effective_method = SheddingMethod::NormalizedSymmetrized;
                }
                return compute_symmetrized_losses(subnetwork, occupation);
            }
            return losses;
        } catch (const std::exception &ex) {
            if (options.fallback_to_stable) {
                if (diagnostics) {
                    diagnostics->fallback_triggered = true;
                    diagnostics->fallback_reason = ex.what();
                    diagnostics->effective_method = SheddingMethod::NormalizedSymmetrized;
                }
                return compute_symmetrized_losses(subnetwork, occupation);
            }
            throw;
        }
    }

    // Method 2: Symmetrization and diagonal extraction
    if (effective == SheddingMethod::NormalizedSymmetrized) {
        return compute_symmetrized_losses(subnetwork, occupation);
    }

    // Method 3: Expected visits / entries
    return compute_expected_visits_losses(subnetwork, occupation);
}

SheddingResult shed_states(const Subnetwork &subnetwork, const num::Vector &occupation,
                           const std::vector<num::idx> &protected_states,
                           const SheddingOptions &options) {
    const num::idx n = subnetwork.size();
    if (occupation.size() != n) {
        throw std::invalid_argument("occupation size must match subnetwork size");
    }

    SheddingResult result;
    result.state_losses =
        compute_shedding_losses(subnetwork, occupation, options, &result.diagnostics);

    std::unordered_set<num::idx> protected_set(protected_states.begin(), protected_states.end());
    if (protected_set.size() >= n) {
        result.kept_indices.resize(n);
        std::iota(result.kept_indices.begin(), result.kept_indices.end(), num::idx{0});
        result.diagnostics.remaining_states = n;
        result.diagnostics.shed_states = 0;
        return result;
    }

    std::vector<num::idx> candidates;
    for (num::idx i = 0; i < n; ++i) {
        if (!protected_set.contains(i)) {
            candidates.push_back(i);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&](num::idx a, num::idx b) {
        return result.state_losses[a] < result.state_losses[b];
    });

    num::idx target_capacity = options.target_capacity;
    if (target_capacity == 0 || target_capacity > n) {
        target_capacity = n;
    }
    if (target_capacity < protected_set.size()) {
        target_capacity = static_cast<num::idx>(protected_set.size());
    }

    const num::idx needed_removals = (n > target_capacity) ? (n - target_capacity) : 0;
    const double total_exit_time = result.diagnostics.initial_mean_exit_time;
    const double max_allowable_loss = options.tolerance * std::max(1e-12, total_exit_time);

    std::unordered_set<num::idx> remove_set;
    double accumulated_loss = 0.0;

    for (num::idx candidate : candidates) {
        const double loss = result.state_losses[candidate];
        const bool needed_for_capacity = (remove_set.size() < needed_removals);
        const bool allowed_by_tolerance = (accumulated_loss + loss <= max_allowable_loss);

        if (needed_for_capacity || (options.target_capacity == 0 && allowed_by_tolerance)) {
            remove_set.insert(candidate);
            accumulated_loss += loss;
            if (remove_set.size() >= candidates.size()) {
                break;
            }
        }
    }

    for (num::idx i = 0; i < n; ++i) {
        if (remove_set.contains(i)) {
            result.shed_indices.push_back(i);
        } else {
            result.kept_indices.push_back(i);
        }
    }

    result.diagnostics.shed_states = result.shed_indices.size();
    result.diagnostics.remaining_states = result.kept_indices.size();
    result.diagnostics.total_estimated_loss = accumulated_loss;
    result.diagnostics.relative_loss = accumulated_loss / std::max(1e-12, total_exit_time);

    return result;
}

void prune_state_space(cme::StateSpace &states, const Subnetwork &subnetwork,
                       const num::Vector &entrance,
                       const std::unordered_set<markovkit::State> &protected_states,
                       const SheddingOptions &options, SheddingDiagnostics *diagnostics) {
    const num::idx n = subnetwork.size();
    if (states.size() != static_cast<int>(n)) {
        throw std::invalid_argument("state space size does not match subnetwork size");
    }

    const auto occupation = subnetwork.occupation(entrance);
    std::vector<num::idx> protected_indices;
    for (const auto &prot : protected_states) {
        int idx = states.find(prot);
        if (idx >= 0) {
            protected_indices.push_back(static_cast<num::idx>(idx));
        }
    }

    auto result = shed_states(subnetwork, occupation, protected_indices, options);
    if (diagnostics) {
        *diagnostics = result.diagnostics;
    }

    if (result.shed_indices.empty()) {
        return;
    }

    std::vector<bool> remove_mask(static_cast<std::size_t>(n), false);
    for (num::idx shed_idx : result.shed_indices) {
        remove_mask[static_cast<std::size_t>(shed_idx)] = true;
    }
    states.remove_states(remove_mask);
}

} // namespace else_sim
