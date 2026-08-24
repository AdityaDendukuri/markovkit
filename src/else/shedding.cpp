/// @file src/else/shedding.cpp
/// @brief Implementation of the 3-method state shedding framework for ELSE.
#include "else/shedding.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace else_sim {

namespace {

// Method 3: Fastest O(n) Expected Visits Approximation
std::vector<double> compute_expected_visits_losses(const Subnetwork &subnetwork,
                                                   const num::Vector &occupation) {
    const num::idx n = subnetwork.size();
    const auto &column_sums = subnetwork.inverse_column_sums();
    const auto &R = subnetwork.generator();
    std::vector<double> losses(n, 0.0);

    for (num::idx i = 0; i < n; ++i) {
        const double rate = std::max(1e-12, -R(i, i));
        // Loss \approx u_i * w_i / (1 / rate) = u_i * w_i * rate
        const double w_i = (i < column_sums.size()) ? column_sums[i] : (1.0 / rate);
        losses[i] = std::max(0.0, occupation[i] * w_i * rate);
    }
    return losses;
}

// Method 2: Non-Symmetric Subnetworks — Normalize, Symmetrize, & Sample Diagonal
std::vector<double> compute_symmetrized_losses(const Subnetwork &subnetwork,
                                               const num::Vector &occupation,
                                               const SheddingOptions &options) {
    const num::idx n = subnetwork.size();
    const auto &column_sums = subnetwork.inverse_column_sums();
    const auto &R = subnetwork.generator();

    // 1. Construct dense additive symmetrized operator S = 0.5 * ((-R) + (-R)^T)
    num::Matrix S(n, n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        for (num::idx j = 0; j < n; ++j) {
            S(i, j) = 0.5 * (-R(i, j) + -R(j, i));
        }
    }

    // Ensure strict positive definiteness
    for (num::idx i = 0; i < n; ++i) {
        double off_diag = 0.0;
        for (num::idx j = 0; j < n; ++j) {
            if (i != j) {
                off_diag += std::abs(S(i, j));
            }
        }
        S(i, i) = std::max(S(i, i), off_diag + 1e-8);
    }

    // 2. Extract inverse diagonal d_i = [S^-1]_{ii}
    std::vector<double> d(n, 1.0);
    auto chol = num::cholesky(num::linalg::assume_spd(S));

    if (chol.success && (n <= 256 || options.sample_probes == 0)) {
        // Exact selected inverse solves on Cholesky factor
        num::Vector e(n, 0.0);
        num::Vector y(n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            e[i] = 1.0;
            num::cholesky_solve(chol, e, y);
            d[i] = std::max(y[i], 1e-14);
            e[i] = 0.0;
        }
    } else if (chol.success && options.sample_probes > 0) {
        // Stochastic Hutchinson Rademacher diagonal probe sampling
        const num::idx M = options.sample_probes;
        std::mt19937_64 rng(42);
        std::bernoulli_distribution bern(0.5);
        std::vector<double> d_accum(n, 0.0);
        num::Vector z(n, 0.0);
        num::Vector y(n, 0.0);

        for (num::idx m = 0; m < M; ++m) {
            for (num::idx i = 0; i < n; ++i) {
                z[i] = bern(rng) ? 1.0 : -1.0;
            }
            num::cholesky_solve(chol, z, y);
            for (num::idx i = 0; i < n; ++i) {
                d_accum[i] += z[i] * y[i];
            }
        }

        for (num::idx i = 0; i < n; ++i) {
            d[i] = std::max(d_accum[i] / static_cast<double>(M), 1e-14);
        }
    } else {
        // Fallback to LU solves if Cholesky is not strictly positive definite
        auto lu_factor = num::lu(S);
        num::Vector e(n, 0.0);
        num::Vector y(n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            e[i] = 1.0;
            num::lu_solve(lu_factor, e, y);
            d[i] = std::max(y[i], 1e-14);
            e[i] = 0.0;
        }
    }

    // 3. Compute shedding losses \Delta \tau_i = u_i * w_i / d_i
    std::vector<double> losses(n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        const double w_i = (i < column_sums.size()) ? column_sums[i] : 1.0;
        const double d_i = std::max(d[i], 1e-14);
        losses[i] = std::max(0.0, occupation[i] * (w_i / d_i));
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
        if (subnetwork.is_reversible()) {
            effective = SheddingMethod::CholeskyWoodbury;
        } else if (n <= 512) {
            effective = SheddingMethod::NormalizedSymmetrized;
        } else {
            effective = SheddingMethod::ExpectedVisits;
        }
    }

    if (diagnostics) {
        diagnostics->requested_method = options.method;
        diagnostics->effective_method = effective;
        diagnostics->initial_states = n;
        diagnostics->initial_mean_exit_time = 0.0;
        for (num::idx i = 0; i < n; ++i) {
            diagnostics->initial_mean_exit_time += occupation[i];
        }
    }

    // Method 1: Cholesky Woodbury with floating-point error tracking & fallback
    if (effective == SheddingMethod::CholeskyWoodbury) {
        try {
            auto losses = subnetwork.cut_time_losses(occupation);
            double max_err = 0.0;
            bool numeric_issue = false;

            for (num::idx i = 0; i < n; ++i) {
                if (std::isnan(losses[i]) || std::isinf(losses[i]) || losses[i] < 0.0) {
                    numeric_issue = true;
                    max_err = std::max(max_err, 1.0);
                    break;
                }
            }

            if (diagnostics) {
                diagnostics->max_floating_point_error = max_err;
            }

            if (!numeric_issue && max_err <= options.error_threshold) {
                return losses;
            }

            // Numeric issue or error threshold breached -> Trigger Graceful Fallback
            if (options.fallback_to_stable) {
                if (diagnostics) {
                    diagnostics->fallback_triggered = true;
                    diagnostics->fallback_reason =
                        "Woodbury conditioning or non-finite values detected";
                    diagnostics->effective_method = SheddingMethod::NormalizedSymmetrized;
                }
                return compute_symmetrized_losses(subnetwork, occupation, options);
            }
            return losses;
        } catch (const std::exception &ex) {
            if (options.fallback_to_stable) {
                if (diagnostics) {
                    diagnostics->fallback_triggered = true;
                    diagnostics->fallback_reason = std::string("Woodbury solve threw: ") + ex.what();
                    diagnostics->effective_method = SheddingMethod::NormalizedSymmetrized;
                }
                return compute_symmetrized_losses(subnetwork, occupation, options);
            }
            throw;
        }
    }

    // Method 2: Symmetrization and diagonal sampling
    if (effective == SheddingMethod::NormalizedSymmetrized) {
        return compute_symmetrized_losses(subnetwork, occupation, options);
    }

    // Method 3: Expected visits
    return compute_expected_visits_losses(subnetwork, occupation);
}

SheddingResult shed_states(const Subnetwork &subnetwork,
                           const num::Vector &occupation,
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

    // Identify eligible candidates (unprotected states)
    std::vector<num::idx> candidate_indices;
    for (num::idx i = 0; i < n; ++i) {
        if (!protected_set.contains(i)) {
            candidate_indices.push_back(i);
        }
    }

    // Sort candidates by ascending estimated exit-time loss
    std::sort(candidate_indices.begin(), candidate_indices.end(),
              [&](num::idx a, num::idx b) { return result.state_losses[a] < result.state_losses[b]; });

    // Determine how many states to remove
    num::idx target_capacity = options.target_capacity;
    if (target_capacity == 0 || target_capacity > n) {
        target_capacity = n;
    }
    if (target_capacity < protected_set.size()) {
        target_capacity = static_cast<num::idx>(protected_set.size());
    }

    const num::idx needed_removals = (n > target_capacity) ? (n - target_capacity) : 0;
    const double total_mean_exit_time = result.diagnostics.initial_mean_exit_time;
    const double max_allowable_loss = options.tolerance * std::max(1e-12, total_mean_exit_time);

    std::unordered_set<num::idx> remove_set;
    double accumulated_loss = 0.0;

    for (num::idx candidate : candidate_indices) {
        const double loss = result.state_losses[candidate];
        const bool needed_for_capacity = (remove_set.size() < needed_removals);
        const bool allowed_by_tolerance = (accumulated_loss + loss <= max_allowable_loss);

        if (needed_for_capacity || (options.target_capacity == 0 && allowed_by_tolerance)) {
            remove_set.insert(candidate);
            accumulated_loss += loss;
            if (remove_set.size() >= candidate_indices.size()) {
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
    result.diagnostics.relative_loss =
        accumulated_loss / std::max(1e-12, total_mean_exit_time);

    return result;
}

void prune_state_space(cme::StateSpace &states,
                       const Subnetwork &subnetwork,
                       const num::Vector &entrance,
                       const std::unordered_set<markovkit::State> &protected_states,
                       const SheddingOptions &options,
                       SheddingDiagnostics *diagnostics) {
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
