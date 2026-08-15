/// @file else/exponential_fit.hpp
/// @brief Exponential distribution bisection fit for first-passage times.
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace markovkit::else_solver {

/// @brief Result of exponential fitting: lambda parameter and fit residual.
struct ExponentialFitResult {
    double lambda;   ///< Estimated rate parameter (lambda = 1 / mean_time)
    double residual; ///< Sum of squared errors
};

/// @brief Fit an exponential distribution P(T <= t) = 1 - exp(-lambda * t) to empirical times.
inline ExponentialFitResult fit_exponential(const std::vector<double>& times) {
    if (times.empty()) throw std::invalid_argument("fit_exponential: empty time vector");

    double sum = 0.0;
    for (double t : times) sum += t;
    double mean_t = sum / static_cast<double>(times.size());

    if (mean_t <= 0.0) throw std::domain_error("fit_exponential: mean time must be positive");

    double lambda = 1.0 / mean_t;

    // Compute residual
    double res = 0.0;
    for (double t : times) {
        double expected_cdf = 1.0 - std::exp(-lambda * t);
        res += expected_cdf * expected_cdf;
    }

    return {lambda, res};
}

} // namespace markovkit::else_solver
