/// @file else/exponential_fit.hpp
/// @brief Exponential distribution bisection fit for first-passage times.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace markovkit::else_solver {

/// @brief Result of exponential fitting: lambda parameter and fit residual.
struct ExponentialFitResult {
  double lambda;   ///< Estimated rate parameter (lambda = 1 / mean_time)
  double residual; ///< Sum of squared errors
};

/// @brief Fit an exponential distribution P(T <= t) = 1 - exp(-lambda * t) to
/// empirical times.
ExponentialFitResult fit_exponential(const std::vector<double>& times);

} // namespace markovkit::else_solver
