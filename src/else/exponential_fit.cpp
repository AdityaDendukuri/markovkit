/// @file src/else/exponential_fit.cpp
/// @brief Compiled fit_exponential.
#include "markovkit/else/exponential_fit.hpp"

namespace markovkit::else_solver {

ExponentialFitResult fit_exponential(const std::vector<double>& times) {
    if (times.empty()) throw std::invalid_argument("fit_exponential: empty time vector");

    double sum = 0.0;
    for (double t : times) sum += t;
    double mean_t = sum / static_cast<double>(times.size());

    if (mean_t <= 0.0) throw std::domain_error("fit_exponential: mean time must be positive");

    double lambda = 1.0 / mean_t;

    double res = 0.0;
    for (double t : times) {
        double expected_cdf = 1.0 - std::exp(-lambda * t);
        res += expected_cdf * expected_cdf;
    }

    return {lambda, res};
}

} // namespace markovkit::else_solver
