/// @file else.hpp
/// @brief Exact Linear Subnetwork Elimination (ELSE) for Continuous-Time Markov Chains (CTMC).
#pragma once

#include <numerics.hpp>
#include <vector>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <algorithm>

namespace markovkit::else_solver {

struct ExponentialFitOptions {
    double tolerance = 1e-12;
    std::size_t max_iters = 1000;
};

/// Fit a^T exp(-b * t) == c in log-space via bisection
inline double exponential_fit(const num::Vector& a, const num::Vector& b, double c, const ExponentialFitOptions& ops = {}) {
    double t = 0.0;
    double sum_ab = 0.0;
    for (num::idx i = 0; i < a.size(); ++i) {
        if (b[i] != 0.0) sum_ab += -a[i] / b[i];
    }
    t = sum_ab;
    double lo = 0.0, up = (t > 0.0) ? t : 1.0;

    for (std::size_t iter = 0; iter < ops.max_iters; ++iter) {
        double num = 0.0;
        for (num::idx i = 0; i < a.size(); ++i) {
            num += a[i] * std::exp(-up * b[i]);
        }
        if (num < c) break;
        lo = up;
        up *= 2.0;
    }

    for (std::size_t iter = 0; iter < ops.max_iters; ++iter) {
        t = 0.5 * (lo + up);
        double num = 0.0;
        for (num::idx i = 0; i < a.size(); ++i) {
            num += a[i] * std::exp(-t * b[i]);
        }
        if (std::abs(num - c) < ops.tolerance) return t;

        if (num > c) {
            lo = t;
        } else {
            up = t;
        }
    }
    return t;
}

/// Discrete Reference Subnetwork representing a transient CTMC subnetwork
class DiscreteReferenceSubnetwork {
protected:
    num::Matrix R_; // Rate matrix (negative definite)
    num::Matrix Z_; // Fundamental matrix Z = -R^{-1} (positive definite)
    num::Vector p_; // Stationarity/boundary probability weights

public:
    DiscreteReferenceSubnetwork() = default;
    explicit DiscreteReferenceSubnetwork(num::idx n) : R_(n, n, 0.0), Z_(n, n, 0.0), p_(n, 0.0) {}

    num::idx length() const { return R_.rows(); }

    const num::Matrix& rate_matrix() const { return R_; }
    const num::Matrix& fundamental_matrix() const { return Z_; }
    const num::Vector& weights() const { return p_; }

    void set_rate_matrix(const num::Matrix& R) {
        R_ = R;
        refresh();
    }

    void refresh() {
        num::idx n = R_.rows();
        if (n == 0) return;
        num::Matrix negR(n, n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            for (num::idx j = 0; j < n; ++j) {
                negR(i, j) = -R_(i, j);
            }
        }
        // Invert negR via LU factorization
        auto fact = num::lu(negR);
        Z_ = num::Matrix(n, n, 0.0);
        for (num::idx j = 0; j < n; ++j) {
            num::Vector e(n, 0.0);
            e[j] = 1.0;
            num::Vector col;
            num::lu_solve(fact, e, col);
            for (num::idx i = 0; i < n; ++i) {
                Z_(i, j) = col[i];
            }
        }
    }

    /// O(N) Unconditional mean escape time from state i
    double escape_time(num::idx i) const {
        double sum = 0.0;
        for (num::idx j = 0; j < Z_.cols(); ++j) {
            sum += Z_(i, j);
        }
        return sum;
    }
};

} // namespace markovkit::else_solver
