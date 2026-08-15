/// @file else/subnetwork.hpp
/// @brief Exact Linear Subnetwork Elimination (ELSE) subnetwork & Schur shedding.
#pragma once

#include <numerics.hpp>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace markovkit::else_solver {

/// @brief Subnetwork state representation and fundamental matrix Z = -R^{-1}.
struct Subnetwork {
    num::Matrix R; ///< Killed subnetwork generator matrix R (negative diagonal)
    num::Matrix Z; ///< Fundamental matrix Z = -R^{-1}

    explicit Subnetwork(num::Matrix generator) : R(std::move(generator)) {
        num::idx n = R.rows();
        if (R.cols() != n) throw std::invalid_argument("Subnetwork: generator must be square");

        // Negate R for inversion: -R
        num::Matrix neg_R(n, n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            for (num::idx j = 0; j < n; ++j) {
                neg_R(i, j) = -R(i, j);
            }
        }

        // Compute Z = (-R)^{-1} via LU factorization
        auto lu_fact = num::lu(neg_R);
        Z = num::Matrix(n, n, 0.0);
        for (num::idx j = 0; j < n; ++j) {
            num::Vector e_j(n, 0.0);
            e_j[j] = 1.0;
            num::Vector col_j(n, 0.0);
            num::lu_solve(lu_fact, e_j, col_j);
            for (num::idx i = 0; i < n; ++i) {
                Z(i, j) = col_j[i];
            }
        }
    }

    /// @brief O(N^2) Schur rank-1 state shedding update:
    ///        Z_new = Z - Z_{:,k} Z_{k,:} / Z_{k,k}
    void shed_state(num::idx k) {
        num::idx n = Z.rows();
        if (k >= n) throw std::out_of_range("shed_state: index k out of bounds");

        double z_kk = Z(k, k);
        if (std::abs(z_kk) < 1e-14) throw std::runtime_error("shed_state: Z(k,k) is zero");

        num::Matrix Z_new(n - 1, n - 1, 0.0);
        for (num::idx i = 0, ni = 0; i < n; ++i) {
            if (i == k) continue;
            for (num::idx j = 0, nj = 0; j < n; ++j) {
                if (j == k) continue;
                Z_new(ni, nj) = Z(i, j) - (Z(i, k) * Z(k, j)) / z_kk;
                ++nj;
            }
            ++ni;
        }

        // Remove row k and column k from R
        num::Matrix R_new(n - 1, n - 1, 0.0);
        for (num::idx i = 0, ni = 0; i < n; ++i) {
            if (i == k) continue;
            for (num::idx j = 0, nj = 0; j < n; ++j) {
                if (j == k) continue;
                R_new(ni, nj) = R(i, j);
                ++nj;
            }
            ++ni;
        }

        R = std::move(R_new);
        Z = std::move(Z_new);
    }
};

} // namespace markovkit::else_solver
