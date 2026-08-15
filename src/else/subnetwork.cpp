/// @file src/else/subnetwork.cpp
/// @brief Compiled Subnetwork methods fully delegating linear algebra to numerics routines.
#include "markovkit/else/subnetwork.hpp"

namespace markovkit::else_solver {

Subnetwork::Subnetwork(num::Matrix generator) : R(std::move(generator)) {
    num::idx n = R.rows();
    if (R.cols() != n) throw std::invalid_argument("Subnetwork: generator must be square");

    // Delegate matrix negation and LU inversion entirely to numerics: Z = (-R)^{-1}
    num::Matrix neg_R(n, n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        for (num::idx j = 0; j < n; ++j) {
            neg_R(i, j) = -R(i, j);
        }
    }

    auto lu_fact = num::lu(neg_R);
    Z = num::lu_inv(lu_fact);
}

num::Vector Subnetwork::occupation(const num::Vector& entrance) const {
    num::Vector u(Z.rows(), 0.0);
    num::operators::DenseOp Zop(Z);
    Zop.apply(entrance, u); // Heavy lifting matrix-vector product via BLAS cblas_dgemv
    return u;
}

std::vector<double> Subnetwork::cut_time_losses(const num::Vector& entrance) const {
    num::idx n = Z.rows();
    num::Vector u = occupation(entrance);

    std::vector<double> losses(n, 0.0);
    for (num::idx k = 0; k < n; ++k) {
        double col_sum = 0.0;
        for (num::idx j = 0; j < n; ++j) {
            col_sum += Z(j, k);
        }
        double z_kk = Z(k, k);
        losses[k] = (z_kk > 0.0) ? (u[k] * col_sum / z_kk) : 0.0;
    }
    return losses;
}

std::vector<double> Subnetwork::schur_losses(const num::Vector& entrance) const {
    num::idx n = Z.rows();
    num::Vector u = occupation(entrance);

    std::vector<double> losses(n, 0.0);
    for (num::idx k = 0; k < n; ++k) {
        double z_kk = Z(k, k);
        losses[k] = (z_kk > 0.0) ? (u[k] / z_kk) : 0.0;
    }
    return losses;
}

std::vector<double> Subnetwork::cut_norm_losses(const num::Vector& entrance) const {
    num::idx n = Z.rows();
    num::Vector u = occupation(entrance);

    std::vector<double> losses(n, 0.0);
    for (num::idx k = 0; k < n; ++k) {
        double z_kk = Z(k, k);
        double col_sum = 0.0;
        for (num::idx j = 0; j < n; ++j) col_sum += Z(j, k);
        losses[k] = (z_kk > 0.0) ? (u[k] * std::sqrt(col_sum) / z_kk) : 0.0;
    }
    return losses;
}

void Subnetwork::shed_state(num::idx k) {
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

} // namespace markovkit::else_solver
