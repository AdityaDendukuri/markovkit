// A dense inverse retained across subnetwork sweeps.
#pragma once

#include "container/matrix_expr.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solve.hpp"
#include <algorithm>
#include <stdexcept>

namespace subsweep {

using num::array;
using num::idx;
using num::real;
using num::view;

class dense_inverse {
  public:
    dense_inverse(num::mat<real> R, const dense_inverse *previous, view<const idx> changed)
        : R_(std::move(R)) {
        if (!reuse(previous, changed)) {
            rebuild();
        }
    }

    dense_inverse(num::mat<real> R, view<const real> h, const dense_inverse *previous,
                  view<const idx> changed)
        : R_(std::move(R)) {
        if (!reuse(previous, changed)) {
            rebuild(h);
        }
    }

    [[nodiscard]] idx size() const { return R_.rows(); }
    [[nodiscard]] bool reused() const { return reused_; }
    [[nodiscard]] const num::mat<real> &matrix() const { return Z_; }

    friend void solve(const dense_inverse &Z, const num::vec<real> &b, num::vec<real> &x) {
        x = num::vec<real>(Z.size(), 0.0);
        num::matvec(Z.Z_, b, x);
    }

    friend void solve(const dense_inverse &Z, const num::mat<real> &B, num::mat<real> &X) {
        X = num::mat<real>(Z.size(), B.cols(), 0.0);
        num::matmul(Z.Z_, B, X);
    }

    friend void solve_transpose(const dense_inverse &Z, const num::vec<real> &b,
                                num::vec<real> &x) {
        if (b.size() != Z.size()) {
            throw std::invalid_argument("dense inverse: right-hand side has the wrong size");
        }
        num::vec<real> result(Z.size(), 0.0);
        for (idx j = 0; j < Z.size(); ++j) {
            for (idx i = 0; i < Z.size(); ++i) {
                result[i] += Z.Z_(j, i) * b[j];
            }
        }
        x = std::move(result);
    }

    friend void solve_transpose(const dense_inverse &Z, const num::mat<real> &B,
                                num::mat<real> &X) {
        if (B.rows() != Z.size()) {
            throw std::invalid_argument("dense inverse: right-hand side has the wrong size");
        }
        num::mat<real> result(Z.size(), B.cols(), 0.0);
        num::blas::gemm(1.0, Z.Z_, true, B, false, 0.0, result);
        X = std::move(result);
    }

  private:
    static constexpr idx rebuild_after = 128;

    bool reuse(const dense_inverse *previous, view<const idx> changed) {
        if (!previous || previous->size() != size()) {
            return false;
        }
        if (changed.empty()) {
            Z_ = previous->Z_;
            updates_ = previous->updates_;
            reused_ = true;
            return true;
        }
        if (changed.size() > 3 || previous->updates_ >= rebuild_after ||
            !update(*previous, changed)) {
            return false;
        }
        updates_ = previous->updates_ + 1;
        reused_ = true;
        return true;
    }

    void rebuild() {
        auto factor = num::lu(R_, num::no_pivot);
        if (factor.singular) {
            throw std::runtime_error("dense inverse: matrix is singular");
        }
        Z_ = num::solve(factor, num::identity(size()));
        updates_ = 0;
        reused_ = false;
    }

    void rebuild(view<const real> h) {
        if (h.size() != size()) {
            throw std::invalid_argument("dense inverse: one similarity weight is needed per row");
        }
        num::mat<real> symmetric(size(), size(), 0.0);
        for (idx i = 0; i < size(); ++i) {
            if (!(h[i] > 0.0)) {
                throw std::invalid_argument("dense inverse: similarity weights must be positive");
            }
            for (idx j = 0; j < size(); ++j) {
                symmetric(i, j) = R_(i, j) * h[i] / h[j];
            }
        }
        auto factor = num::cholesky(num::assume_spd(symmetric));
        if (!factor.positive_definite) {
            throw std::runtime_error("dense inverse: symmetrized matrix is not positive definite");
        }
        Z_ = num::solve(factor, num::identity(size()));
        for (idx i = 0; i < size(); ++i) {
            for (idx j = 0; j < size(); ++j) {
                Z_(i, j) *= h[j] / h[i];
            }
        }
        updates_ = 0;
        reused_ = false;
    }

    bool update(const dense_inverse &previous, view<const idx> changed) {
        const idx rank = 2 * changed.size();
        num::mat<real> P(size(), rank, 0.0);
        num::mat<real> Q(size(), rank, 0.0);
        for (idx k = 0; k < changed.size(); ++k) {
            if (changed[k] >= size()) {
                throw std::out_of_range("dense inverse: changed row is outside the matrix");
            }
            const idx row = changed[k];
            P(row, k) = 1.0;
            for (idx j = 0; j < size(); ++j) {
                Q(j, k) = R_(row, j) - previous.R_(row, j);
            }
            Q(row, changed.size() + k) = 1.0;
            for (idx i = 0; i < size(); ++i) {
                if (std::find(changed.begin(), changed.end(), i) == changed.end()) {
                    P(i, changed.size() + k) = R_(i, row) - previous.R_(i, row);
                }
            }
        }

        num::mat<real> ZP(size(), rank, 0.0);
        num::blas::gemm(1.0, previous.Z_, false, P, false, 0.0, ZP);
        num::mat<real> QtZ(rank, size(), 0.0);
        num::blas::gemm(1.0, Q, true, previous.Z_, false, 0.0, QtZ);
        num::mat<real> reduced_matrix(rank, rank, 0.0);
        num::blas::gemm(1.0, Q, true, ZP, false, 0.0, reduced_matrix);
        for (idx k = 0; k < rank; ++k) {
            reduced_matrix(k, k) += 1.0;
        }
        auto reduced = num::lu(reduced_matrix);
        if (reduced.singular) {
            return false;
        }
        num::mat<real> correction = num::solve(reduced, QtZ);
        Z_ = previous.Z_;
        num::blas::gemm(-1.0, ZP, false, correction, false, 1.0, Z_);
        return true;
    }

    num::mat<real> R_;
    num::mat<real> Z_;
    idx updates_ = 0;
    bool reused_ = false;
};

} // namespace subsweep
