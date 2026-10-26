// sample X_{tau-} by walking through triangular M-matrix factors
#pragma once

#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu.hpp"
#include "stochastic/categorical.hpp"
#include "subsweep/exit.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace subsweep {

namespace detail {

inline real nonnegative_factor_weight(real value, real scale) {
    real tolerance = 1e-12 * std::max(real{1.0}, scale);
    if (value < -tolerance) {
        throw std::invalid_argument(
            "factor sampling requires nonpositive off-diagonal factor entries");
    }
    return std::max(real{0.0}, value);
}

} // namespace detail

/// One two-stage factor draw and its sweep-duration contribution.
///
/// The duration contribution is not a sampled physical exit time. Conditional on the
/// entrance and last interior state, its expectation is the expected sweep duration.
struct factor_sweep_sample {
    idx factor_index;
    idx last_interior;
    real duration_contribution;
};

// exact two-stage samples of X_{tau-} from triangular factors of R
class factor_sampler {
  public:
    factor_sampler(const num::lu_result<real> &LU, view<const real> w) : n_(LU.size()) {
        if (w.size() != n_ || LU.singular) {
            throw std::invalid_argument("factor and exit-rate dimensions must agree");
        }
        if (!LU.swaps.empty()) {
            throw std::invalid_argument("factor sampling requires an unpivoted LU factor");
        }
        initialize(
            w, [&](idx i, idx j) { return LU.LU(i, j); },
            [&](idx i, idx j) { return i == j ? real{1.0} : LU.LU(i, j); });
    }

    factor_sampler(const num::cholesky_result &C, view<const real> w) : n_(C.L.rows()) {
        if (w.size() != n_ || C.L.cols() != n_ || !C.positive_definite) {
            throw std::invalid_argument("factor and exit-rate dimensions must agree");
        }

        initialize(
            w, [&](idx i, idx j) { return C.L(j, i); }, [&](idx i, idx j) { return C.L(i, j); });
    }

    [[nodiscard]] idx size() const { return n_; }

    // Include the contributions needed for conditional expected sweep durations.
    [[nodiscard]] static factor_sampler with_duration_contributions(const num::lu_result<real> &LU,
                                                                    view<const real> w) {
        factor_sampler sampler(LU, w);
        sampler.prepare_duration_contributions(LU);
        return sampler;
    }

    // Run the U-chain from i and return the factor index where it exits.
    template <typename RNG>
    [[nodiscard]] idx sample_u_chain_exit(idx i, RNG &random) {
        if (i >= n_) {
            throw std::out_of_range("entrance index is outside the factor");
        }
        idx l = i;
        while (idx choice = upper_draw_[l](random)) {
            l += choice;
        }
        return l;
    }

    // Run the L-chain from l and return the last interior state j where it exits.
    template <typename RNG>
    [[nodiscard]] idx sample_l_chain_exit(idx l, RNG &random) {
        idx j = l;
        while (true) {
            idx choice = lower_draw_[j](random);
            if (choice == j) {
                return j;
            }
            j = choice;
        }
    }

    // Sample the last interior state X_{tau-} for entrance state i.
    template <typename RNG>
    [[nodiscard]] idx sample_last_interior(idx i, RNG &random) {
        return sample_l_chain_exit(sample_u_chain_exit(i, random), random);
    }

    // Draw the same two stages and accumulate -d/ds log of their local probabilities.
    template <typename RNG>
    [[nodiscard]] factor_sweep_sample sample_with_duration(idx i, RNG &random) {
        if (i >= n_) {
            throw std::out_of_range("entrance index is outside the factor");
        }
        if (u_duration_.empty()) {
            throw std::logic_error("duration contributions require an unpivoted LU factor");
        }

        idx l = i;
        real duration = 0.0;
        while (true) {
            idx choice = upper_draw_[l](random);
            duration += u_duration_[l][choice];
            if (choice == 0) {
                break;
            }
            l += choice;
        }

        idx j = l;
        while (true) {
            idx choice = lower_draw_[j](random);
            duration += l_duration_[j][choice];
            if (choice == j) {
                return {l, j, duration};
            }
            j = choice;
        }
    }

    // Draw many exchangeable walkers and return their last-interior-state counts.
    template <typename RNG>
    [[nodiscard]] array<idx> sample_last_interior_counts(idx i, idx walkers, RNG &random) const {
        if (i >= n_) {
            throw std::out_of_range("entrance index is outside the factor");
        }

        array<idx> entrance(n_, 0);
        entrance[i] = walkers;
        return sample_last_interior_counts(entrance, random);
    }

    // Apply the same two stages to a population distributed over the entrance indices.
    template <typename RNG>
    [[nodiscard]] array<idx> sample_last_interior_counts(view<const idx> entrance,
                                                         RNG &random) const {
        if (entrance.size() != n_) {
            throw std::invalid_argument("entrance counts must have one entry per factor index");
        }

        array<idx> at_state(entrance.begin(), entrance.end());
        array<idx> factor_index(n_, 0), last_interior(n_, 0);
        for (idx j = 0; j < n_; ++j) {
            if (at_state[j] == 0) {
                continue;
            }
            array<idx> count = num::sample_multinomial(upper_weights_[j], at_state[j], random);
            factor_index[j] += count[0];
            for (idx choice = 1; choice < count.size(); ++choice) {
                at_state[j + choice] += count[choice];
            }
        }

        at_state = std::move(factor_index);
        for (idx j = n_; j-- > 0;) {
            if (at_state[j] == 0) {
                continue;
            }
            array<idx> count = num::sample_multinomial(lower_weights_[j], at_state[j], random);
            for (idx k = 0; k < j; ++k) {
                at_state[k] += count[k];
            }
            last_interior[j] += count[j];
        }
        return last_interior;
    }

  private:
    template <typename Upper, typename Lower>
    void initialize(view<const real> w, Upper U, Lower L) {
        // z = U 1 = L^{-1} w
        num::vec<real> z(n_, 0.0);
        for (idx j = 0; j < n_; ++j) {
            for (idx k = j; k < n_; ++k) {
                z[j] += U(j, k);
            }
        }

        upper_weights_.reserve(n_);
        lower_weights_.reserve(n_);
        for (idx j = 0; j < n_; ++j) {
            real U_jj = U(j, j);
            if (!(U_jj > 0.0) || !(L(j, j) > 0.0)) {
                throw std::invalid_argument("factor sampling requires positive pivots");
            }

            // stop at j with weight z_j, or move upward to k with weight -U_jk
            num::vec<real> upper(n_ - j, 0.0);
            upper[0] = detail::nonnegative_factor_weight(z[j], U_jj);
            for (idx k = j + 1; k < n_; ++k) {
                upper[k - j] = detail::nonnegative_factor_weight(-U(j, k), U_jj);
            }
            upper_weights_.push_back(std::move(upper));

            // return j with weight w_j, or move downward to k with weight -L_jk z_k
            num::vec<real> lower(j + 1, 0.0);
            if (z[j] > 0.0) {
                real scale = std::abs(L(j, j) * z[j]);
                for (idx k = 0; k < j; ++k) {
                    lower[k] = detail::nonnegative_factor_weight(-L(j, k) * z[k], scale);
                }
                lower[j] = detail::nonnegative_factor_weight(w[j], scale);
            } else {
                // this index cannot terminate the upper walk
                lower[j] = 1.0;
            }
            lower_weights_.push_back(std::move(lower));
        }
        prepare_draws();
    }

    void prepare_draws() {
        upper_draw_.reserve(n_);
        lower_draw_.reserve(n_);
        for (idx j = 0; j < n_; ++j) {
            upper_draw_.emplace_back(upper_weights_[j].span());
            lower_draw_.emplace_back(lower_weights_[j].span());
        }
    }

    void prepare_duration_contributions(const num::lu_result<real> &LU) {
        const num::mat<real> &F = LU.LU;
        num::mat<real> derivative = num::lu_diagonal_shift_derivative(LU);
        num::vec<real> z(n_, 0.0), z_derivative(n_, 0.0);
        for (idx j = 0; j < n_; ++j) {
            for (idx k = j; k < n_; ++k) {
                z[j] += F(j, k);
                z_derivative[j] += derivative(j, k);
            }
        }

        u_duration_.reserve(n_);
        l_duration_.reserve(n_);
        for (idx j = 0; j < n_; ++j) {
            num::vec<real> upper(n_ - j, 0.0);
            if (upper_weights_[j][0] > 0.0) {
                upper[0] = derivative(j, j) / F(j, j) - z_derivative[j] / z[j];
            }
            for (idx k = j + 1; k < n_; ++k) {
                if (upper_weights_[j][k - j] > 0.0) {
                    upper[k - j] = derivative(j, j) / F(j, j) - derivative(j, k) / F(j, k);
                }
            }
            u_duration_.push_back(std::move(upper));

            num::vec<real> lower(j + 1, 0.0);
            for (idx k = 0; k < j; ++k) {
                if (lower_weights_[j][k] > 0.0) {
                    lower[k] = -derivative(j, k) / F(j, k) - z_derivative[k] / z[k] +
                               z_derivative[j] / z[j];
                }
            }
            if (lower_weights_[j][j] > 0.0 && z[j] > 0.0) {
                lower[j] = z_derivative[j] / z[j];
            }
            l_duration_.push_back(std::move(lower));
        }
    }

    idx n_;
    array<num::vec<real>> upper_weights_;
    array<num::vec<real>> lower_weights_;
    array<num::categorical_sampler> upper_draw_;
    array<num::categorical_sampler> lower_draw_;
    array<num::vec<real>> u_duration_;
    array<num::vec<real>> l_duration_;
};

// Apply one population sweep by sampling the two triangular chains.
template <typename State, detail::oracle<State> Rates, typename RNG>
[[nodiscard]] unordered_map<State, idx>
sample_boundary_counts(const Rates &rates, const subnetwork<State> &D, view<const real> rho,
                       idx walkers, const factor_sampler &factors, RNG &random) {
    if (rho.size() != D.size() || factors.size() != D.size()) {
        throw std::invalid_argument(
            "rho, the factor sampler, and the subnetwork must have the same size");
    }
    array<idx> entrance = num::sample_multinomial(rho, walkers, random);
    array<idx> last_interior = factors.sample_last_interior_counts(entrance, random);
    return sample_first_exterior_counts(rates, D, last_interior, random);
}

namespace detail {

// x with L x = b, for the unit lower factor of a packed LU; `transpose` solves L^T x = b
inline num::vec<real> unit_lower_solve(const num::mat<real> &LU, num::vec<real> x, bool transpose) {
    idx n = LU.rows();
    if (!transpose) {
        for (idx r = 0; r < n; ++r) {
            for (idx k = 0; k < r; ++k) {
                x[r] -= LU(r, k) * x[k];
            }
        }
    } else {
        for (idx r = n; r-- > 0;) {
            for (idx k = r + 1; k < n; ++k) {
                x[r] -= LU(k, r) * x[k];
            }
        }
    }
    return x;
}

// x with U x = b, for the upper factor of a packed LU; `transpose` solves U^T x = b
inline num::vec<real> upper_solve(const num::mat<real> &LU, num::vec<real> x, bool transpose) {
    idx n = LU.rows();
    if (!transpose) {
        for (idx r = n; r-- > 0;) {
            for (idx k = r + 1; k < n; ++k) {
                x[r] -= LU(r, k) * x[k];
            }
            x[r] /= LU(r, r);
        }
    } else {
        for (idx r = 0; r < n; ++r) {
            for (idx k = 0; k < r; ++k) {
                x[r] -= LU(k, r) * x[k];
            }
            x[r] /= LU(r, r);
        }
    }
    return x;
}

inline num::vec<real> unit_vector(idx n, idx k) {
    num::vec<real> e(n, 0.0);
    e[k] = 1.0;
    return e;
}

} // namespace detail

// T^up_il = -d/ds log[U_s^{-1}]_il at s = 0, for R + sI = L_s U_s with L_s unit
// lower. With X = L^{-1} U^{-1}, T^up_il = sum_{k<=l} (U^{-1})_ik X_kl / (U^{-1})_il
[[nodiscard]] inline real u_chain_duration_contribution(const num::lu_result<real> &LU, idx i,
                                                        idx l) {
    const num::mat<real> &F = LU.LU;
    idx n = LU.size();
    num::vec<real> U_row = detail::upper_solve(F, detail::unit_vector(n, i), true);
    num::vec<real> U_column = detail::upper_solve(F, detail::unit_vector(n, l), false);
    num::vec<real> X_column = detail::unit_lower_solve(F, U_column, false);
    real sum = 0.0;
    for (idx k = 0; k <= l; ++k) {
        sum += U_row[k] * X_column[k];
    }
    return sum / U_row[l];
}

// T^down_lj = -d/ds log[L_s^{-1}]_lj at s = 0, which is
// sum_{k<l} X_lk (L^{-1})_kj / (L^{-1})_lj
[[nodiscard]] inline real l_chain_duration_contribution(const num::lu_result<real> &LU, idx l,
                                                        idx j) {
    const num::mat<real> &F = LU.LU;
    idx n = LU.size();
    num::vec<real> X_row =
        detail::upper_solve(F, detail::unit_lower_solve(F, detail::unit_vector(n, l), true), true);
    num::vec<real> L_column = detail::unit_lower_solve(F, detail::unit_vector(n, j), false);
    real sum = 0.0;
    for (idx k = 0; k < l; ++k) {
        sum += X_row[k] * L_column[k];
    }
    return sum / L_column[l];
}

// E[tau | X_0=i, X_{tau-}=j] = sum_l pi(l | i, j) (T^up_il + T^down_lj), with
// pi(l | i, j) = (U^{-1})_il (L^{-1})_lj / Z_ij
[[nodiscard]] inline real conditional_expected_sweep_duration(const num::lu_result<real> &LU, idx i,
                                                              idx j) {
    const num::mat<real> &F = LU.LU;
    idx n = LU.size();
    num::vec<real> U_row = detail::upper_solve(F, detail::unit_vector(n, i), true);
    num::vec<real> L_column = detail::unit_lower_solve(F, detail::unit_vector(n, j), false);
    real Z_ij = 0.0, sum = 0.0;
    for (idx l = std::max(i, j); l < n; ++l) {
        real weight = U_row[l] * L_column[l];
        if (weight != 0.0) {
            Z_ij += weight;
            sum += weight * (u_chain_duration_contribution(LU, i, l) +
                             l_chain_duration_contribution(LU, l, j));
        }
    }
    return Z_ij > 0.0 ? sum / Z_ij : 0.0;
}

} // namespace subsweep
