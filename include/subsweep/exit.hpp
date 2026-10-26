// first-exit operations on Z = R^{-1} for one finite subnetwork D
#pragma once

#include "linear/eigen/jacobi_eig.hpp"
#include "linear/factorization/factor.hpp"
#include "stochastic/categorical.hpp"
#include "stochastic/multinomial.hpp"
#include "stochastic/rng.hpp"
#include "subsweep/subnetwork.hpp"
#include "subsweep/sweep.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace subsweep {

// the diagonal R_jj, the total rate out of each state
[[nodiscard]] inline num::vec<real> total_rates(const num::spmat &R) {
    num::vec<real> diagonal(R.n_rows(), 0.0);
    for (idx i = 0; i < R.n_rows(); ++i) {
        for (idx entry = R.row_ptr()[i]; entry < R.row_ptr()[i + 1]; ++entry) {
            if (R.col_idx()[entry] == i) {
                diagonal[i] += R.values()[entry];
            }
        }
    }
    return diagonal;
}

// w = R 1, the rate of leaving D from each state
[[nodiscard]] inline num::vec<real> exit_rates(const num::spmat &R) {
    num::vec<real> w(R.n_rows(), 0.0);
    for (idx i = 0; i < R.n_rows(); ++i) {
        for (idx entry = R.row_ptr()[i]; entry < R.row_ptr()[i + 1]; ++entry) {
            w[i] += R.values()[entry];
        }
        w[i] = std::max(real{0.0}, w[i]);
    }
    return w;
}

template <typename State>
[[nodiscard]] num::vec<real> total_rates(const subnetwork<State> &D) {
    auto rates = D.total_rates();
    return num::vec<real>(rates.begin(), rates.end());
}

template <typename State>
[[nodiscard]] num::vec<real> exit_rates(const subnetwork<State> &D) {
    return D.exit_rates();
}

// Pr(X_{tau-}=j | X_0=i) = Z_ij (R 1)_j
[[nodiscard]] inline num::vec<real> last_interior_law(view<const real> Z_i, view<const real> w) {
    if (Z_i.size() != w.size()) {
        throw std::invalid_argument("Z_i and R 1 must have the same size");
    }
    num::vec<real> probability(Z_i.size(), 0.0);
    real total = 0.0;
    for (idx j = 0; j < Z_i.size(); ++j) {
        probability[j] = std::max(real{0.0}, Z_i[j] * w[j]);
        total += probability[j];
    }
    if (!(total > 0.0)) {
        throw std::runtime_error("the finite state set has no reachable exit");
    }
    for (real &mass : probability) {
        mass /= total;
    }
    return probability;
}

// one sample from a discrete law on the rows of D
template <typename RNG>
[[nodiscard]] idx sample_last_interior_state(view<const real> probability, RNG &random) {
    return num::sample_categorical(probability, random);
}

template <typename RNG>
[[nodiscard]] array<idx> sample_last_interior_counts(view<const real> probability, idx draws,
                                                     RNG &random) {
    return num::sample_multinomial(probability, draws, random);
}

// The physical jump from the last interior state to the first exterior state.
template <typename State, detail::oracle<State> Rates, typename RNG>
[[nodiscard]] State sample_first_exterior_state(const Rates &rates, const subnetwork<State> &D,
                                                idx j, RNG &random) {
    if (j >= D.size()) {
        throw std::out_of_range("the last interior index is outside the subnetwork");
    }
    return rates.sample_exterior(D.state(j), j, D, random);
}

template <typename State>
struct reversible_sweep_sample {
    real exit_time;
    idx last_interior;
    State first_exterior;
    array<State> interior_states;
};

namespace detail {

inline num::vec<real> semigroup_row(const num::eigen_result &eigen, real time, idx row) {
    idx n = eigen.values.size();
    num::vec<real> result(n, 0.0);
    for (idx mode = 0; mode < n; ++mode) {
        real coefficient = eigen.vectors(row, mode) * std::exp(-time * eigen.values[mode]);
        for (idx column = 0; column < n; ++column) {
            result[column] += coefficient * eigen.vectors(column, mode);
        }
    }
    return result;
}

inline real conditional_survival(const num::eigen_result &eigen, idx entrance, idx last_interior,
                                 real time) {
    real numerator = 0.0;
    real denominator = 0.0;
    for (idx mode = 0; mode < eigen.values.size(); ++mode) {
        real coefficient =
            eigen.vectors(entrance, mode) * eigen.vectors(last_interior, mode) / eigen.values[mode];
        denominator += coefficient;
        numerator += coefficient * std::exp(-time * eigen.values[mode]);
    }
    return std::clamp(numerator / denominator, 0.0, 1.0);
}

template <typename RNG>
real sample_conditional_exit_time(const num::eigen_result &eigen, idx entrance, idx last_interior,
                                  RNG &random) {
    real target = 1.0 - std::uniform_real_distribution<real>(0.0, 1.0)(random);
    real lower = 0.0;
    real upper = 1e-16;
    while (conditional_survival(eigen, entrance, last_interior, upper) > target) {
        upper *= 2.0;
    }
    for (int iteration = 0; iteration < 56; ++iteration) {
        real midpoint = 0.5 * (lower + upper);
        if (conditional_survival(eigen, entrance, last_interior, midpoint) > target) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    return 0.5 * (lower + upper);
}

} // namespace detail

// Sample one exit from a finite reversible subnetwork and the chain state at
// each requested time before that exit. `h(x)` is any positive diagonal
// similarity that makes H R H^{-1} symmetric.
template <typename State, detail::oracle<State> Rates, typename Symmetrizer, typename RNG>
[[nodiscard]] reversible_sweep_sample<State>
sample_reversible_sweep(const Rates &rates, array<State> states, const State &entrance,
                        view<const real> observation_times, Symmetrizer h, RNG &random) {
    subnetwork<State> D = make_subnetwork(rates, std::move(states));
    auto entrance_row = D.find(entrance);
    if (!entrance_row) {
        throw std::invalid_argument("the entrance must belong to the subnetwork");
    }

    num::vec<real> diagonal(D.size());
    for (idx i = 0; i < D.size(); ++i) {
        diagonal[i] = h(D.state(i));
        if (!(diagonal[i] > 0.0)) {
            throw std::invalid_argument("the reversible similarity must be positive");
        }
    }

    auto Z = num::cholesky(D.matrix(), diagonal);
    num::vec<real> e_i(D.size(), 0.0);
    e_i[*entrance_row] = 1.0;
    num::vec<real> Z_i = num::solve(num::transpose(Z), e_i);
    idx last_interior =
        sample_last_interior_state(last_interior_law(Z_i, D.exit_rates()), random);

    num::mat<real> symmetric = num::dense(D.matrix());
    for (idx i = 0; i < D.size(); ++i) {
        for (idx j = 0; j < D.size(); ++j) {
            symmetric(i, j) *= diagonal[i] / diagonal[j];
        }
    }
    num::eigen_result eigen = num::eig_sym(num::assume_symmetric(symmetric));
    real exit_time =
        detail::sample_conditional_exit_time(eigen, *entrance_row, last_interior, random);

    array<State> interior_states;
    for (real time : observation_times) {
        if (time >= exit_time) {
            break;
        }
        num::vec<real> left = detail::semigroup_row(eigen, time, *entrance_row);
        num::vec<real> right = detail::semigroup_row(eigen, exit_time - time, last_interior);
        num::vec<real> bridge(D.size(), 0.0);
        for (idx i = 0; i < D.size(); ++i) {
            bridge[i] = std::max(0.0, left[i] * right[i]);
        }
        interior_states.push_back(
            D.state(num::sample_categorical(view<const real>(bridge), random)));
    }

    State first_exterior = sample_first_exterior_state(rates, D, last_interior, random);
    return {exit_time, last_interior, std::move(first_exterior), std::move(interior_states)};
}

// Sample the first exterior states of a walker population and merge repeated states.
template <typename State, detail::oracle<State> Rates, typename RNG>
[[nodiscard]] unordered_map<State, idx>
sample_first_exterior_counts(const Rates &rates, const subnetwork<State> &D,
                             view<const idx> last_interior, RNG &random) {
    if (last_interior.size() != D.size()) {
        throw std::invalid_argument(
            "last-interior-state counts must have one entry per subnetwork state");
    }
    unordered_map<State, idx> exterior;
    for (idx j = 0; j < D.size(); ++j) {
        for (idx draw = 0; draw < last_interior[j]; ++draw) {
            ++exterior[sample_first_exterior_state(rates, D, j, random)];
        }
    }
    return exterior;
}

// E[tau | X_0=i, X_{tau-}=j] = (Z^2)_ij / Z_ij
[[nodiscard]] inline real conditional_expected_sweep_duration(view<const real> Z_i,
                                                              view<const real> Z2_i, idx j) {
    if (Z_i.size() != Z2_i.size() || j >= Z_i.size()) {
        throw std::invalid_argument("the rows of Z and Z^2 must contain j");
    }
    return Z_i[j] > 0.0 ? Z2_i[j] / Z_i[j] : 0.0;
}

// u = Z^T rho
template <typename Factor>
[[nodiscard]] num::vec<real> occupation_times(const Factor &Z, const num::vec<real> &rho) {
    return num::solve(num::transpose(Z), rho);
}

// Apply one population sweep to D using the selected factorization regime.
template <typename State, detail::oracle<State> Rates, typename RNG>
[[nodiscard]] unordered_map<State, idx>
sample_boundary_counts(const Rates &rates, const subnetwork<State> &D, const num::vec<real> &rho,
                       idx walkers, regime solver, RNG &random) {
    if (rho.size() != D.size()) {
        throw std::invalid_argument("rho must have one entry per subnetwork state");
    }
    return with_regime<State, Rates>(solver, rates, [&](auto factorize) {
        auto Z = factorize(D, nullptr);
        auto u = occupation_times(Z, rho);
        auto last_interior_probability = last_interior_law(u, exit_rates(D));
        auto last_interior =
            sample_last_interior_counts(last_interior_probability, walkers, random);
        return sample_first_exterior_counts(rates, D, last_interior, random);
    });
}

// q = -R_{D,D^c}^T u, keyed by the first exterior state
template <typename State, listed_oracle<State> Rates>
[[nodiscard]] unordered_map<State, real>
boundary_law(const Rates &rates, const subnetwork<State> &D, view<const real> u) {
    if (u.size() != D.size()) {
        throw std::invalid_argument("u must have one entry per subnetwork state");
    }
    unordered_map<State, real> boundary;
    for (idx j = 0; j < D.size(); ++j) {
        for (auto &[destination, rate] : rates.transitions(D.state(j))) {
            if (rate < 0.0) {
                throw std::invalid_argument("transition rates must be nonnegative");
            }
            if (rate > 0.0 && !D.find(destination)) {
                boundary[std::move(destination)] += u[j] * rate;
            }
        }
    }
    return boundary;
}

} // namespace subsweep
