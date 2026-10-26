// Settings and shared operations for repeated subnetwork sweeps.
#pragma once

#include "linear/factorization/factor.hpp"
#include "linear/factorization/reuse.hpp"
#include "subsweep/dense_inverse.hpp"
#include "subsweep/subnetwork.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace subsweep {

// the computational regime of Z = R^{-1} (Section 4 of the paper)
enum class regime {
    // block when the chain has levels, sparse otherwise
    automatic,
    // a dense inverse, initialized by Cholesky for reversible chains and LU otherwise
    dense,
    // block LU in the chain's levels, refactored from the first changed level
    block,
    // block Cholesky after the oracle's diagonal similarity, with the same suffix update
    block_cholesky,
    // sparse LU, refactored every sweep
    sparse,
};

// everything a sweep needs besides the chain
struct sweep_settings {
    // the largest number of states in one subnetwork
    idx capacity = 30;
    // reaction layers grown around the entrances, for a chain that lists its transitions
    int depth = 0;
    // a state with total rate at most this is not admitted
    real negligible_rate = 0.0;
    // break ties in shedding by the chain's levels, highest level first
    bool shed_by_level = false;
    subsweep::regime regime = subsweep::regime::automatic;
    unsigned seed = 42;

    // The most sweeps of one walker.
    idx sweep_limit = std::numeric_limits<idx>::max();
    // Keep each sweep's R, entrance index, and last interior index for reconstruction.
    bool keep_subnetworks = false;

    // Walkers used for a population boundary-law sample, or the capacity when zero.
    idx walkers = 0;
    // Stop a population calculation below this entrance mass.
    real negligible_mass = 0.0;

    [[nodiscard]] idx walker_count() const { return walkers == 0 ? capacity : walkers; }
};

// the representation of Z = R^{-1} that `compute_Z` returns
template <typename ComputeZ, typename State>
using factor_of = std::invoke_result_t<ComputeZ &, const subnetwork<State> &, std::nullptr_t>;

template <typename Factor, typename State, typename ComputeZ>
void update_factor(std::optional<Factor> &Z, const subnetwork<State> &D, ComputeZ &compute_Z) {
    const Factor *previous = Z ? &*Z : nullptr;
    Z = compute_Z(D, previous);
}

// run(compute_Z) in the chosen regime
template <typename State, typename Rates, typename Run>
decltype(auto) with_regime(regime chosen, const Rates &rates, Run &&run) {
    if (chosen == regime::automatic) {
        chosen = leveled_oracle<Rates, State> ? regime::block : regime::sparse;
    }
    switch (chosen) {
    case regime::dense:
        if constexpr (requires(const Rates &r, const State &x) {
                          {r.stationary_weight(x)}->std::convertible_to<real>;
                      }) {
            return run([&rates](const subnetwork<State> &D, auto previous) {
                num::vec<real> h(D.size());
                for (idx i = 0; i < D.size(); ++i) {
                    h[i] = std::sqrt(rates.stationary_weight(D.state(i)));
                }
                return dense_inverse(D.dense_matrix(), h, previous, D.changed());
            });
        } else {
            return run([](const subnetwork<State> &D, auto previous) {
                return dense_inverse(D.dense_matrix(), previous, D.changed());
            });
        }
    case regime::block:
        if constexpr (leveled_oracle<Rates, State>) {
            return run([](const subnetwork<State> &D, auto previous) {
                return num::lu(D.matrix(), num::blocks(D.levels()), previous, D.changed());
            });
        } else {
            throw std::invalid_argument("the block regime needs a chain with levels");
        }
    case regime::block_cholesky:
        if constexpr (reversible_oracle<Rates, State>) {
            return run([&rates](const subnetwork<State> &D, auto previous) {
                num::vec<real> h(D.size());
                for (idx i = 0; i < D.size(); ++i) {
                    h[i] = std::sqrt(rates.stationary_weight(D.state(i)));
                }
                return num::cholesky(D.matrix(), num::blocks(D.levels()), h, previous, D.changed());
            });
        } else {
            throw std::invalid_argument(
                "the block-Cholesky regime needs level(x) and stationary_weight(x)");
        }
    default:
        return run(
            [](const subnetwork<State> &D, auto) { return num::lu(D.matrix(), num::sparse); });
    }
}

} // namespace subsweep
