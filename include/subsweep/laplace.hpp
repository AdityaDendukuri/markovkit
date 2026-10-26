// Laplace-domain composition across a recorded sequence of finite subnetworks
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/solvers/auto_resolvent.hpp"
#include "linear/sparse/sparse.hpp"
#include "quadrature/talbot.hpp"
#include <algorithm>
#include <stdexcept>

namespace subsweep {

using num::real;

// a probability on the listed states
template <typename State>
struct distribution {
    num::array<State> states;
    num::vec<real> probability;
};

// the nonnegative part of a reconstructed density, normalized when asked
inline void clip_density(num::vec<real> &probability, bool normalize) {
    num::real mass = 0.0;
    for (num::real &value : probability) {
        value = std::max(num::real{0.0}, value);
        mass += value;
    }
    if (normalize && mass > 0.0) {
        for (num::real &value : probability) {
            value /= mass;
        }
    }
}

// R of one subnetwork, and the transfer T to the next, empty for the last
struct sweep_transform {
    num::spmat R;
    num::spmat T = num::spmat::from_triplets(0, 0, {}, {}, {});
};

// the local densities of each subnetwork at the contour `nodes`
[[nodiscard]] inline num::array<num::vec<real>>
inverse_laplace(num::view<const sweep_transform> transforms, num::idx initial_row,
                num::view<const num::contour_node> nodes) {
    if (transforms.empty() || nodes.empty() || initial_row >= transforms.front().R.n_rows()) {
        throw std::invalid_argument(
            "Laplace reconstruction needs transforms, contour nodes, and an initial row");
    }
    // the resolvent of -R_k^T for each subnetwork, factored again at every node
    num::array<num::auto_resolvent> resolvent;
    num::array<num::vec<real>> result;
    for (num::idx k = 0; k < transforms.size(); ++k) {
        const auto &[R, T] = transforms[k];
        if (R.n_rows() != R.n_cols()) {
            throw std::invalid_argument("each R must be square");
        }
        if (k + 1 < transforms.size() &&
            (T.n_rows() != transforms[k + 1].R.n_rows() || T.n_cols() != R.n_rows())) {
            throw std::invalid_argument("T must map one sweep to the next");
        }
        resolvent.emplace_back(num::scaled(num::transpose(R), -1.0));
        result.emplace_back(R.n_rows(), 0.0);
    }

    // at each node s: x^(k) = (sI + R_k^T)^{-1} a^(k), and a^(k+1) = T_k x^(k)
    for (const auto &[s, weight] : nodes) {
        num::vec<num::cplx> a(transforms.front().R.n_rows(), 0.0);
        a[initial_row] = 1.0;
        for (num::idx k = 0; k < transforms.size(); ++k) {
            num::vec<num::cplx> x;
            num::solve(num::shift(resolvent[k], s), a, x);
            for (num::idx i = 0; i < x.size(); ++i) {
                result[k][i] += (weight * x[i]).real();
            }
            if (k + 1 == transforms.size()) {
                break;
            }
            const num::spmat &T = transforms[k].T;
            a = num::vec<num::cplx>(T.n_rows(), 0.0);
            for (num::idx i = 0; i < T.n_rows(); ++i) {
                for (num::idx entry = T.row_ptr()[i]; entry < T.row_ptr()[i + 1]; ++entry) {
                    a[i] += T.values()[entry] * x[T.col_idx()[entry]];
                }
            }
        }
    }
    return result;
}

} // namespace subsweep
