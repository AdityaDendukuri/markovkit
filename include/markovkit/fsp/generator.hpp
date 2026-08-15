/// @file generator.hpp
/// @brief CME generator matrix construction.
#pragma once

#include "markovkit/fsp/state_space.hpp"
#include "markovkit/fsp/reaction_system.hpp"
#include "linalg/sparse/sparse.hpp"
#include <vector>

namespace fsp {

/// @brief Result of build_generator.
struct GeneratorResult {
    num::SparseMatrix A;                    ///< full generator (in_flow + out_flow)
    std::vector<double> exit_rate_boundary; ///< per-state outflux to states outside domain
    std::vector<double> diag;              ///< per-state within-domain outflux = -A(i,i)
};

/// @brief Build the CME generator restricted to sp.
inline GeneratorResult build_generator(const StateSpace& sp,
                                        const ReactionSystem& model,
                                        const std::vector<double>& rates,
                                        double t,
                                        bool absorbing = false) {
    int n = sp.size();
    std::vector<num::idx> rows, cols;
    std::vector<double> vals;

    std::vector<double> exit_rate_boundary(n, 0.0);
    std::vector<double> diag(n, 0.0);

    for (int i = 0; i < n; ++i) {
        const auto& s = sp.states[i];
        double total_exit = 0.0;

        for (std::size_t r = 0; r < model.propensities.size(); ++r) {
            double rate = model.propensities[r](s, rates, t);
            if (rate <= 0.0) continue;
            total_exit += rate;

            MultiIndex nxt = s;
            for (std::size_t c = 0; c < s.size(); ++c) nxt[c] += model.stoich_vecs[r][c];

            auto it = sp.index.find(nxt);
            if (it != sp.index.end()) {
                int j = it->second;
                rows.push_back((num::idx)j);
                cols.push_back((num::idx)i);
                vals.push_back(rate);
                diag[i] += rate;
            } else {
                exit_rate_boundary[i] += rate;
            }
        }

        double diag_val = absorbing ? -total_exit : -diag[i];
        rows.push_back((num::idx)i);
        cols.push_back((num::idx)i);
        vals.push_back(diag_val);
    }

    num::SparseMatrix A = num::SparseMatrix::from_triplets((num::idx)n, (num::idx)n, rows, cols, vals);
    return {std::move(A), std::move(exit_rate_boundary), std::move(diag)};
}

} // namespace fsp
