#include "cme/generator.hpp"
#include "core/types.hpp"

namespace cme {

generator_result build_generator(const state_space &sp, const reaction_system &model,
                                 const num::array<double> &rates, double t, bool absorbing) {
    int n = sp.size();
    model.validate();

    num::array<num::idx> rows, cols;
    num::array<num::real> vals;
    rows.reserve(static_cast<std::size_t>(n) * (model.size() + 1));
    cols.reserve(rows.capacity());
    vals.reserve(rows.capacity());

    num::vec<double> exit_rate_boundary(static_cast<num::idx>(n), 0.0);
    num::vec<double> diag(static_cast<num::idx>(n), 0.0);

    // Assemble column-oriented jumps and record exits.
    for (int j = 0; j < n; ++j) {
        const markovkit::state &x = sp.states[j];
        double col_sum = 0.0;

        // Each enabled reaction contributes one jump from this source state.
        for (std::size_t k = 0; k < model.size(); ++k) {
            double alpha = model.propensities[k](x, rates, t);
            if (alpha <= 0.0) {
                continue;
            }

            markovkit::state y = x;
            for (num::idx species = 0; species < y.size(); ++species)
                y[species] += model.changes[k][species];
            auto it = sp.index.find(y);
            if (it != sp.index.end()) {
                int i = it->second;
                rows.push_back(static_cast<num::idx>(i));
                cols.push_back(static_cast<num::idx>(j));
                vals.push_back(alpha);
                col_sum += alpha;
            } else {
                exit_rate_boundary[j] += alpha;
                // Absorbing FSP keeps outward rates in the diagonal, so mass can leave.
                if (absorbing) {
                    col_sum += alpha;
                }
            }
        }

        diag[j] = col_sum;
        if (col_sum > 0.0) {
            rows.push_back(static_cast<num::idx>(j));
            cols.push_back(static_cast<num::idx>(j));
            vals.push_back(-col_sum);
        }
    }

    num::spmat A = num::spmat::from_triplets(static_cast<num::idx>(n), static_cast<num::idx>(n),
                                             rows, cols, vals);

    return {std::move(A), std::move(exit_rate_boundary), std::move(diag)};
}

} // namespace cme
