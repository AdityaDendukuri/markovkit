// the generator Q of a chain restricted to a finite state space
#pragma once

#include "linear/sparse/sparse.hpp"
#include "markovkit/chain.hpp"
#include <algorithm>
#include <cstdint>

namespace markovkit {

// what happens to a jump that leaves the state space: an absorbing generator keeps its
// rate on the diagonal, so lost probability measures the truncation error
enum class boundary : std::uint8_t { reflecting, absorbing };

// Q with columns as sources, dp/dt = Q p, stored as the jumps out of each state;
// a matrix-free linear operator for expv and Krylov solvers
class generator {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    struct jump {
        idx to;
        real rate;
    };

    template <state_space Space, markov_model Model>
    generator(const Space &space, const Model &model, boundary leaks = boundary::absorbing)
        : first_(space.size() + 1, 0), exit_(space.size(), 0.0), leak_(space.size(), 0.0) {
        for (idx j = 0; j < space.size(); ++j) {
            for (const auto &[y, rate] : model.transitions(space[j])) {
                if (auto i = space.find(y)) {
                    jumps_.push_back({*i, rate});
                    exit_[j] += rate;
                } else {
                    leak_[j] += rate;
                }
            }
            if (leaks == boundary::absorbing) {
                exit_[j] += leak_[j];
            }
            first_[j + 1] = jumps_.size();
        }
    }

    [[nodiscard]] idx rows() const { return exit_.size(); }
    [[nodiscard]] idx cols() const { return exit_.size(); }
    [[nodiscard]] idx n_jumps() const { return jumps_.size(); }

    // the jumps out of state j that stay in the space
    [[nodiscard]] view<const jump> jumps(idx j) const {
        return {jumps_.data() + first_[j], first_[j + 1] - first_[j]};
    }

    // -Q_jj for each state j
    [[nodiscard]] const array<real> &exit_rates() const { return exit_; }

    // the rate from each state to states outside the space
    [[nodiscard]] const array<real> &leak_rates() const { return leak_; }

    // y = Q x
    void apply(const vec<real> &x, vec<real> &y) const {
        std::fill(y.begin(), y.end(), 0.0);
        for (idx j = 0; j < rows(); ++j) {
            y[j] -= exit_[j] * x[j];
            for (const jump &e : jumps(j)) {
                y[e.to] += e.rate * x[j];
            }
        }
    }

    // y = shift x - Q x
    void apply_shifted(real shift, view<const real> x, view<real> y) const {
        std::fill(y.begin(), y.end(), 0.0);
        for (idx j = 0; j < rows(); ++j) {
            y[j] += (shift + exit_[j]) * x[j];
            for (const jump &e : jumps(j)) {
                y[e.to] -= e.rate * x[j];
            }
        }
    }

    // Q, assembled
    [[nodiscard]] num::spmat matrix() const { return assemble(0.0, 1.0); }

    // shift I - Q, assembled
    [[nodiscard]] num::spmat shifted_matrix(real shift) const { return assemble(shift, -1.0); }

  private:
    // sign * (Q - shift I)
    [[nodiscard]] num::spmat assemble(real shift, real sign) const {
        array<idx> row, col;
        array<real> value;
        for (idx j = 0; j < rows(); ++j) {
            row.push_back(j);
            col.push_back(j);
            value.push_back(-sign * (exit_[j] + shift));
            for (const jump &e : jumps(j)) {
                row.push_back(e.to);
                col.push_back(j);
                value.push_back(sign * e.rate);
            }
        }
        return num::spmat::from_triplets(rows(), cols(), row, col, value);
    }

    array<idx> first_;
    array<jump> jumps_;
    array<real> exit_;
    array<real> leak_;
};

// shift I - Q, the operator of a resolvent solve; it reads Q by reference
class shifted_generator {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    shifted_generator(const generator &Q, real shift) : Q_(&Q), shift_(shift) {}

    [[nodiscard]] idx rows() const { return Q_->rows(); }
    [[nodiscard]] idx cols() const { return Q_->cols(); }
    [[nodiscard]] real shift() const { return shift_; }

    void apply(const vec<real> &x, vec<real> &y) const { Q_->apply_shifted(shift_, x, y); }
    [[nodiscard]] num::spmat matrix() const { return Q_->shifted_matrix(shift_); }

  private:
    const generator *Q_;
    real shift_;
};

} // namespace markovkit
