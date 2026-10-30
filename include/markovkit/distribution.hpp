// a probability distribution on a growable finite set of states
#pragma once

#include "markovkit/chain.hpp"
#include "stats/probability.hpp"
#include <optional>

namespace markovkit {

// states, their probabilities, and a lookup from state to index; a state_space
class distribution {
  public:
    distribution() = default;

    // the point mass at x
    explicit distribution(const state &x) { add(x, 1.0); }

    [[nodiscard]] idx size() const { return states_.size(); }
    [[nodiscard]] const state &operator[](idx i) const { return states_[i]; }
    [[nodiscard]] const array<state> &states() const { return states_; }
    [[nodiscard]] array<real> &probabilities() { return p_; }
    [[nodiscard]] const array<real> &probabilities() const { return p_; }

    [[nodiscard]] std::optional<idx> find(const state &x) const {
        auto found = index_.find(x);
        return found == index_.end() ? std::nullopt : std::optional<idx>(found->second);
    }

    // add x with probability `mass`, unless it is already present
    void add(const state &x, real mass = 0.0) {
        if (index_.emplace(x, states_.size()).second) {
            states_.push_back(x);
            p_.push_back(mass);
        }
    }

    // remove every state i with remove(i), keeping the rest in order
    template <typename Remove>
    void remove_if(Remove remove) {
        array<char> drop(size());
        for (idx i = 0; i < size(); ++i) {
            drop[i] = remove(i) ? 1 : 0;
        }
        idx kept = 0;
        index_.clear();
        for (idx i = 0; i < size(); ++i) {
            if (drop[i]) {
                continue;
            }
            if (kept != i) {
                states_[kept] = std::move(states_[i]);
                p_[kept] = p_[i];
            }
            index_[states_[kept]] = kept;
            ++kept;
        }
        states_.resize(kept);
        p_.resize(kept);
    }

    // clip negative roundoff and rescale to total probability one; returns the old total
    real normalize() { return num::clip_and_normalize_nonnegative(p_); }

    // the mean count of each species
    [[nodiscard]] vec<real> means() const {
        vec<real> result(states_.empty() ? 0 : states_.front().size(), 0.0);
        for (idx i = 0; i < size(); ++i) {
            for (idx s = 0; s < result.size(); ++s) {
                result[s] += p_[i] * states_[i][s];
            }
        }
        return result;
    }

    // the distribution of one species' count
    [[nodiscard]] num::map<int, real> marginal(idx species) const {
        num::map<int, real> result;
        for (idx i = 0; i < size(); ++i) {
            result[states_[i][species]] += p_[i];
        }
        return result;
    }

  private:
    array<state> states_;
    array<real> p_;
    num::unordered_map<state, idx, state_hash> index_;
};

} // namespace markovkit
