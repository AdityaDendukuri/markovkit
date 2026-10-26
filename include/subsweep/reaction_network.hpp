// a chemical reaction network, the rate oracle of the chemical master equation
#pragma once

#include "stochastic/categorical.hpp"
#include "subsweep/subnetwork.hpp"
#include <concepts>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

namespace subsweep {

// the level of a state, for a reaction network built without one
struct no_level {};

// one reaction: x becomes x + change at rate(x)
template <int Species>
struct reaction {
    state<Species> change;
    std::function<real(const state<Species> &)> rate;
};

// a reaction network, with an optional level that changes by at most one per reaction
template <int Species, typename Level = no_level>
struct reaction_network {
    array<reaction<Species>> reactions;
    Level level_of{};

    [[nodiscard]] std::tuple<real, array<std::pair<idx, real>>, array<std::pair<idx, real>>>
    add_to_subnetwork(const state<Species> &x, const subnetwork<state<Species>> &D) const {
        return std::tuple{total_rate(x), outgoing(x, D), incoming(x, D)};
    }

    [[nodiscard]] state<Species> sample_exterior(const state<Species> &x, idx,
                                                 const subnetwork<state<Species>> &D,
                                                 num::rng &random) const {
        array<state<Species>> boundary;
        array<real> weight;
        for (auto &[y, rate] : transitions(x)) {
            if (!D.find(y)) {
                boundary.push_back(std::move(y));
                weight.push_back(rate);
            }
        }
        if (boundary.empty()) {
            throw std::runtime_error("the state has no transition out of the subnetwork");
        }
        return boundary[num::sample_categorical(view<const real>(weight), random)];
    }

    [[nodiscard]] array<std::pair<state<Species>, real>>
    transitions(const state<Species> &x) const {
        array<std::pair<state<Species>, real>> result;
        for (const reaction<Species> &r : reactions) {
            if (auto y = shifted(x, r.change, 1)) {
                if (real rate = propensity(r, x); rate > 0.0) {
                    result.push_back({std::move(*y), rate});
                }
            }
        }
        return result;
    }

    [[nodiscard]] idx level(const state<Species> &x) const
        requires(!std::same_as<Level, no_level>) {
        return static_cast<idx>(level_of(x));
    }

  private:
    [[nodiscard]] real total_rate(const state<Species> &x) const {
        real total = 0.0;
        for (const reaction<Species> &r : reactions) {
            if (auto y = shifted(x, r.change, 1)) {
                if (real rate = propensity(r, x); rate > 0.0) {
                    total += rate;
                }
            }
        }
        return total;
    }

    [[nodiscard]] array<std::pair<idx, real>> outgoing(const state<Species> &x,
                                                       const subnetwork<state<Species>> &D) const {
        array<std::pair<idx, real>> rates;
        for (const reaction<Species> &r : reactions) {
            if (auto y = shifted(x, r.change, 1)) {
                if (const auto j = D.find(*y)) {
                    if (real rate = propensity(r, x); rate > 0.0) {
                        rates.push_back({*j, rate});
                    }
                }
            }
        }
        return rates;
    }

    [[nodiscard]] array<std::pair<idx, real>> incoming(const state<Species> &x,
                                                       const subnetwork<state<Species>> &D) const {
        array<std::pair<idx, real>> rates;
        for (const reaction<Species> &r : reactions) {
            if (auto w = shifted(x, r.change, -1)) {
                if (const auto j = D.find(*w)) {
                    if (real rate = propensity(r, *w); rate > 0.0) {
                        rates.push_back({*j, rate});
                    }
                }
            }
        }
        return rates;
    }

    // the rate of r at a state where it can fire
    [[nodiscard]] static real propensity(const reaction<Species> &r, const state<Species> &x) {
        real rate = r.rate(x);
        if (!(rate >= 0.0)) {
            throw std::invalid_argument("a reaction rate is negative or not a number");
        }
        return rate;
    }

    // x + sign * change, or nothing when a count would be negative
    [[nodiscard]] static std::optional<state<Species>>
    shifted(const state<Species> &x, const state<Species> &change, int sign) {
        state<Species> y = x;
        for (idx species = 0; species < y.size(); ++species) {
            y[species] += sign * change[species];
            if (y[species] < 0) {
                return std::nullopt;
            }
        }
        return y;
    }
};

template <int Species>
reaction_network(array<reaction<Species>>) -> reaction_network<Species>;

template <int Species, typename Level>
reaction_network(array<reaction<Species>>, Level) -> reaction_network<Species, Level>;

} // namespace subsweep
