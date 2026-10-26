// a continuous-time Markov chain, as the sweeps see it
#pragma once

#include "container/lattice_point.hpp"
#include "core/types.hpp"
#include "stochastic/rng.hpp"
#include <concepts>
#include <functional>
#include <tuple>
#include <utility>

namespace subsweep {

// the population of each species, a point of the integer lattice
template <int Species>
using state = num::lattice_point<Species>;

using num::array;
using num::idx;
using num::real;
using num::unordered_map;
using num::unordered_set;
using num::view;

template <typename State>
class subnetwork;

template <typename State>
concept markov_state = std::regular<State> && requires(const State &state) {
    {std::hash<State>{}(state)}->std::convertible_to<std::size_t>;
};

// add_to_subnetwork returns
//   (total_rate(x), outgoing(x, D), incoming(x, D)).
template <typename Rates, typename State>
concept rate_oracle =
    markov_state<State> &&
    requires(const Rates &rates, const State &x, const subnetwork<State> &D, num::rng &random) {
    {rates.add_to_subnetwork(x, D)}->std::
        convertible_to<std::tuple<real, array<std::pair<idx, real>>, array<std::pair<idx, real>>>>;
    {rates.sample_exterior(x, idx{}, D, random)}->std::convertible_to<State>;
};

template <typename Rates, typename State>
concept split_rate_oracle =
    markov_state<State> &&
    requires(const Rates &rates, const State &x, const subnetwork<State> &D, num::rng &random) {
    {rates.incoming(x, D)}->std::convertible_to<array<std::pair<idx, real>>>;
    {rates.outgoing(x, D)}->std::convertible_to<array<std::pair<idx, real>>>;
    {rates.total_rate(x)}->std::convertible_to<real>;
    {rates.sample_exterior(x, idx{}, D, random)}->std::convertible_to<State>;
};

namespace detail {

template <typename Rates, typename State>
concept oracle = rate_oracle<Rates, State> || split_rate_oracle<Rates, State>;

template <typename Rates, typename State>
[[nodiscard]] real total_rate(const Rates &rates, const State &x) {
    if constexpr (rate_oracle<Rates, State>) {
        return std::get<0>(rates.add_to_subnetwork(x, subnetwork<State>{}));
    } else {
        return rates.total_rate(x);
    }
}

template <typename Rates, typename State>
[[nodiscard]] array<std::pair<idx, real>> outgoing(const Rates &rates, const State &x,
                                                   const subnetwork<State> &D) {
    if constexpr (rate_oracle<Rates, State>) {
        return std::get<1>(rates.add_to_subnetwork(x, D));
    } else {
        return rates.outgoing(x, D);
    }
}

template <typename Rates, typename State>
[[nodiscard]] array<std::pair<idx, real>> incoming(const Rates &rates, const State &x,
                                                   const subnetwork<State> &D) {
    if constexpr (rate_oracle<Rates, State>) {
        return std::get<2>(rates.add_to_subnetwork(x, D));
    } else {
        return rates.incoming(x, D);
    }
}

} // namespace detail

// a rate oracle whose `level(x)` changes by at most one per transition
template <typename Rates, typename State>
concept leveled_oracle =
    detail::oracle<Rates, State> && requires(const Rates &rates, const State &x) {
    {rates.level(x)}->std::convertible_to<idx>;
};

// a leveled reversible chain with a positive stationary weight
template <typename Rates, typename State>
concept reversible_oracle =
    leveled_oracle<Rates, State> && requires(const Rates &rates, const State &x) {
    {rates.stationary_weight(x)}->std::convertible_to<real>;
};

// a rate oracle that lists every transition out of x, as (y, rate)
template <typename Rates, typename State>
concept listed_oracle =
    detail::oracle<Rates, State> && requires(const Rates &rates, const State &x) {
    {rates.transitions(x)}->std::convertible_to<array<std::pair<State, real>>>;
};

} // namespace subsweep
