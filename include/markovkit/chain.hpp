// a continuous-time Markov chain, as the solvers see it
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include <concepts>
#include <cstddef>
#include <optional>

namespace markovkit {

using num::array;
using num::idx;
using num::real;
using num::vec;
using num::view;

// molecule counts, one per coordinate; an RDME state is voxel-major (voxel * species + s)
using state = array<int>;

struct state_hash {
    std::size_t operator()(const state &x) const noexcept {
        std::size_t seed = x.size();
        for (int count : x) {
            seed ^= static_cast<std::size_t>(count) + 0x9e3779b97f4a7c15ULL + (seed << 6U) +
                    (seed >> 2U);
        }
        return seed;
    }
};

// one jump out of a state
struct transition {
    state to;
    real rate;
};

// a model lists every jump out of a state with a positive rate
template <typename Model>
concept markov_model = requires(const Model &model, const state &x) {
    {model.transitions(x)}->std::convertible_to<array<transition>>;
};

// a finite state space numbers its states 0, ..., size() - 1
template <typename Space>
concept state_space = requires(const Space &space, const state &x, idx i) {
    {space.size()}->std::convertible_to<idx>;
    {space[i]}->std::convertible_to<const state &>;
    {space.find(x)}->std::convertible_to<std::optional<idx>>;
};

} // namespace markovkit
