#pragma once
#include "core/types.hpp"
#include "markovkit/state.hpp"
#include <algorithm>
#include <functional>

namespace cme {

// Returns true if state x is within domain
using boundary_condition = std::function<bool(const markovkit::state &)>;

inline boundary_condition nonnegative_boundary() {
    return [](const markovkit::state &state) {
        return std::all_of(state.begin(), state.end(), [](int value) { return value >= 0; });
    };
}

// [0, bound] on all coordinates
inline boundary_condition rect_boundary(int bound) {
    return [bound](const markovkit::state &x) -> bool {
        for (int c : x) {
            if (c < 0 || c > bound) {
                return false;
            }
        }
        return true;
    };
}

// [0, bounds[i]] per coordinate
inline boundary_condition rect_boundary(const num::array<int> &bounds) {
    return [bounds](const markovkit::state &x) -> bool {
        if (x.size() != bounds.size()) {
            return false;
        }
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (x[i] < 0 || x[i] > bounds[i]) {
                return false;
            }
        }
        return true;
    };
}

} // namespace cme
