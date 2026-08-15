/// @file boundary.hpp
/// @brief Boundary conditions for FSP state-space truncation.
#pragma once

#include "markovkit/fsp/multi_index.hpp"
#include <functional>
#include <vector>

namespace fsp {

/// @brief Boundary condition: returns true if state x is inside the domain.
using BoundaryCondition = std::function<bool(const MultiIndex&)>;

/// @brief Rectangular boundary: each component of x must lie in [0, bound].
///
/// Equivalent to RectLatticeBoundaryCondition(x, bound) in Julia.
inline BoundaryCondition rect_boundary(int bound) {
    return [bound](const MultiIndex& x) -> bool {
        for (int c : x) {
            if (c < 0 || c > bound) return false;
        }
        return true;
    };
}

/// @brief Rectangular boundary with per-species upper limits.
///
/// bounds[i] = max allowed value for species i.
inline BoundaryCondition rect_boundary(const std::vector<int>& bounds) {
    return [bounds](const MultiIndex& x) -> bool {
        if (x.size() != bounds.size()) return false;
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (x[i] < 0 || x[i] > bounds[i]) return false;
        }
        return true;
    };
}

} // namespace fsp
