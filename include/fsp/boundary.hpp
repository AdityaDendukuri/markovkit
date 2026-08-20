#pragma once
#include "fsp/multi_index.hpp"
#include <functional>
#include <vector>

namespace fsp {

// Returns true if state x is within domain
using BoundaryCondition = std::function<bool(const MultiIndex&)>;

// [0, bound] on all coordinates
inline BoundaryCondition rect_boundary(int bound) {
  return [bound](const MultiIndex& x) -> bool {
    for (int c : x) {
      if (c < 0 || c > bound) { return false;
}
    }
    return true;
  };
}

// [0, bounds[i]] per coordinate
inline BoundaryCondition rect_boundary(const std::vector<int>& bounds) {
  return [bounds](const MultiIndex& x) -> bool {
    if (x.size() != bounds.size()) { return false;
}
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (x[i] < 0 || x[i] > bounds[i]) { return false;
}
    }
    return true;
  };
}

} // namespace fsp
