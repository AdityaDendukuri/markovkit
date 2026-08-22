#pragma once

#include "markovkit/state.hpp"
#include <numerics.hpp>

namespace markovkit {

/// A transition leaving a restricted state set; source is a local matrix index.
struct BoundaryTransition {
    num::idx source{};
    State destination;
    double rate{};
};

} // namespace markovkit
