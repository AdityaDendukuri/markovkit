#pragma once

#include "markovkit/state.hpp"
#include <numerics.hpp>

namespace markovkit {

/// A transition leaving a restricted state set; source is a local matrix index.
struct boundary_transition {
    num::idx source{};
    state destination;
    double rate{};
};

} // namespace markovkit
