/// @file else/subnetwork.hpp
/// @brief Exact Linear Subnetwork Elimination (ELSE) subnetwork & Schur shedding.
#pragma once

#include <numerics.hpp>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace markovkit::else_solver {

/// @brief Subnetwork state representation and fundamental matrix Z = -R^{-1}.
struct Subnetwork {
    num::Matrix R; ///< Killed subnetwork generator matrix R (negative diagonal)
    num::Matrix Z; ///< Fundamental matrix Z = -R^{-1}

    explicit Subnetwork(num::Matrix generator);
    void shed_state(num::idx k);
};

} // namespace markovkit::else_solver
