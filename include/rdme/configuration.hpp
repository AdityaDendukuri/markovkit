#pragma once

#include "markovkit/state.hpp"

namespace rdme {

/// Voxel-major molecule counts: coordinate = voxel * species_count + species.
using configuration = markovkit::state;

} // namespace rdme
