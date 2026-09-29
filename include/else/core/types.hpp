// Shared value types for ELSE.
#pragma once

#include "container/multi_index.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include <cstddef>

namespace else_sim {

// Numerics' vocabulary: `idx`/`real` for scalars, `array`/`view`/`unordered_map`/
// `unordered_set` for storage, and `num::vec`/`num::mat`/`num::spmat` for the
// mathematics. Numeric data of fixed length is a `num::vec`; `array<real>`
// appears only where the length grows.
using num::array;
using num::idx;
using num::unordered_set;
using num::real;
using num::unordered_map;
using num::view;

// A transition from an interior state of the subnetwork to a state outside it.
template <typename State = num::multi_index>
struct boundary_transition {
    idx source = 0;      // interior state the transition leaves from
    State destination{}; // state outside the subnetwork
    real rate = 0.0;     // rate of this individual transition
};

// A sampled macrostep path: jump times and the state entered at each.
template <typename State = num::multi_index>
struct trajectory {
    array<real> times; // grows one entry per jump
    array<State> states;
};

// Transient probability over the represented states at one time.
template <typename State = num::multi_index>
struct density_solution {
    array<State> states;
    num::vec probability;
};

// Which rule the ensemble applies when the expanded set exceeds capacity.
enum class shedding_rule {
    expected_entries, // score = (-Rjj) uj - rhoj
    exact_cut_time,   // loss = uj qj / Zjj
    probed_cut_time,  // cut-time loss with a randomized estimate of Zjj
};

// How density IB reduces an outgoing distribution to the
// entrance weights used for the next subnetwork.
enum class density_resampling {
    weighted_support,      // distinct states without replacement; retain original masses
    multinomial_particles, // draws with replacement; aggregate empirical particle masses
};

// Options for the finite ensemble.
struct ensemble_options {
    idx capacity = 30;       // largest subnetwork to factor
    int expansion_depth = 1; // reactions to grow outward from the entrances
    real tolerance = 1e-6;   // propensity below which a state is not admitted
    idx maximum_steps = 100000;
    shedding_rule rule = shedding_rule::expected_entries;
    idx shedding_probes = 40;
    idx shedding_krylov_steps = 64;
    real shedding_krylov_tolerance = 1e-8;
    unsigned shedding_seed = 42;
    bool reuse_factorization = true;
    // Zero uses three changed slots for dense Woodbury reuse. Block factors
    // instead refactor the affected suffix and do not use this cutoff.
    idx maximum_reuse_slots = 0;
    density_resampling resampling = density_resampling::weighted_support;
    idx density_particles = 0; // zero uses `capacity`
};

// Work shared by one labeled-ensemble macrostep. These counters expose the
// decisions already made by the algorithm without changing its sampling law.
struct ensemble_step_diagnostics {
    idx active_trajectories = 0;
    idx distinct_entrances = 0;
    idx subnetwork_size = 0;
    idx changed_state_slots = 0;
    bool reuse_attempted = false;
    bool reuse_accepted = false;
    bool fresh_factorization = false;
    idx block_count = 0;
    idx reused_prefix_blocks = 0;
    idx reused_prefix_states = 0;
    idx reused_entrance_law_rows = 0;
    double selection_seconds = 0.0;
    double restriction_and_factor_seconds = 0.0;
    double factor_update_seconds = 0.0;
    double entrance_law_seconds = 0.0;
    double score_seconds = 0.0;
    double escape_setup_seconds = 0.0;
    double sampling_seconds = 0.0;
};

struct ensemble_diagnostics {
    array<ensemble_step_diagnostics> steps;
};

} // namespace else_sim
