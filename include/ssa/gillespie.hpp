#pragma once
#include "stochastic/categorical.hpp"

#include "core/types.hpp"
#include "markovkit/reaction_system.hpp"
#include "markovkit/trajectory.hpp"
#include "stochastic/rng.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace ssa {

/// Gillespie's exact direct stochastic simulation method templated over PRNG.
template <typename rng_state = num::rng>
inline markovkit::trajectory
gillespie(const markovkit::reaction_system &model, const num::array<double> &rates,
          const markovkit::state &initial, double t0, double tf, rng_state &rng,
          std::size_t maximum_steps = std::numeric_limits<std::size_t>::max()) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    model.validate();
    markovkit::trajectory path;
    path.times.push_back(t0);
    path.states.push_back(initial);

    markovkit::state current = initial;
    double t = t0;
    num::vec<double> prop(model.size(), 0.0);

    // Recompute propensities after every reaction event.
    while (t < tf && path.reactions.size() < maximum_steps) {
        const double total = model.evaluate_propensities(current, rates, t, prop);
        if (!(total > 0.0)) {
            break;
        }

        // Independent draws choose the exponential waiting time and reaction channel.
        double dt = -std::log(std::max(uniform(rng), 1e-300)) / total;
        if (t + dt > tf) {
            break;
        }
        t += dt;

        const std::size_t fired = num::sample_categorical(prop, rng);

        // Apply the selected stoichiometric change.
        for (num::idx species = 0; species < current.size(); ++species)
            current[species] += model.changes[fired][species];
        path.times.push_back(t);
        path.states.push_back(current);
        path.reactions.push_back(fired);
    }

    if (path.reactions.size() < maximum_steps) {
        // Extend an absorbed path to the requested horizon.
        path.times.push_back(tf);
        path.states.push_back(current);
    }
    return path;
}

/// Seed-based overload for backward compatibility.
inline markovkit::trajectory
gillespie(const markovkit::reaction_system &model, const num::array<double> &rates,
          const markovkit::state &initial, double t0, double tf, int seed = 42,
          std::size_t maximum_steps = std::numeric_limits<std::size_t>::max()) {
    num::rng rng(static_cast<unsigned int>(seed));
    return gillespie(model, rates, initial, t0, tf, rng, maximum_steps);
}

} // namespace ssa
