#pragma once

#include "ssa/gillespie.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace ssa {

struct leap_options {
    double fixed_step = 0.0; // Positive values disable adaptive step selection.
    double epsilon = 0.03;   // Expected total events per adaptive step.
    double min_step = 1e-6;
};

/// Poisson tau-leaping templated over PRNG.
template <typename rng_state = num::rng>
inline markovkit::trajectory tau_leaping(const markovkit::reaction_system &model,
                                         const num::array<double> &rates,
                                         const markovkit::state &initial, double t0, double tf,
                                         leap_options options, rng_state &rng) {
    model.validate();
    markovkit::trajectory path;
    path.times.push_back(t0);
    path.states.push_back(initial);

    markovkit::state current = initial;
    double t = t0;
    num::vec<double> prop(model.size(), 0.0);

    // Recompute propensities at the start of every leap.
    while (t < tf) {
        const double total = model.evaluate_propensities(current, rates, t, prop);
        if (!(total > 0.0)) {
            break;
        }

        double dt = options.fixed_step > 0.0 ? options.fixed_step
                                             : std::max(options.min_step, options.epsilon / total);
        // Shorten the last leap to end exactly at tf.
        dt = std::min(dt, tf - t);
        t += dt;

        // Freeze propensities over the step and draw each reaction count independently.
        for (std::size_t r = 0; r < model.size(); ++r) {
            if (prop[r] <= 0.0) {
                continue;
            }
            std::poisson_distribution<int> poisson(prop[r] * dt);
            int count = poisson(rng);
            if (count <= 0) {
                continue;
            }

            // Apply all firings while preventing negative counts.
            for (std::size_t c = 0; c < current.size(); ++c) {
                current[c] = std::max(0, current[c] + (count * model.changes[r][c]));
            }
        }

        path.times.push_back(t);
        path.states.push_back(current);
    }
    return path;
}

/// Seed-based overload for backward compatibility.
inline markovkit::trajectory tau_leaping(const markovkit::reaction_system &model,
                                         const num::array<double> &rates,
                                         const markovkit::state &initial, double t0, double tf,
                                         leap_options options = {}, int seed = 42) {
    num::rng rng(static_cast<unsigned int>(seed));
    return tau_leaping(model, rates, initial, t0, tf, options, rng);
}

} // namespace ssa
