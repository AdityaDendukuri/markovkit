#include "ssa/tau_leaping.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace ssa {

markovkit::Trajectory tau_leaping(const markovkit::ReactionSystem &model,
                                  const std::vector<double> &rates, const markovkit::State &initial,
                                  double t0, double tf, LeapOptions options, int seed) {
    std::mt19937 rng(seed);
    model.validate();
    markovkit::Trajectory path;
    path.times.push_back(t0);
    path.states.push_back(initial);

    markovkit::State current = initial;
    double t = t0;
    std::vector<double> prop(model.size(), 0.0);

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

} // namespace ssa
