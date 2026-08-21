#include "ssa/gillespie.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace ssa {

markovkit::Trajectory gillespie(const markovkit::ReactionSystem& model,
                                const std::vector<double>& rates,
                                const markovkit::State& initial,
                                double t0,
                                double tf,
                                int seed,
                                std::size_t maximum_steps) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  model.validate();
  markovkit::Trajectory path;
  path.times.push_back(t0);
  path.states.push_back(initial);

  markovkit::State current = initial;
  double t = t0;
  std::vector<double> prop(model.size(), 0.0);

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
    current = current + model.changes[fired];
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

} // namespace ssa
