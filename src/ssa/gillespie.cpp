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

  while (t < tf && path.reactions.size() < maximum_steps) {
    double total = 0.0;
    for (std::size_t r = 0; r < model.size(); ++r) {
      prop[r] = std::max(0.0, model.propensities[r](current, rates, t));
      total += prop[r];
    }
    if (!(total > 0.0)) {
      break;
    }

    // Independent draws choose the exponential waiting time and reaction channel.
    double dt = -std::log(std::max(uniform(rng), 1e-300)) / total;
    if (t + dt > tf) {
      break;
    }
    t += dt;

    double target = uniform(rng) * total;
    double csum = 0.0;
    std::size_t fired = model.size() - 1;
    for (std::size_t r = 0; r < model.size(); ++r) {
      csum += prop[r];
      if (target <= csum) {
        fired = r;
        break;
      }
    }

    current = current + model.changes[fired];
    path.times.push_back(t);
    path.states.push_back(current);
    path.reactions.push_back(fired);
  }

  if (path.reactions.size() < maximum_steps) {
    path.times.push_back(tf);
    path.states.push_back(current);
  }
  return path;
}

} // namespace ssa
