#include "else/trajectory.hpp"
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace else_sim {
namespace {

using StateWeights = std::map<markovkit::State, double>;

void expand_from_entrances(cme::StateSpace& workspace,
                           const StateWeights& entrances,
                           const markovkit::ReactionSystem& model,
                           const std::vector<double>& rates,
                           const ELSEOptions& options) {
  // Seed the expansion frontier with the current entrance distribution.
  std::vector<markovkit::State> frontier;
  std::set<markovkit::State> visited;
  for (const auto& [state, weight] : entrances) {
    workspace.add_state(state, weight);
    frontier.push_back(state);
    visited.insert(state);
  }

  const auto boundary = cme::nonnegative_boundary();
  for (int depth = 0; depth < options.expansion_depth; ++depth) {
    std::vector<markovkit::State> next;
    for (const auto& state : frontier) {
      for (const auto& change : model.changes) {
        const auto destination = state + change;
        if (!visited.insert(destination).second || !boundary(destination)) {
          continue;
        }
        double total_rate = 0.0;
        for (const auto& propensity : model.propensities) {
          total_rate += std::max(0.0, propensity(destination, rates, 0.0));
        }
        // Ignore destinations with no meaningful outgoing activity.
        if (total_rate <= options.tolerance) {
          continue;
        }
        workspace.add_state(destination);
        next.push_back(destination);
      }
    }
    frontier = std::move(next);
  }
}

void validate(const ELSEOptions& options) {
  if (options.capacity == 0 || options.expansion_depth < 0 || !(options.tolerance > 0.0) ||
      options.maximum_steps == 0) {
    throw std::invalid_argument("invalid ELSE options");
  }
}

struct ExitChoice {
  std::vector<markovkit::State> destinations;
  std::vector<double> rates;
};

void prune(cme::StateSpace& states,
           const Subnetwork& subnetwork,
           const num::Vector& entrance,
           const StateWeights& protected_states,
           num::idx capacity) {
  if (states.size() <= static_cast<int>(capacity)) {
    return;
  }
  if (protected_states.size() >= capacity) {
    throw std::invalid_argument("ELSE capacity must exceed the number of entrance states");
  }

  const auto occupation = subnetwork.occupation(entrance);
  const auto visits = subnetwork.expected_entries(occupation, entrance);
  // Entrance states are protected; rank all others by expected visits.
  std::vector<int> candidate_states;
  std::vector<double> candidate_visits;
  for (int index = 0; index < states.size(); ++index) {
    if (!protected_states.contains(states.states[index])) {
      candidate_states.push_back(index);
      candidate_visits.push_back(visits[static_cast<num::idx>(index)]);
    }
  }

  const auto remove_count = static_cast<std::size_t>(states.size() - static_cast<int>(capacity));
  std::vector<bool> remove(static_cast<std::size_t>(states.size()), false);
  // Remove the least-visited unprotected states first.
  for (num::idx candidate :
       num::smallest_indices(std::span<const double>(candidate_visits), remove_count)) {
    remove[static_cast<std::size_t>(candidate_states[candidate])] = true;
  }
  states.remove_states(remove);
}

Subnetwork prepare_subnetwork(cme::StateSpace& workspace,
                              const StateWeights& entrances,
                              const markovkit::ReactionSystem& model,
                              const std::vector<double>& rates,
                              const ELSEOptions& options) {
  // Grow reaction shells around protected entrances.
  expand_from_entrances(workspace, entrances, model, rates, options);

  if (workspace.size() > static_cast<int>(options.capacity)) {
    Subnetwork expanded(cme::restrict_cme(workspace, model, rates));
    num::Vector entrance(workspace.size(), 0.0);
    for (const auto& [state, weight] : entrances) {
      entrance[static_cast<num::idx>(workspace.find(state))] = weight;
    }
    prune(workspace, expanded, entrance, entrances, options.capacity);
  }
  return Subnetwork(cme::restrict_cme(workspace, model, rates));
}

void finish(markovkit::Trajectory& trajectory, double& current_time, double final_time) {
  current_time = final_time;
  if (trajectory.times.back() < final_time) {
    trajectory.times.push_back(final_time);
    trajectory.states.push_back(trajectory.states.back());
  }
}

std::vector<ExitChoice> boundary_choices(const Subnetwork& subnetwork) {
  std::vector<ExitChoice> choices(subnetwork.size());
  for (const auto& transition : subnetwork.boundary()) {
    auto& choice = choices[transition.source];
    choice.destinations.push_back(transition.destination);
    choice.rates.push_back(transition.rate);
  }
  return choices;
}

} // namespace

std::vector<markovkit::Trajectory> else_ensemble(const markovkit::ReactionSystem& model,
                                                 const std::vector<double>& rates,
                                                 const markovkit::State& initial,
                                                 std::size_t count,
                                                 double initial_time,
                                                 double final_time,
                                                 ELSEOptions options,
                                                 int seed) {
  model.validate();
  validate(options);
  if (count == 0) {
    throw std::invalid_argument("ELSE trajectory count must be positive");
  }
  if (initial_time > final_time) {
    throw std::invalid_argument("ELSE time interval is invalid");
  }

  std::vector<markovkit::Trajectory> trajectories(count);
  std::vector<double> times(count, initial_time);
  std::vector<markovkit::State> states(count, initial);
  std::vector<std::size_t> steps(count, 0);
  std::vector<std::mt19937> generators;
  generators.reserve(count);
  // Give each trajectory a reproducible independent stream.
  for (std::size_t trajectory = 0; trajectory < count; ++trajectory) {
    trajectories[trajectory].times.push_back(initial_time);
    trajectories[trajectory].states.push_back(initial);
    generators.emplace_back(seed + static_cast<int>(trajectory));
  }

  cme::StateSpace workspace;
  while (true) {
    // Group unfinished trajectories by their current entrance state.
    std::vector<std::size_t> unfinished;
    StateWeights entrances;
    for (std::size_t trajectory = 0; trajectory < count; ++trajectory) {
      if (times[trajectory] < final_time && steps[trajectory] < options.maximum_steps) {
        unfinished.push_back(trajectory);
        entrances[states[trajectory]] += 1.0;
      }
    }
    if (unfinished.empty()) {
      break;
    }
    for (auto& [state, weight] : entrances) {
      // Entrance weights describe the current ensemble distribution.
      weight /= static_cast<double>(unfinished.size());
    }

    auto subnetwork = prepare_subnetwork(workspace, entrances, model, rates, options);
    if (subnetwork.boundary_states().empty()) {
      for (std::size_t trajectory : unfinished) {
        finish(trajectories[trajectory], times[trajectory], final_time);
      }
      break;
    }

    // One solve column serves every trajectory currently at the same entrance state.
    std::vector<markovkit::State> entrance_states;
    std::unordered_map<markovkit::State, num::idx> entrance_column;
    for (const auto& [state, weight] : entrances) {
      entrance_column[state] = entrance_states.size();
      entrance_states.push_back(state);
    }

    num::Matrix right_hand_sides(subnetwork.size(), entrance_states.size(), 0.0);
    for (num::idx column = 0; column < entrance_states.size(); ++column) {
      right_hand_sides(static_cast<num::idx>(subnetwork.find(entrance_states[column])), column) =
          1.0;
    }
    const auto integrals = subnetwork.occupation_integrals(right_hand_sides);

    std::vector<num::CategoricalSampler> exit_distributions;
    exit_distributions.reserve(entrance_states.size());
    // Cache one exit distribution for each distinct entrance.
    for (num::idx column = 0; column < entrance_states.size(); ++column) {
      const auto weights = subnetwork.exit_probabilities(integrals, column);
      exit_distributions.emplace_back(weights);
    }
    auto destinations = boundary_choices(subnetwork);

    // Advance each trajectory with its sampled exit state and waiting time.
    for (std::size_t trajectory : unfinished) {
      const num::idx column = entrance_column[states[trajectory]];
      auto& rng = generators[trajectory];
      const auto boundary_position = exit_distributions[column](rng);
      const auto boundary_state = subnetwork.boundary_states()[boundary_position];
      const double waiting_time =
          subnetwork.conditional_exit_time(integrals, boundary_state, column);
      auto& choice = destinations[boundary_state];

      if (!(waiting_time > 0.0) || choice.destinations.empty() ||
          times[trajectory] + waiting_time >= final_time) {
        finish(trajectories[trajectory], times[trajectory], final_time);
        continue;
      }

      times[trajectory] += waiting_time;
      states[trajectory] = choice.destinations[num::sample_categorical(choice.rates, rng)];
      trajectories[trajectory].times.push_back(times[trajectory]);
      trajectories[trajectory].states.push_back(states[trajectory]);
      ++steps[trajectory];
    }
  }
  return trajectories;
}

markovkit::Trajectory else_trajectory(const markovkit::ReactionSystem& model,
                                      const std::vector<double>& rates,
                                      const markovkit::State& initial,
                                      double initial_time,
                                      double final_time,
                                      ELSEOptions options,
                                      int seed) {
  auto trajectories =
      else_ensemble(model, rates, initial, 1, initial_time, final_time, options, seed);
  return std::move(trajectories.front());
}

std::vector<cme::RestrictedGenerator> density_subnetworks(const markovkit::ReactionSystem& model,
                                                          const std::vector<double>& rates,
                                                          const markovkit::State& initial,
                                                          int count,
                                                          ELSEOptions options) {
  model.validate();
  validate(options);
  if (count < 1) {
    throw std::invalid_argument("ELSE density requires at least one subnetwork");
  }

  cme::StateSpace workspace;
  StateWeights density{{initial, 1.0}};
  std::vector<cme::RestrictedGenerator> result;
  result.reserve(static_cast<std::size_t>(count));

  for (int number = 0; number < count; ++number) {
    // Normalize the incoming boundary mass before building the next subnetwork.
    const double mass =
        std::accumulate(density.begin(), density.end(), 0.0, [](double total, const auto& item) {
          return total + item.second;
        });
    if (!(mass > options.tolerance)) {
      break;
    }
    for (auto& [state, weight] : density) {
      weight /= mass;
    }

    auto subnetwork = prepare_subnetwork(workspace, density, model, rates, options);
    num::Vector entrance(subnetwork.size(), 0.0);
    for (const auto& [state, weight] : density) {
      entrance[static_cast<num::idx>(subnetwork.find(state))] = weight;
    }
    const auto occupation = subnetwork.occupation(entrance);

    // Boundary flux becomes the entrance measure for the next subnetwork.
    StateWeights next;
    for (const auto& transition : subnetwork.boundary()) {
      const double flux = transition.rate * occupation[transition.source];
      if (flux > options.tolerance) {
        next[transition.destination] += flux;
      }
    }

    cme::RestrictedGenerator stored;
    // Store only the data needed by the density solver.
    stored.states = subnetwork.states();
    for (num::idx index = 0; index < subnetwork.size(); ++index) {
      stored.index[stored.states[index]] = static_cast<int>(index);
    }
    stored.generator = subnetwork.generator();
    stored.boundary = subnetwork.boundary();
    result.push_back(std::move(stored));

    if (next.empty()) {
      break;
    }
    density = std::move(next);
  }
  return result;
}

} // namespace else_sim
