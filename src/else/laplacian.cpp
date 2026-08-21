#include "else/laplacian.hpp"
#include <algorithm>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>

namespace else_sim {
namespace {

markovkit::State graph_state(num::idx state) {
  return markovkit::State{static_cast<int>(state)};
}

} // namespace

std::vector<num::idx> laplacian_neighborhood(const num::SparseMatrix& laplacian,
                                             const std::vector<double>& stationary_sqrt,
                                             num::idx initial,
                                             num::idx capacity) {
  const cme::ReversibleLaplacian validated(laplacian, stationary_sqrt);
  (void)validated;
  if (initial >= laplacian.n_rows() || capacity == 0 || capacity > laplacian.n_rows()) {
    throw std::invalid_argument("invalid Laplacian neighborhood request");
  }

  std::vector<bool> selected(laplacian.n_rows(), false);
  std::vector<bool> queued(laplacian.n_rows(), false);
  std::priority_queue<std::pair<double, num::idx>> frontier;
  std::vector<num::idx> states{initial};
  selected[initial] = true;

  // Connectivity constrains the support; stationary weight orders its frontier.
  const auto expose = [&](num::idx state) {
    for (num::idx k = laplacian.row_ptr()[state]; k < laplacian.row_ptr()[state + 1]; ++k) {
      const num::idx neighbor = laplacian.col_idx()[k];
      if (neighbor != state && !selected[neighbor] && !queued[neighbor]) {
        queued[neighbor] = true;
        frontier.emplace(stationary_sqrt[neighbor], neighbor);
      }
    }
  };

  expose(initial);
  // Grow the connected region from the highest-weight frontier state.
  while (states.size() < capacity) {
    if (frontier.empty()) {
      throw std::invalid_argument("initial component is smaller than the requested capacity");
    }
    const num::idx state = frontier.top().second;
    frontier.pop();
    if (selected[state]) {
      continue;
    }
    selected[state] = true;
    states.push_back(state);
    expose(state);
  }
  return states;
}

std::vector<cme::RestrictedGenerator>
laplacian_restrictions(const num::SparseMatrix& laplacian,
                       const std::vector<double>& stationary_sqrt,
                       num::idx initial,
                       num::idx capacity,
                       int count) {
  if (count < 1) {
    throw std::invalid_argument("Laplacian density requires at least one subnetwork");
  }

  auto states = laplacian_neighborhood(laplacian, stationary_sqrt, initial, capacity);
  std::vector<bool> included(laplacian.n_rows(), false);
  for (num::idx state : states) {
    included[state] = true;
  }

  std::vector<cme::RestrictedGenerator> subnetworks;
  subnetworks.reserve(static_cast<std::size_t>(count));
  // Expand each restriction with the previous boundary states.
  for (int number = 0; number < count; ++number) {
    subnetworks.push_back(cme::restrict_laplacian(laplacian, stationary_sqrt, states));
    if (subnetworks.back().boundary.empty()) {
      break;
    }
    // The next restriction contains every state reached through this boundary.
    for (const auto& transition : subnetworks.back().boundary) {
      const auto state = static_cast<num::idx>(transition.destination[0]);
      if (!included[state]) {
        included[state] = true;
        states.push_back(state);
      }
    }
  }
  return subnetworks;
}

markovkit::Trajectory laplacian_else_trajectory(const cme::ReversibleLaplacian& generator,
                                                num::idx initial,
                                                double initial_time,
                                                double final_time,
                                                ELSEOptions options,
                                                int seed) {
  const auto& laplacian = generator.matrix();
  const auto& stationary_sqrt = generator.stationary_sqrt();
  if (initial >= laplacian.n_rows() || initial_time > final_time || options.capacity == 0 ||
      options.capacity > laplacian.n_rows()) {
    throw std::invalid_argument("invalid Laplacian trajectory request");
  }

  std::mt19937 random(seed);
  markovkit::Trajectory trajectory;
  trajectory.times.push_back(initial_time);
  trajectory.states.push_back(graph_state(initial));

  num::idx current = initial;
  double time = initial_time;
  std::size_t completed_steps = 0;
  // Replace each local excursion with one sampled exit event.
  for (; completed_steps < options.maximum_steps && time < final_time; ++completed_steps) {
    const auto states =
        laplacian_neighborhood(laplacian, stationary_sqrt, current, options.capacity);
    auto assembled = cme::restrict_laplacian(laplacian, stationary_sqrt, states);
    const auto local_h =
        num::gather(std::span<const double>(stationary_sqrt), std::span<const num::idx>(states));

    Subnetwork subnetwork(cme::reversible_generator(std::move(assembled), local_h));
    if (subnetwork.boundary_states().empty()) {
      break;
    }

    num::Matrix source(subnetwork.size(), 1, 0.0);
    source(static_cast<num::idx>(subnetwork.find(graph_state(current))), 0) = 1.0;
    const auto integrals = subnetwork.occupation_integrals(source);
    const auto exit_probabilities = subnetwork.exit_probabilities(integrals);
    // Sample the local state through which the path exits.
    const num::idx local_exit =
        subnetwork.boundary_states()[num::sample_categorical(exit_probabilities, random)];
    const double waiting_time = subnetwork.conditional_exit_time(integrals, local_exit);

    std::vector<num::idx> destinations;
    std::vector<double> rates;
    // Resolve the chosen boundary state to its external destinations.
    for (const auto& transition : subnetwork.boundary()) {
      if (transition.source == local_exit) {
        destinations.push_back(static_cast<num::idx>(transition.destination[0]));
        rates.push_back(transition.rate);
      }
    }
    if (!(waiting_time > 0.0) || destinations.empty() || time + waiting_time >= final_time) {
      break;
    }

    time += waiting_time;
    current = destinations[num::sample_categorical(rates, random)];
    trajectory.times.push_back(time);
    trajectory.states.push_back(graph_state(current));
  }

  if (completed_steps < options.maximum_steps && time < final_time &&
      final_time < std::numeric_limits<double>::infinity()) {
    // Extend a completed path to the finite observation horizon.
    trajectory.times.push_back(final_time);
    trajectory.states.push_back(graph_state(current));
  }
  return trajectory;
}

} // namespace else_sim
