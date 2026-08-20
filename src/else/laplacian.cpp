#include "else/laplacian.hpp"

#include <algorithm>
#include <map>
#include <queue>
#include <stdexcept>

namespace else_sim {
namespace {

MultiIndex graph_state(num::idx state) {
  return MultiIndex{static_cast<int>(state)};
}

void validate_laplacian(const num::SparseMatrix& laplacian,
                        const std::vector<double>& stationary_sqrt) {
  if (laplacian.n_rows() != laplacian.n_cols()) {
    throw std::invalid_argument("Laplacian must be square");
  }
  if (stationary_sqrt.size() != laplacian.n_rows()) {
    throw std::invalid_argument("stationary weights must match the Laplacian size");
  }
  if (!std::all_of(stationary_sqrt.begin(), stationary_sqrt.end(), [](double value) {
        return value > 0.0;
      })) {
    throw std::invalid_argument("stationary weights must be positive");
  }
}

} // namespace

std::vector<num::idx> laplacian_neighborhood(const num::SparseMatrix& laplacian,
                                             const std::vector<double>& stationary_sqrt,
                                             num::idx initial,
                                             num::idx capacity) {
  validate_laplacian(laplacian, stationary_sqrt);
  if (initial >= laplacian.n_rows() || capacity == 0 || capacity > laplacian.n_rows()) {
    throw std::invalid_argument("invalid Laplacian neighborhood request");
  }

  std::vector<bool> selected(laplacian.n_rows(), false);
  std::vector<bool> queued(laplacian.n_rows(), false);
  std::priority_queue<std::pair<double, num::idx>> frontier;
  std::vector<num::idx> states{initial};
  selected[initial] = true;

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

DensityStep laplacian_density_step(const num::SparseMatrix& laplacian,
                                   const std::vector<double>& stationary_sqrt,
                                   const std::vector<num::idx>& states) {
  validate_laplacian(laplacian, stationary_sqrt);
  if (states.empty()) {
    throw std::invalid_argument("a Laplacian density step requires states");
  }

  DensityStep step;
  step.states.reserve(states.size());
  for (num::idx state : states) {
    if (state >= laplacian.n_rows()) {
      throw std::out_of_range("Laplacian state is out of range");
    }
    const auto id = graph_state(state);
    if (step.index.count(id)) {
      throw std::invalid_argument("Laplacian density states must be unique");
    }
    step.index[id] = static_cast<int>(step.states.size());
    step.states.push_back(id);
  }

  std::vector<num::idx> rows;
  std::vector<num::idx> columns;
  std::vector<double> values;
  std::map<num::idx, num::idx> destination_index;

  for (num::idx local_column = 0; local_column < states.size(); ++local_column) {
    const num::idx global_column = states[local_column];
    bool diagonal_found = false;
    for (num::idx k = laplacian.row_ptr()[global_column];
         k < laplacian.row_ptr()[global_column + 1];
         ++k) {
      const num::idx global_row = laplacian.col_idx()[k];
      if (global_row == global_column) {
        rows.push_back(local_column);
        columns.push_back(local_column);
        values.push_back(-laplacian.values()[k]);
        diagonal_found = true;
        continue;
      }

      const double rate =
          -laplacian.values()[k] * stationary_sqrt[global_row] / stationary_sqrt[global_column];
      if (rate < 0.0) {
        throw std::invalid_argument("Laplacian has a positive off-diagonal entry");
      }
      if (rate == 0.0) {
        continue;
      }

      const auto local = step.index.find(graph_state(global_row));
      if (local != step.index.end()) {
        rows.push_back(static_cast<num::idx>(local->second));
        columns.push_back(local_column);
        values.push_back(rate);
      } else {
        const auto [destination, inserted] =
            destination_index.emplace(global_row, destination_index.size());
        if (inserted) {
          step.destinations.push_back(graph_state(global_row));
        }
        step.edges.push_back({local_column, destination->second, rate});
      }
    }
    if (!diagonal_found) {
      throw std::invalid_argument("Laplacian requires an explicit diagonal");
    }
  }

  step.generator =
      num::SparseMatrix::from_triplets(states.size(), states.size(), rows, columns, values);
  return step;
}

std::vector<DensityStep> laplacian_density_steps(const num::SparseMatrix& laplacian,
                                                 const std::vector<double>& stationary_sqrt,
                                                 num::idx initial,
                                                 num::idx capacity,
                                                 int step_count) {
  if (step_count < 1) {
    throw std::invalid_argument("Laplacian density requires at least one step");
  }

  auto states = laplacian_neighborhood(laplacian, stationary_sqrt, initial, capacity);
  std::vector<bool> included(laplacian.n_rows(), false);
  for (num::idx state : states) {
    included[state] = true;
  }

  std::vector<DensityStep> steps;
  steps.reserve(static_cast<std::size_t>(step_count));
  for (int number = 0; number < step_count; ++number) {
    auto step = laplacian_density_step(laplacian, stationary_sqrt, states);
    if (number == 0) {
      step.start[graph_state(initial)] = 1.0;
    }
    steps.push_back(std::move(step));
    if (steps.back().destinations.empty()) {
      break;
    }
    for (const auto& destination : steps.back().destinations) {
      const auto state = static_cast<num::idx>(destination[0]);
      if (!included[state]) {
        included[state] = true;
        states.push_back(state);
      }
    }
  }
  return steps;
}

} // namespace else_sim
