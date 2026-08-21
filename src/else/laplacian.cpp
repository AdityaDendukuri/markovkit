#include "else/laplacian.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace else_sim {
namespace {

markovkit::State graph_state(num::idx state) {
  return markovkit::State{static_cast<int>(state)};
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

// Column generator: R = -diag(h) * L * diag(h)^-1, i.e. R_ij = -L_ij * (h_i / h_j)
// Satisfies 1^T R = 0 (column sums zero) and R_ij >= 0 (i != j)
DensitySubnetwork laplacian_density_subnetwork(const num::SparseMatrix& laplacian,
                                               const std::vector<double>& stationary_sqrt,
                                               const std::vector<num::idx>& states) {
  validate_laplacian(laplacian, stationary_sqrt);
  if (states.empty()) {
    throw std::invalid_argument("a Laplacian density subnetwork requires states");
  }

  DensitySubnetwork subnetwork;
  subnetwork.states.reserve(states.size());
  for (num::idx state : states) {
    if (state >= laplacian.n_rows()) {
      throw std::out_of_range("Laplacian state is out of range");
    }
    const auto id = graph_state(state);
    if (subnetwork.index.count(id)) {
      throw std::invalid_argument("Laplacian density states must be unique");
    }
    subnetwork.index[id] = static_cast<int>(subnetwork.states.size());
    subnetwork.states.push_back(id);
  }

  std::vector<num::idx> rows;
  std::vector<num::idx> columns;
  std::vector<double> values;

  for (num::idx local_column = 0; local_column < states.size(); ++local_column) {
    const num::idx global_column = states[local_column];
    bool diagonal_found = false;
    for (num::idx k = laplacian.row_ptr()[global_column];
         k < laplacian.row_ptr()[global_column + 1];
         ++k) {
      const num::idx global_row = laplacian.col_idx()[k];
      if (global_row == global_column) {
        // Keep the full diagonal so omitted off-diagonal rates remain exit rates.
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

      const auto local = subnetwork.index.find(graph_state(global_row));
      if (local != subnetwork.index.end()) {
        rows.push_back(static_cast<num::idx>(local->second));
        columns.push_back(local_column);
        values.push_back(rate);
      } else {
        subnetwork.boundary.push_back({local_column, graph_state(global_row), rate});
      }
    }
    if (!diagonal_found) {
      throw std::invalid_argument("Laplacian requires an explicit diagonal");
    }
  }

  subnetwork.generator =
      num::SparseMatrix::from_triplets(states.size(), states.size(), rows, columns, values);
  subnetwork.symmetric_pattern = true;
  return subnetwork;
}

std::vector<DensitySubnetwork>
laplacian_density_subnetworks(const num::SparseMatrix& laplacian,
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

  std::vector<DensitySubnetwork> subnetworks;
  subnetworks.reserve(static_cast<std::size_t>(count));
  for (int number = 0; number < count; ++number) {
    subnetworks.push_back(laplacian_density_subnetwork(laplacian, stationary_sqrt, states));
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

} // namespace else_sim
