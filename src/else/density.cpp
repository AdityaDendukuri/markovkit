#include "else/density.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace else_sim {

// Precomputed boundary edge transfer entry from subnetwork s to s+1
struct BoundaryTransfer {
  num::idx source;
  num::idx target_local;
  double rate;
};

class ShiftedSolver {
public:
  ShiftedSolver(const num::SparseMatrix& matrix, bool symmetric_pattern) {
    // LAPACK has lower overhead for small systems; sparse fill dominates larger ones.
    constexpr num::idx dense_limit = 128;
    if (matrix.n_rows() <= dense_limit) {
      dense = std::make_unique<num::DenseResolventSolver>(matrix);
      return;
    }
    if (!num::sparse_resolvent_available()) {
      throw std::runtime_error("large Talbot systems require the sparse complex backend");
    }
    sparse = std::make_unique<num::SparseResolventSolver>(
        matrix, num::SparseResolventOptions{.symmetric_pattern = symmetric_pattern});
  }

  void factorize(num::cplx shift) {
    if (dense) {
      dense->factorize(shift);
    } else {
      sparse->factorize(shift);
    }
  }

  void solve(const std::vector<num::cplx>& rhs, std::vector<num::cplx>& result) const {
    if (dense) {
      dense->solve(rhs, result);
    } else {
      sparse->solve(rhs, result);
    }
  }

private:
  std::unique_ptr<num::DenseResolventSolver> dense;
  std::unique_ptr<num::SparseResolventSolver> sparse;
};

struct TalbotDensitySolver::Impl {
  explicit Impl(std::vector<RestrictedGenerator> input_subnetworks)
      : subnetworks(std::move(input_subnetworks)) {
    if (subnetworks.empty()) {
      throw std::invalid_argument("Talbot density requires at least one ELSE subnetwork");
    }
    // Build global states and shifted solvers per subnetwork.
    for (std::size_t subnetwork_index = 0; subnetwork_index < subnetworks.size();
         ++subnetwork_index) {
      const auto& subnetwork = subnetworks[subnetwork_index];
      if (subnetwork.states.empty() || subnetwork.generator.n_rows() != subnetwork.states.size() ||
          subnetwork.generator.n_cols() != subnetwork.states.size()) {
        throw std::invalid_argument("invalid ELSE density subnetwork");
      }
      for (const auto& state : subnetwork.states) {
        if (!position.count(state)) {
          position[state] = states.size();
          states.push_back(state);
        }
      }
      solvers.emplace_back(subnetwork.generator, subnetwork.symmetric_pattern);
    }

    // Resolve global boundary states to local indices once, outside the contour loop.
    transfers.resize(subnetworks.size());
    for (std::size_t subnetwork_index = 0; subnetwork_index + 1 < subnetworks.size();
         ++subnetwork_index) {
      const auto& next_subnetwork = subnetworks[subnetwork_index + 1];
      for (const auto& edge : subnetworks[subnetwork_index].boundary) {
        auto it = next_subnetwork.index.find(edge.destination);
        if (it != next_subnetwork.index.end()) {
          transfers[subnetwork_index].push_back(
              {edge.source, static_cast<num::idx>(it->second), edge.rate});
        }
      }
    }
  }

  std::vector<DensitySolution> solve(const std::vector<std::map<markovkit::State, double>>& initial,
                                     double time,
                                     num::idx nodes) {
    if (!(time > 0.0)) {
      throw std::invalid_argument("Talbot density time must be positive");
    }
    if (nodes < 2 || nodes % 2 != 0) {
      throw std::invalid_argument("Talbot node count must be positive and even");
    }

    const std::size_t n_samples = initial.size();
    const std::size_t n_global = states.size();

    // Complex Talbot contributions are accumulated in the union of all states.
    std::vector<std::vector<num::cplx>> density(
        n_samples, std::vector<num::cplx>(n_global, num::cplx(0.0, 0.0)));

    // Reuse arrival and occupation buffers at every contour node.
    std::vector<std::vector<std::vector<num::cplx>>> arrivals(subnetworks.size());
    std::vector<std::vector<std::vector<num::cplx>>> occupations(subnetworks.size());
    for (std::size_t s = 0; s < subnetworks.size(); ++s) {
      arrivals[s].assign(n_samples,
                         std::vector<num::cplx>(subnetworks[s].states.size(), num::cplx(0.0, 0.0)));
      occupations[s].assign(
          n_samples, std::vector<num::cplx>(subnetworks[s].states.size(), num::cplx(0.0, 0.0)));
    }

    for (const auto& node : num::talbot_nodes(time, nodes)) {
      for (std::size_t s = 0; s < subnetworks.size(); ++s) {
        for (std::size_t sample = 0; sample < n_samples; ++sample) {
          std::fill(arrivals[s][sample].begin(), arrivals[s][sample].end(), num::cplx(0.0, 0.0));
        }
      }

      // Initial mass enters the first subnetwork only.
      for (std::size_t sample = 0; sample < n_samples; ++sample) {
        for (const auto& [state, mass] : initial[sample]) {
          auto it = subnetworks[0].index.find(state);
          if (it != subnetworks[0].index.end()) {
            arrivals[0][sample][static_cast<num::idx>(it->second)] += mass;
          }
        }
      }

      // Factor once per shift and reuse it for all initial conditions.
      for (std::size_t s = 0; s < subnetworks.size(); ++s) {
        solvers[s].factorize(node.shift);

        for (std::size_t sample = 0; sample < n_samples; ++sample) {
          solvers[s].solve(arrivals[s][sample], occupations[s][sample]);
          const auto& occ = occupations[s][sample];

          // node.weight already includes the Talbot quadrature scale.
          for (num::idx local = 0; local < subnetworks[s].states.size(); ++local) {
            density[sample][position[subnetworks[s].states[local]]] += node.weight * occ[local];
          }

          // B_s (zI-R_s)^-1 supplies the next subnetwork's arrival density.
          if (s + 1 < subnetworks.size()) {
            for (const auto& tf : transfers[s]) {
              arrivals[s + 1][sample][tf.target_local] += tf.rate * occ[tf.source];
            }
          }
        }
      }
    }

    // Roundoff may create tiny negative real parts; clip before normalization.
    std::vector<DensitySolution> result;
    result.reserve(n_samples);
    for (std::size_t sample = 0; sample < n_samples; ++sample) {
      DensitySolution solution{states, std::vector<double>(n_global, 0.0)};
      double mass = 0.0;
      for (num::idx i = 0; i < n_global; ++i) {
        solution.probability[i] = std::max(0.0, density[sample][i].real());
        mass += solution.probability[i];
      }
      if (mass > 0.0) {
        for (double& p : solution.probability) {
          p /= mass;
        }
      }
      result.push_back(std::move(solution));
    }
    return result;
  }

  std::vector<RestrictedGenerator> subnetworks;
  std::vector<markovkit::State> states;
  std::unordered_map<markovkit::State, num::idx> position;
  std::vector<ShiftedSolver> solvers;
  std::vector<std::vector<BoundaryTransfer>> transfers;
};

TalbotDensitySolver::TalbotDensitySolver(std::vector<RestrictedGenerator> subnetworks)
    : impl_(std::make_unique<Impl>(std::move(subnetworks))) {}

TalbotDensitySolver::~TalbotDensitySolver() = default;
TalbotDensitySolver::TalbotDensitySolver(TalbotDensitySolver&&) noexcept = default;
TalbotDensitySolver& TalbotDensitySolver::operator=(TalbotDensitySolver&&) noexcept = default;

const std::vector<markovkit::State>& TalbotDensitySolver::states() const {
  return impl_->states;
}

DensitySolution
TalbotDensitySolver::solve(const markovkit::State& initial, double time, num::idx nodes) {
  return solve(std::map<markovkit::State, double>{{initial, 1.0}}, time, nodes);
}

DensitySolution TalbotDensitySolver::solve(const std::map<markovkit::State, double>& initial,
                                           double time,
                                           num::idx nodes) {
  return impl_->solve({initial}, time, nodes).front();
}

std::vector<DensitySolution> TalbotDensitySolver::solve(
    const std::vector<markovkit::State>& initial_states, double time, num::idx nodes) {
  std::vector<std::map<markovkit::State, double>> initial;
  initial.reserve(initial_states.size());
  for (const auto& state : initial_states) {
    initial.push_back({{state, 1.0}});
  }
  return impl_->solve(initial, time, nodes);
}

} // namespace else_sim
