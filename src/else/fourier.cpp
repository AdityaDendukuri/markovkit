#include "else/fourier.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <unordered_map>

namespace else_sim {

namespace {
constexpr double kPi = std::numbers::pi_v<double>;
}

std::vector<num::cplx> fourier_shifts(int modes, double period, double shift) {
  if (modes < 1 || !(period > 0.0)) {
    throw std::invalid_argument("fourier_shifts: invalid modes or period");
  }
  std::vector<num::cplx> shifts;
  shifts.reserve(modes + 1);
  for (int mode = 0; mode <= modes; ++mode) {
    shifts.emplace_back(shift, mode * kPi / period);
  }
  return shifts;
}

FourierCoefficients::FourierCoefficients(num::idx state_count,
                                         int mode_count,
                                         double period,
                                         double shift)
    : states(state_count),
      modes(mode_count),
      period(period),
      shift(shift),
      values(state_count * (mode_count + 1), num::cplx(0.0, 0.0)) {}

FourierCoefficients density_transform(const std::vector<DensityStep>& steps,
                                      int modes,
                                      double period,
                                      double shift,
                                      std::vector<MultiIndex>* states_out) {
  if (steps.empty()) {
    throw std::invalid_argument("density_transform: empty steps");
  }

  std::vector<MultiIndex> all_states;
  fsp::FlatHashMap<MultiIndex, num::idx> pos_map;
  for (const auto& step : steps) {
    for (const auto& s : step.states) {
      if (!pos_map.count(s)) {
        pos_map[s] = all_states.size();
        all_states.push_back(s);
      }
    }
  }

  FourierCoefficients coeffs(all_states.size(), modes, period, shift);
  const auto shifts = fourier_shifts(modes, period, shift);

  for (int mode = 0; mode <= modes; ++mode) {
    std::unordered_map<MultiIndex, num::cplx> arrival;
    for (const auto& [s, p] : steps.front().start) {
      arrival[s] = p * steps.front().mass;
    }

    for (std::size_t step_i = 0; step_i < steps.size(); ++step_i) {
      const auto& step = steps[step_i];
      fsp::StateSpace sp;
      sp.states = step.states;
      sp.index = step.index;

      // Dummy reaction system since we already have the step generator
      std::vector<num::cplx> rhs(step.states.size(), num::cplx(0.0, 0.0));
      for (const auto& [st, val] : arrival) {
        auto it = step.index.find(st);
        if (it != step.index.end()) {
          rhs[it->second] += val;
        }
      }

      // Solve (sI - R) x = rhs using dense or sparse resolvent
      std::vector<num::cplx> x;
      if (num::klu_available() && step.states.size() > 200) {
        num::SparseResolventSolver solver(step.generator);
        solver.factorize(shifts[mode]);
        x = solver.solve(rhs);
      } else {
        num::Matrix dense(step.states.size(), step.states.size(), 0.0);
        for (num::idx r = 0; r < step.generator.n_rows(); ++r) {
          for (num::idx k = step.generator.row_ptr()[r]; k < step.generator.row_ptr()[r + 1]; ++k) {
            dense(r, step.generator.col_idx()[k]) = step.generator.values()[k];
          }
        }
        x = num::ResolventFactor(shifts[mode], dense).solve(rhs);
      }

      for (num::idx i = 0; i < step.states.size(); ++i) {
        coeffs(pos_map[step.states[i]], mode) += x[i];
      }

      arrival.clear();
      for (const auto& edge : step.edges) {
        arrival[step.destinations[edge.destination]] += edge.rate * x[edge.source];
      }
    }
  }

  if (states_out) {
    *states_out = std::move(all_states);
  }
  return coeffs;
}

std::vector<double>
invert_fourier(const FourierCoefficients& coefficients, double time, bool normalize) {
  if (time < 0.0) {
    throw std::invalid_argument("invert_fourier: negative time");
  }

  std::vector<double> result(coefficients.states, 0.0);
  for (num::idx i = 0; i < coefficients.states; ++i) {
    double val = 0.5 * coefficients(i, 0).real();
    for (int mode = 1; mode <= coefficients.modes; ++mode) {
      double angle = mode * kPi * time / coefficients.period;
      double fejer = 1.0 - (static_cast<double>(mode) / (coefficients.modes + 1.0));
      val += fejer * (coefficients(i, mode) * std::polar(1.0, angle)).real();
    }
    val *= std::exp(coefficients.shift * time) / coefficients.period;
    result[i] = std::max(0.0, val);
  }

  double total = std::accumulate(result.begin(), result.end(), 0.0);
  if (normalize && total > 0.0) {
    for (double& v : result) {
      v /= total;
    }
  }
  return result;
}

num::Vector
propagate_talbot(const Subnetwork& sub, const num::Vector& v, double time, num::idx nodes) {
  if (nodes % 2 != 0) {
    nodes += 1;
  }

  std::vector<num::cplx> rhs(v.size());
  for (num::idx i = 0; i < v.size(); ++i) {
    rhs[i] = v[i];
  }

  num::Vector result(v.size(), 0.0);
  for (const auto& node : num::talbot_nodes(time, nodes)) {
    const auto solution = sub.solve_resolvent(node.shift, rhs);
    for (num::idx i = 0; i < result.size(); ++i) {
      result[i] += (node.weight * solution[i]).real();
    }
  }
  return result;
}

struct TalbotDensitySolver::Impl {
  explicit Impl(std::vector<DensityStep> input_steps)
      : steps(std::move(input_steps)) {
    if (steps.empty()) {
      throw std::invalid_argument("Talbot density requires at least one ELSE step");
    }
    if (!num::sparse_resolvent_available()) {
      throw std::runtime_error("Talbot density requires the sparse complex resolvent backend");
    }

    for (const auto& step : steps) {
      if (step.states.empty() || step.generator.n_rows() != step.states.size() ||
          step.generator.n_cols() != step.states.size()) {
        throw std::invalid_argument("invalid ELSE density step");
      }
      for (const auto& state : step.states) {
        if (!position.count(state)) {
          position[state] = states.size();
          states.push_back(state);
        }
      }
      resolvents.push_back(std::make_unique<num::SparseResolventSolver>(step.generator));
      rhs.emplace_back(step.states.size(), num::cplx(0.0, 0.0));
      occupation.emplace_back(step.states.size(), num::cplx(0.0, 0.0));
    }
  }

  std::vector<DensitySolution>
  solve(const std::vector<std::map<MultiIndex, double>>& initial, double time, num::idx nodes) {
    if (!(time > 0.0)) {
      throw std::invalid_argument("Talbot density time must be positive");
    }
    if (nodes < 2 || nodes % 2 != 0) {
      throw std::invalid_argument("Talbot node count must be positive and even");
    }

    const std::size_t count = initial.size();
    std::vector<std::vector<num::cplx>> density(
        count, std::vector<num::cplx>(states.size(), num::cplx(0.0, 0.0)));

    for (const auto& node : num::talbot_nodes(time, nodes)) {
      std::vector<std::map<MultiIndex, num::cplx>> arrivals(count);
      for (std::size_t sample = 0; sample < count; ++sample) {
        for (const auto& [state, mass] : initial[sample]) {
          arrivals[sample][state] = mass;
        }
      }
      std::vector<std::vector<num::cplx>> totals(
          count, std::vector<num::cplx>(states.size(), num::cplx(0.0, 0.0)));

      for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
        const auto& step = steps[step_index];
        resolvents[step_index]->factorize(node.shift);
        for (std::size_t sample = 0; sample < count; ++sample) {
          auto& right_hand_side = rhs[step_index];
          std::fill(right_hand_side.begin(), right_hand_side.end(), num::cplx(0.0, 0.0));
          for (const auto& [state, mass] : arrivals[sample]) {
            const auto local = step.index.find(state);
            if (local != step.index.end()) {
              right_hand_side[static_cast<num::idx>(local->second)] += mass;
            }
          }

          auto& time_spent = occupation[step_index];
          resolvents[step_index]->solve(right_hand_side, time_spent);
          for (num::idx local = 0; local < step.states.size(); ++local) {
            totals[sample][position[step.states[local]]] += time_spent[local];
          }

          auto& arrival = arrivals[sample];
          arrival.clear();
          for (const auto& edge : step.edges) {
            if (edge.source >= time_spent.size() || edge.destination >= step.destinations.size()) {
              throw std::invalid_argument("invalid ELSE boundary edge");
            }
            arrival[step.destinations[edge.destination]] += edge.rate * time_spent[edge.source];
          }
        }
      }

      for (std::size_t sample = 0; sample < count; ++sample) {
        for (num::idx state = 0; state < states.size(); ++state) {
          density[sample][state] += node.weight * totals[sample][state];
        }
      }
    }

    std::vector<DensitySolution> result;
    result.reserve(count);
    for (std::size_t sample = 0; sample < count; ++sample) {
      DensitySolution solution{states, std::vector<double>(states.size(), 0.0)};
      double mass = 0.0;
      for (num::idx state = 0; state < states.size(); ++state) {
        solution.probability[state] = std::max(0.0, density[sample][state].real());
        mass += solution.probability[state];
      }
      if (mass > 0.0) {
        for (double& probability : solution.probability) {
          probability /= mass;
        }
      }
      result.push_back(std::move(solution));
    }
    return result;
  }

  std::vector<DensityStep> steps;
  std::vector<MultiIndex> states;
  fsp::FlatHashMap<MultiIndex, num::idx> position;
  std::vector<std::unique_ptr<num::SparseResolventSolver>> resolvents;
  std::vector<std::vector<num::cplx>> rhs;
  std::vector<std::vector<num::cplx>> occupation;
};

TalbotDensitySolver::TalbotDensitySolver(std::vector<DensityStep> steps)
    : impl_(std::make_unique<Impl>(std::move(steps))) {}

TalbotDensitySolver::~TalbotDensitySolver() = default;
TalbotDensitySolver::TalbotDensitySolver(TalbotDensitySolver&&) noexcept = default;
TalbotDensitySolver& TalbotDensitySolver::operator=(TalbotDensitySolver&&) noexcept = default;

const std::vector<MultiIndex>& TalbotDensitySolver::states() const {
  return impl_->states;
}

DensitySolution TalbotDensitySolver::solve(const MultiIndex& initial, double time, num::idx nodes) {
  return solve(std::map<MultiIndex, double>{{initial, 1.0}}, time, nodes);
}

DensitySolution TalbotDensitySolver::solve(const std::map<MultiIndex, double>& initial,
                                           double time,
                                           num::idx nodes) {
  return impl_->solve({initial}, time, nodes).front();
}

std::vector<DensitySolution> TalbotDensitySolver::solve(
    const std::vector<MultiIndex>& initial_states, double time, num::idx nodes) {
  std::vector<std::map<MultiIndex, double>> initial;
  initial.reserve(initial_states.size());
  for (const auto& state : initial_states) {
    initial.push_back({{state, 1.0}});
  }
  return impl_->solve(initial, time, nodes);
}

} // namespace else_sim
