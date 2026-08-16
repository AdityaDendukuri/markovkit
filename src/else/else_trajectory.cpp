/// @file src/else/else_trajectory.cpp
/// @brief Expected-Visits Low-Probability State Extrapolation (ELSE) trajectory solver implementation.
#include "markovkit/else/else_trajectory.hpp"
#include <cmath>
#include <map>

namespace markovkit::else_solver {

ELSETrajectory else_trajectory_simulate(const fsp::ReactionSystem& model,
                                        const std::vector<double>& rates,
                                        const fsp::MultiIndex& u0,
                                        double t0,
                                        double tf,
                                        int radius,
                                        int rng_seed) {
  std::mt19937 rng(rng_seed);
  std::uniform_real_distribution<double> u_dist(0.0, 1.0);

  ELSETrajectory result;
  result.t.push_back(t0);
  result.x.push_back(u0);

  double current_time = t0;
  fsp::MultiIndex current_state = u0;
  int step_count = 0;
  const int max_steps = 10000;

  while (current_time < tf && step_count < max_steps) {
    step_count++;

    // 1. Build local subnetwork around current_state with L1 Manhattan radius
    std::vector<fsp::MultiIndex> states;
    std::map<fsp::MultiIndex, num::idx> state_map;

    int u_center = current_state[0];
    int v_center = current_state[1];

    for (int du = -radius; du <= radius; ++du) {
      for (int dv = -(radius - std::abs(du)); dv <= (radius - std::abs(du)); ++dv) {
        int u_cand = u_center + du;
        int v_cand = v_center + dv;
        if (u_cand >= 0 && v_cand >= 0) {
          fsp::MultiIndex idx{u_cand, v_cand};
          state_map[idx] = states.size();
          states.push_back(idx);
        }
      }
    }

    num::idx n_sub = states.size();
    if (n_sub == 0) break;

    // 2. Build generator matrix Q for local subnetwork and exit rates
    num::Matrix Q(n_sub, n_sub, 0.0);
    std::vector<double> exit_rates(n_sub, 0.0);

    for (num::idx i = 0; i < n_sub; ++i) {
      const auto& state_i = states[i];
      double diag = 0.0;

      for (std::size_t r = 0; r < model.stoich_vecs.size(); ++r) {
        double a = model.propensities[r](state_i, rates, current_time);
        if (a <= 0.0) continue;

        diag += a;
        fsp::MultiIndex next_idx = state_i + model.stoich_vecs[r];
        auto it = state_map.find(next_idx);

        if (it != state_map.end()) {
          Q(i, it->second) += a;
        } else {
          exit_rates[i] += a;
        }
      }
      Q(i, i) = -diag;
    }

    // 3. Solve fundamental matrix Z = -Q^-1 to get occupation vector u = Z * e_start
    num::Matrix R = Q;
    num::Matrix Z(n_sub, n_sub, 0.0);
    try {
      auto lu_fact = num::lu(R);
      num::Matrix I(n_sub, n_sub, 0.0);
      for (num::idx i = 0; i < n_sub; ++i) I(i, i) = -1.0;
      num::lu_solve(lu_fact, I, Z);
    } catch (...) {
      break;
    }

    num::idx start_idx = state_map[current_state];
    double mean_exit_time = 0.0;
    for (num::idx j = 0; j < n_sub; ++j) {
      double z_val = Z(start_idx, j);
      if (z_val > 0.0) mean_exit_time += z_val;
    }

    if (mean_exit_time <= 1e-14 || std::isnan(mean_exit_time)) break;

    // Sample waiting time tau ~ Exponential(mean_exit_time)
    double u_rand = u_dist(rng);
    while (u_rand <= 1e-12) u_rand = u_dist(rng);
    double waiting_time = -mean_exit_time * std::log(u_rand);
    if (waiting_time <= 0.0 || std::isnan(waiting_time)) waiting_time = 1e-5;

    if (current_time + waiting_time > tf) {
      break;
    }

    // 4. Sample pre-exit state using exit_rates * max(0, Z(start_idx, i))
    std::vector<double> preexit_weights(n_sub, 0.0);
    double total_exit_weight = 0.0;
    for (num::idx i = 0; i < n_sub; ++i) {
      double z_val = std::max(0.0, Z(start_idx, i));
      preexit_weights[i] = exit_rates[i] * z_val;
      total_exit_weight += preexit_weights[i];
    }

    num::idx preexit_idx = start_idx;
    if (total_exit_weight > 1e-14) {
      double r_cutoff = u_dist(rng) * total_exit_weight;
      double r_sum = 0.0;
      for (num::idx i = 0; i < n_sub; ++i) {
        r_sum += preexit_weights[i];
        if (r_sum >= r_cutoff) {
          preexit_idx = i;
          break;
        }
      }
    }

    const auto& preexit_state = states[preexit_idx];

    // 5. Pick outside destination state from preexit_state
    std::vector<fsp::MultiIndex> outside_dest;
    std::vector<double> outside_rates;
    double total_out_rate = 0.0;

    for (std::size_t r = 0; r < model.stoich_vecs.size(); ++r) {
      double a = model.propensities[r](preexit_state, rates, current_time);
      fsp::MultiIndex dest = preexit_state + model.stoich_vecs[r];

      if (a > 0.0 && state_map.find(dest) == state_map.end()) {
        outside_dest.push_back(dest);
        outside_rates.push_back(a);
        total_out_rate += a;
      }
    }

    fsp::MultiIndex next_state = current_state;
    if (total_out_rate > 0.0) {
      double r_cutoff = u_dist(rng) * total_out_rate;
      double r_sum = 0.0;
      for (std::size_t k = 0; k < outside_dest.size(); ++k) {
        r_sum += outside_rates[k];
        if (r_sum >= r_cutoff) {
          next_state = outside_dest[k];
          break;
        }
      }
    } else {
      break;
    }

    current_time += waiting_time;
    result.t.push_back(current_time);
    result.x.push_back(preexit_state);
    result.t.push_back(current_time);
    result.x.push_back(next_state);

    current_state = next_state;
    result.number_of_else_steps++;
  }

  return result;
}

} // namespace markovkit::else_solver
