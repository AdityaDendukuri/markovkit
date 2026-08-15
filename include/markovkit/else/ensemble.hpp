/// @file else/ensemble.hpp
/// @brief Ensemble ELSE: multi-trajectory stochastic sampling sharing subnetwork solves.
#pragma once

#include "markovkit/else/subnetwork.hpp"
#include "markovkit/else/fourier.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace markovkit::else_solver {

struct EnsembleTrajectory {
    int current_state;
    double time;
    bool active;
};

struct EnsembleResult {
    int n_trajectories;
    double final_time;
    std::vector<int> final_states;
    std::vector<double> exit_times;
};

/// @brief Run Ensemble ELSE over N parallel trajectories sharing a subnetwork generator solve.
inline EnsembleResult ensemble_else_simulate(
    std::size_t n_trajectories,
    const Subnetwork& subnetwork,
    double t_max,
    int rng_seed = 42
) {
    num::idx n_states = subnetwork.R.rows();
    std::mt19937 rng(rng_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    EnsembleResult res;
    res.n_trajectories = static_cast<int>(n_trajectories);
    res.final_time = t_max;
    res.final_states.resize(n_trajectories, 0);
    res.exit_times.resize(n_trajectories, 0.0);

    // For each trajectory k in 0..n_trajectories-1
    for (std::size_t k = 0; k < n_trajectories; ++k) {
        int state = 0; // Initial state in subnetwork
        double t = 0.0;

        while (t < t_max) {
            // Mean holding time in current state: tau = Z(state, state)
            double tau = subnetwork.Z(state, state);
            if (tau <= 0.0) break;

            // Exponential holding time sample
            double u = dist(rng);
            double dt = -tau * std::log(u);
            if (t + dt > t_max) {
                t = t_max;
                break;
            }

            t += dt;

            // Transition probability out of state i to j: P_ij = R_ij / (-R_ii)
            double r_ii = -subnetwork.R(state, state);
            double r = dist(rng) * r_ii;
            double cumulative = 0.0;
            int next_state = state;

            for (num::idx j = 0; j < n_states; ++j) {
                if (j == (num::idx)state) continue;
                cumulative += subnetwork.R(state, j);
                if (r <= cumulative) {
                    next_state = static_cast<int>(j);
                    break;
                }
            }

            state = next_state;
        }

        res.final_states[k] = state;
        res.exit_times[k] = t;
    }

    return res;
}

} // namespace markovkit::else_solver
