// stochastic simulation: sample paths of a chain
#pragma once

#include "markovkit/chain.hpp"
#include "stochastic/categorical.hpp"
#include <algorithm>
#include <limits>
#include <random>
#include <utility>

namespace markovkit {

// a piecewise-constant path: the chain is in states[k] from times[k] until times[k + 1]
struct trajectory {
    array<real> times;
    array<state> states;

    // the count of one species along the path
    [[nodiscard]] vec<real> component(idx species) const {
        vec<real> result(states.size());
        for (idx k = 0; k < states.size(); ++k) {
            result[k] = states[k][species];
        }
        return result;
    }

    // the index of the state occupied at each of a sorted list of times
    [[nodiscard]] array<idx> indices_at(view<const real> observation_times) const {
        array<idx> result;
        idx k = 0;
        for (real t : observation_times) {
            while (k + 1 < times.size() && times[k + 1] <= t) {
                ++k;
            }
            result.push_back(k);
        }
        return result;
    }
};

// Gillespie's direct method, from time 0 to final_time or until max_jumps jumps
template <markov_model Model, std::uniform_random_bit_generator Rng>
trajectory gillespie(const Model &model, state x, real final_time, Rng &random,
                     idx max_jumps = std::numeric_limits<idx>::max()) {
    trajectory path{{0.0}, {x}};
    real t = 0.0;
    array<real> rates;
    for (idx jump = 0; jump < max_jumps; ++jump) {
        auto jumps = model.transitions(x);
        rates.clear();
        real total = 0.0;
        for (const auto &e : jumps) {
            rates.push_back(e.rate);
            total += e.rate;
        }
        if (!(total > 0.0)) {
            break;
        }
        t += std::exponential_distribution<real>(total)(random);
        if (t > final_time) {
            break;
        }
        x = std::move(jumps[num::sample_categorical(view<const real>(rates), random)].to);
        path.times.push_back(t);
        path.states.push_back(x);
    }
    // a path that stopped short of max_jumps holds its last state until final_time
    if (path.states.size() <= max_jumps) {
        path.times.push_back(final_time);
        path.states.push_back(x);
    }
    return path;
}

struct leap_options {
    real step = 0.0;     // a fixed leap; zero chooses each leap from epsilon
    real epsilon = 0.03; // expected number of jumps per leap
    real min_step = 1e-6;
};

// Poisson tau-leaping: each transition fires a Poisson number of times per leap
template <markov_model Model, std::uniform_random_bit_generator Rng>
trajectory tau_leaping(const Model &model, state x, real final_time, Rng &random,
                       leap_options options = {}) {
    trajectory path{{0.0}, {x}};
    real t = 0.0;
    while (t < final_time) {
        const auto jumps = model.transitions(x);
        real total = 0.0;
        for (const auto &e : jumps) {
            total += e.rate;
        }
        if (!(total > 0.0)) {
            break;
        }
        real dt = options.step > 0.0 ? options.step
                                     : std::max(options.min_step, options.epsilon / total);
        dt = std::min(dt, final_time - t);
        t += dt;

        state next = x;
        for (const auto &[y, rate] : jumps) {
            const int fired = std::poisson_distribution<int>(rate * dt)(random);
            for (idx s = 0; s < next.size(); ++s) {
                next[s] = std::max(0, next[s] + (fired * (y[s] - x[s])));
            }
        }
        x = std::move(next);
        path.times.push_back(t);
        path.states.push_back(x);
    }
    return path;
}

} // namespace markovkit
