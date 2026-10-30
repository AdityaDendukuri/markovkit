// finite state projection: evolve a distribution on a support that grows and shrinks with it
#pragma once

#include "container/matrix.hpp"
#include "linear/expv/expv.hpp"
#include "markovkit/distribution.hpp"
#include "markovkit/generator.hpp"
#include "stats/selection.hpp"
#include "stochastic/categorical.hpp"
#include "stochastic/rng.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>

namespace markovkit {

// the states an FSP may add to its support
using region = std::function<bool(const state &)>;

inline region everywhere() {
    return [](const state &) { return true; };
}

// 0 <= x[s] <= bounds[s] for every coordinate
inline region box(array<int> bounds) {
    return [bounds = std::move(bounds)](const state &x) {
        for (idx s = 0; s < x.size(); ++s) {
            if (x[s] < 0 || x[s] > bounds[s]) {
                return false;
            }
        }
        return true;
    };
}

// 0 <= x[s] <= bound for every coordinate
inline region box(int bound) {
    return [bound](const state &x) {
        return std::all_of(x.begin(), x.end(), [&](int c) { return c >= 0 && c <= bound; });
    };
}

// add `layers` rings of neighbors inside the region; after the first ring, only the
// states added by the previous ring can have new neighbors
template <markov_model Model>
void expand(distribution &p, const Model &model, const region &inside, int layers = 1) {
    idx begin = 0;
    for (int layer = 0; layer < layers; ++layer) {
        const idx end = p.size();
        for (idx i = std::exchange(begin, end); i < end; ++i) {
            for (const auto &[y, rate] : model.transitions(p[i])) {
                if (inside(y)) {
                    p.add(y);
                }
            }
        }
    }
}

// add one neighbor per state and layer, chosen with probability proportional to its rate
template <markov_model Model>
void expand_sampled(distribution &p, const Model &model, const region &inside, int layers,
                    num::rng &random) {
    array<real> rates;
    for (int layer = 0; layer < layers; ++layer) {
        const idx n = p.size();
        for (idx i = 0; i < n; ++i) {
            const auto jumps = model.transitions(p[i]);
            if (jumps.empty()) {
                continue;
            }
            rates.clear();
            for (const auto &e : jumps) {
                rates.push_back(e.rate);
            }
            const state &y = jumps[num::sample_categorical(view<const real>(rates), random)].to;
            if (inside(y)) {
                p.add(y);
            }
        }
    }
}

// the distribution at each saved time
struct fsp_solution {
    array<real> times;
    array<distribution> snapshots;

    // means(k, s): the mean count of species s at times[k]
    [[nodiscard]] num::mat<real> means() const {
        const idx species = snapshots.front()[0].size();
        num::mat<real> result(snapshots.size(), species, 0.0);
        for (idx k = 0; k < snapshots.size(); ++k) {
            const vec<real> m = snapshots[k].means();
            for (idx s = 0; s < species; ++s) {
                result(k, s) = m[s];
            }
        }
        return result;
    }

    void save(real t, const distribution &p) {
        times.push_back(t);
        snapshots.push_back(p);
    }
};

struct adaptive_fsp_options {
    region inside = everywhere();
    real step_scale = 1.0;         // each step expects about this many jumps
    real prune_fraction = 0.1;     // the least likely fraction of states may be pruned
    real flux_tolerance = 1e-6;    // ... if its probability flux is below this share of the total
    real min_probability = 0.0;    // also prune any state below this probability
    int layers = 1;                // neighbor rings added per step
    bool sampled_expansion = false; // add sampled neighbors instead of every neighbor
    unsigned seed = 42;
    int save_every = 1000;
    int max_steps = 1 << 28;
    int krylov_dim = 30;
    real krylov_tol = 1e-8;
};

// FSP with flux-controlled steps: expand, propagate with expv, prune, renormalize
template <markov_model Model>
fsp_solution solve_adaptive_fsp(const Model &model, const state &initial, real final_time,
                                const adaptive_fsp_options &options = {}) {
    distribution p(initial);
    fsp_solution solution;
    solution.save(0.0, p);
    num::rng random(options.seed);

    real t = 0.0;
    for (int step = 1; t < final_time && step <= options.max_steps; ++step) {
        if (options.sampled_expansion) {
            expand_sampled(p, model, options.inside, options.layers, random);
        } else {
            expand(p, model, options.inside, options.layers);
        }
        const generator Q(p, model, boundary::reflecting);
        auto &prob = p.probabilities();

        // the expected jump rate sets the step
        real activity = 0.0;
        for (idx i = 0; i < p.size(); ++i) {
            activity += prob[i] * Q.exit_rates()[i];
        }
        const real dt = activity > 0.0 ? std::min(options.step_scale / activity, final_time - t)
                                       : final_time - t;

        const vec<real> next =
            num::expv(dt, Q.matrix(), vec<real>(prob), options.krylov_dim, options.krylov_tol);
        for (idx i = 0; i < p.size(); ++i) {
            prob[i] = std::max(0.0, next[i]);
        }

        // prune unlikely states whose probability flux is negligible
        const idx n = p.size();
        array<char> drop(n, 0);
        if (n > 1) {
            array<real> flux(n);
            real total_flux = 0.0;
            for (idx i = 0; i < n; ++i) {
                flux[i] = prob[i] * (Q.exit_rates()[i] + Q.leak_rates()[i]);
                total_flux += flux[i];
            }
            const auto candidates =
                std::min<idx>(static_cast<idx>(std::round(options.prune_fraction * real(n))), n - 1);
            for (idx i : num::smallest_indices(view<const real>(prob), candidates)) {
                drop[i] = flux[i] < options.flux_tolerance * total_flux ? 1 : 0;
            }
            for (idx i = 0; i < n; ++i) {
                drop[i] = (drop[i] || prob[i] < options.min_probability) ? 1 : 0;
            }
        }
        p.remove_if([&](idx i) { return drop[i]; });
        p.normalize();

        t += dt;
        if (step % options.save_every == 0 || t >= final_time) {
            solution.save(t, p);
        }
    }
    if (solution.times.back() < t) {
        solution.save(t, p);
    }
    return solution;
}

struct krylov_fsp_options {
    region inside = everywhere();
    real tolerance = 1e-6;         // the probability that may leak by final_time
    real initial_step = 1.0;
    real step_growth = 2.0;        // step multiplier after an accepted step
    real drop_probability = 1e-10; // drop a state below this probability ...
    real drop_derivative = 1e-16;  // ... whose probability is also changing slower than this
    int layers = 1;
    int save_every = 1000;
    int max_steps = 1 << 28;
    int krylov_dim = 30;
    real krylov_tol = 1e-8;
};

// the Krylov FSP of Vo and Sidje: a step is accepted when the absorbed probability
// stays within its share of the tolerance, and halved otherwise
template <markov_model Model>
fsp_solution solve_krylov_fsp(const Model &model, const state &initial, real final_time,
                              const krylov_fsp_options &options = {}) {
    distribution p(initial);
    fsp_solution solution;
    solution.save(0.0, p);

    real t = 0.0;
    real tau = options.initial_step;
    int accepted = 0;
    for (int attempt = 0; t < final_time && attempt < options.max_steps; ++attempt) {
        tau = std::min(tau, final_time - t);
        expand(p, model, options.inside, options.layers);
        const generator Q(p, model, boundary::absorbing);
        auto &prob = p.probabilities();

        const vec<real> w =
            num::expv(tau, Q.matrix(), vec<real>(prob), options.krylov_dim, options.krylov_tol);
        real mass = 0.0;
        for (real value : w) {
            mass += std::max(0.0, value);
        }
        if (mass < 1.0 - (options.tolerance * (t + tau) / final_time) && tau > 1e-14) {
            tau *= 0.5;
            continue;
        }

        vec<real> dp(w.size());
        Q.apply(w, dp);
        for (idx i = 0; i < p.size(); ++i) {
            prob[i] = std::max(0.0, w[i]);
        }
        p.remove_if([&](idx i) {
            return prob[i] < options.drop_probability && std::abs(dp[i]) < options.drop_derivative;
        });
        p.normalize();

        t += tau;
        ++accepted;
        if (accepted % options.save_every == 0 || t >= final_time) {
            solution.save(t, p);
        }
        tau = std::min(tau * options.step_growth, final_time - t);
    }
    if (solution.times.back() < t) {
        solution.save(t, p);
    }
    return solution;
}

} // namespace markovkit
