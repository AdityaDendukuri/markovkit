// adaptive FSP for a bistable genetic toggle switch
#include "markovkit.hpp"
#include "plot/plot.hpp"

using namespace markovkit;

int main() {
    // mutual cubic repression creates two preferred expression states
    const real alpha = 20.0, beta = 400.0, K3 = 100.0 * 100.0 * 100.0;
    const real d_u = 1.0 + (0.1 / 1.1), d_v = 1.0;

    const reaction_network model{{
        {{1, 0}, [=](const state &x) { return alpha + (beta * K3 / (K3 + std::pow(x[1], 3))); }},
        {{-1, 0}, [=](const state &x) { return d_u * x[0]; }},
        {{0, 1}, [=](const state &x) { return alpha + (beta * K3 / (K3 + std::pow(x[0], 3))); }},
        {{0, -1}, [=](const state &x) { return d_v * x[1]; }},
    }};

    // absolute pruning keeps the support compact
    const auto solution = solve_adaptive_fsp(model, {100, 100}, 10.0,
                                             {.inside = box(500),
                                              .step_scale = 10.0,
                                              .prune_fraction = 0.0,
                                              .flux_tolerance = 0.0,
                                              .min_probability = 1e-6,
                                              .layers = 5,
                                              .save_every = 200});
    const auto means = solution.means();

    array<real> u, v, sizes;
    for (idx k = 0; k < solution.times.size(); ++k) {
        u.push_back(means(k, 0));
        v.push_back(means(k, 1));
        sizes.push_back(real(solution.snapshots[k].size()));
    }

    // the final U and V marginals
    array<real> counts_u, probs_u, counts_v, probs_v;
    for (const auto &[count, p] : solution.snapshots.back().marginal(0)) {
        if (p > 1e-8) {
            counts_u.push_back(count);
            probs_u.push_back(p);
        }
    }
    for (const auto &[count, p] : solution.snapshots.back().marginal(1)) {
        if (p > 1e-8) {
            counts_v.push_back(count);
            probs_v.push_back(p);
        }
    }

    num::plt::subplot(2, 2);
    num::plt::plot(solution.times, u, "U", "lines lw 2");
    num::plt::plot(solution.times, v, "V", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean");
    num::plt::legend();
    num::plt::next();
    num::plt::plot(solution.times, sizes, "states", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("states");
    num::plt::next();
    num::plt::plot(counts_u, probs_u, "U", "boxes");
    num::plt::xlabel("count");
    num::plt::ylabel("probability");
    num::plt::next();
    num::plt::plot(counts_v, probs_v, "V", "boxes");
    num::plt::xlabel("count");
    num::plt::ylabel("probability");
    num::plt::savefig("fsp_toggle_switch.png");
}
