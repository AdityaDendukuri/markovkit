// adaptive FSP for the stochastic Oregonator (Field-Koros-Noyes oscillator)
#include "markovkit.hpp"
#include "plot/plot.hpp"

using namespace markovkit;

int main() {
    // these scales reproduce the stochastic Oregonator regime
    const real y1 = 500.0, y2 = 1000.0, y3 = 2000.0;
    const real mu1 = 2000.0, mu2 = 50000.0;
    const real k1 = mu1 / y2;              // 2.0
    const real k2 = mu2 / (y1 * y2);       // 0.1
    const real k3 = (mu1 + mu2) / y1;      // 104.0
    const real k4 = 2.0 * mu1 / (y1 * y1); // 0.016
    const real k5 = (mu1 + mu2) / y3;      // 26.0

    const reaction_network model{{
        {{1, -1, 0}, [=](const state &x) { return k1 * x[1]; }},
        {{-1, -1, 0}, [=](const state &x) { return k2 * x[0] * x[1]; }},
        {{1, 0, 1}, [=](const state &x) { return k3 * x[0]; }},
        {{-2, 0, 0}, [=](const state &x) { return 0.5 * k4 * x[0] * (x[0] - 1); }},
        {{0, 1, -1}, [=](const state &x) { return k5 * x[2]; }},
    }};

    // sampled expansion keeps the support small; start from the deterministic counts
    const auto solution = solve_adaptive_fsp(model, {500, 1000, 2000}, 2.0,
                                             {.inside = box(50000),
                                              .flux_tolerance = 1.0,
                                              .sampled_expansion = true});
    const auto means = solution.means();

    array<real> x, y, z, sizes;
    for (idx k = 0; k < solution.times.size(); ++k) {
        x.push_back(means(k, 0));
        y.push_back(means(k, 1));
        z.push_back(means(k, 2));
        sizes.push_back(real(solution.snapshots[k].size()));
    }

    num::plt::subplot(2, 2);
    num::plt::plot(solution.times, x, "X", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean X");
    num::plt::next();
    num::plt::plot(solution.times, y, "Y", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean Y");
    num::plt::next();
    num::plt::plot(solution.times, z, "Z", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean Z");
    num::plt::next();
    num::plt::plot(solution.times, sizes, "states", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("states");
    num::plt::savefig("fsp_oregonator.png");
}
