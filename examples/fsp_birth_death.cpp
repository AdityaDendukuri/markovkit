#include "markovkit.hpp"
#include "plot/plot.hpp"

using namespace markovkit;

int main() {
    const reaction_network model{{
        {{1}, [](const state &) { return 5.0; }},
        {{-1}, [](const state &x) { return 0.1 * x[0]; }},
    }};

    const auto solution = solve_adaptive_fsp(model, {0}, 20.0,
                                             {.inside = box(100), .step_scale = 0.5, .prune_fraction = 0.0,
                                              .layers = 5, .save_every = 5});

    for (const real target : {2.0, 5.0, 10.0, 20.0}) {
        idx k = 0;
        while (k + 1 < solution.times.size() &&
               std::abs(solution.times[k + 1] - target) < std::abs(solution.times[k] - target)) {
            ++k;
        }
        array<real> counts, probabilities;
        for (const auto &[count, probability] : solution.snapshots[k].marginal(0)) {
            counts.push_back(count);
            probabilities.push_back(probability);
        }
        num::plt::plot(counts, probabilities, "t = " + std::to_string(int(target)), "lines lw 2");
    }
    num::plt::xlabel("molecule count");
    num::plt::ylabel("probability");
    num::plt::legend();
    num::plt::savefig("fsp_birth_death.png");
}
