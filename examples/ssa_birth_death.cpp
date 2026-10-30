// Gillespie SSA and tau-leaping on a birth-death process
#include "markovkit.hpp"
#include "plot/plot.hpp"

using namespace markovkit;

int main() {
    const reaction_network model{{
        {{1}, [](const state &) { return 5.0; }},
        {{-1}, [](const state &x) { return 0.1 * x[0]; }},
    }};

    num::rng random(42);
    const auto exact = gillespie(model, {0}, 20.0, random);
    const auto leaped = tau_leaping(model, {0}, 20.0, random);

    num::plt::plot(exact.times, exact.component(0), "Gillespie", "steps lw 2");
    num::plt::plot(leaped.times, leaped.component(0), "tau-leaping", "steps lw 2 dt 2");
    num::plt::xlabel("time");
    num::plt::ylabel("count");
    num::plt::legend();
    num::plt::savefig("ssa_birth_death.png");
}
