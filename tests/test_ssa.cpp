#include "check.hpp"
#include "markovkit.hpp"
#include <cmath>
#include <iostream>

using namespace markovkit;

namespace {

reaction_network birth_death(real birth, real death) {
    return {{{{1}, [=](const state &) { return birth; }},
             {{-1}, [=](const state &x) { return death * x[0]; }}}};
}

void test_transitions() {
    const reaction_network model{{{{1}, [](const state &) { return 2.0; }},
                                  {{-1}, [](const state &) { return -1.0; }},
                                  {{-4}, [](const state &) { return 1.0; }}}};
    const auto jumps = model.transitions({3});
    check::that(jumps.size() == 1, "nonpositive rates and negative counts are skipped");
    check::that(jumps[0].to == state{4} && jumps[0].rate == 2.0, "the birth is listed");
}

void test_gillespie_birth_death() {
    const auto model = birth_death(5.0, 0.1);
    num::rng random(100);
    const int paths = 2000;
    real mean = 0.0;
    for (int k = 0; k < paths; ++k) {
        mean += gillespie(model, {0}, 2.0, random).states.back()[0];
    }
    mean /= paths;
    const real exact = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
    check::that(std::abs(mean - exact) < 0.5, "Gillespie matches the birth-death mean");
    std::cout << "Gillespie mean " << mean << ", exact " << exact << '\n';
}

void test_tau_leaping_birth_death() {
    const auto model = birth_death(5.0, 0.1);
    num::rng random(7);
    const int paths = 500;
    real mean = 0.0;
    for (int k = 0; k < paths; ++k) {
        mean += tau_leaping(model, {0}, 2.0, random).states.back()[0];
    }
    mean /= paths;
    const real exact = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
    check::that(std::abs(mean - exact) < 1.0, "tau-leaping matches the birth-death mean");
}

void test_jump_limit_and_custom_rng() {
    const reaction_network model{{{{1}, [](const state &) { return 1.0; }}}};
    std::mt19937_64 random(12345);
    const auto path = gillespie(model, {0}, 100.0, random, 5);
    check::that(path.states.size() == 6, "five jumps after the initial state");
    check::that(path.times.back() < 100.0, "a path cut at max_jumps ends at its last jump");
}

void test_trajectory_helpers() {
    const trajectory path{{0.0, 1.0, 3.0}, {{2}, {4}, {8}}};
    const vec<real> counts = path.component(0);
    check::that(counts.size() == 3 && counts[0] == 2.0 && counts[2] == 8.0, "component");
    const array<real> observations{0.5, 1.0, 2.0, 4.0};
    check::that(path.indices_at(observations) == array<idx>{0, 1, 1, 2}, "indices_at");
}

} // namespace

int main() {
    test_transitions();
    test_gillespie_birth_death();
    test_tau_leaping_birth_death();
    test_jump_limit_and_custom_rng();
    test_trajectory_helpers();
    return check::report("ssa");
}
