#include "markovkit.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_gillespie_birth_death() {
    markovkit::reaction_system model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const markovkit::state &, const num::array<double> &r, double) { return r[0]; },
        [](const markovkit::state &x, const num::array<double> &r, double) { return r[1] * x[0]; }};

    num::array<double> rates = {5.0, 0.1};
    markovkit::state initial{0};

    const int paths = 2000;
    double mean = 0.0;
    for (int i = 0; i < paths; ++i) {
        auto path = ssa::gillespie(model, rates, initial, 0.0, 2.0, 100 + i);
        mean += path.states.back()[0];
    }
    mean /= paths;

    double expected = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
    assert(std::abs(mean - expected) < 0.5);
    std::cout << "[PASS] test_gillespie_birth_death (mean=" << mean << ", expected=" << expected
              << ")\n";
}

void test_gillespie_step_limit() {
    markovkit::reaction_system model{
        .changes = {{1}},
        .propensities = {[](const markovkit::state &, const auto &, double) { return 1.0; }}};
    const auto path = ssa::gillespie(model, {}, {0}, 0.0, 100.0, 42, 5);
    assert(path.reactions.size() == 5);
    assert(path.states.size() == 6);
    assert(path.times.back() < 100.0);
}

void test_reaction_and_trajectory_helpers() {
    markovkit::reaction_system model{
        .changes = {{1}, {-1}},
        .propensities = {
            [](const markovkit::state &, const auto &, double) { return 2.0; },
            [](const markovkit::state &, const auto &, double) { return -1.0; },
        }};
    num::array<double> propensities(2, 0.0);
    assert(model.evaluate_propensities({3}, {}, 0.0, propensities) == 2.0);
    assert(propensities[0] == 2.0);
    assert(propensities[1] == 0.0);
    assert(model.total_propensity({3}, {}, 0.0) == 2.0);

    const markovkit::trajectory trajectory{{0.0, 1.0, 3.0}, {{2}, {4}, {8}}, {0, 0}};
    const num::vec counts = markovkit::trajectory_component(trajectory, 0);
    assert(counts.size() == 3 && counts[0] == 2.0 && counts[1] == 4.0 && counts[2] == 8.0);
    const num::array<double> observations{0.5, 1.0, 2.0, 4.0};
    assert(markovkit::trajectory_indices_at(trajectory, observations) ==
           num::array<std::size_t>({0, 1, 1, 2}));
}

void test_templated_rng() {
    markovkit::reaction_system model{
        .changes = {{1}},
        .propensities = {[](const markovkit::state &, const auto &, double) { return 1.0; }}};
    std::mt19937_64 custom_rng(12345);
    const auto path = ssa::gillespie(model, {}, {0}, 0.0, 100.0, custom_rng, 5);
    assert(path.reactions.size() == 5);
    assert(path.states.size() == 6);
    assert(path.times.back() < 100.0);
}

int main() {
    test_gillespie_birth_death();
    test_gillespie_step_limit();
    test_templated_rng();
    test_reaction_and_trajectory_helpers();
    std::cout << "All SSA tests passed.\n";
    return 0;
}
