#include "markovkit.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_state_space_expand_compress() {
    markovkit::ReactionSystem model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const markovkit::State &, const std::vector<double> &r, double) { return r[0]; },
        [](const markovkit::State &x, const std::vector<double> &r, double) {
            return r[1] * x[0];
        }};

    cme::StateSpace sp;
    sp.add_state({0}, 1.0);
    assert(sp.size() == 1);

    sp.expand(model, cme::rect_boundary(10), 2);
    assert(sp.size() == 3); // 0, 1, 2

    std::vector<double> rates = {5.0, 0.1};
    auto generator = cme::build_generator(sp, model, rates, 0.0, false);
    assert(generator.matrix.n_rows() == 3);
    assert(generator.diagonal_rates[0] == 5.0);

    auto [flux, n_removed] = sp.compress(model, rates, 0.0, 0.5, 1e-6);
    assert(sp.size() <= 3);

    double mass = sp.renormalize();
    assert(std::abs(mass - 1.0) < 1e-12);
    std::cout << "[PASS] test_state_space_expand_compress\n";
}

void test_adaptive_fsp_birth_death() {
    markovkit::ReactionSystem model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const markovkit::State &, const std::vector<double> &r, double) { return r[0]; },
        [](const markovkit::State &x, const std::vector<double> &r, double) {
            return r[1] * x[0];
        }};

    fsp::FSPProblem prob;
    prob.model = model;
    prob.u0 = {0};
    prob.t0 = 0.0;
    prob.tf = 2.0;
    prob.rates = {5.0, 0.1};
    prob.bc = cme::rect_boundary(100);

    fsp::AdaptiveFSPParams params;
    params.eps_dt = 0.1;
    params.flux_tolerance = 1e-6;

    auto [sol, diag] = fsp::solve_adaptive_fsp(prob, params);
    assert(sol.n_snapshots() > 1);

    auto means = sol.mean_trajectory();
    double final_mean = means.back()[0];
    // Analytical mean for birth-death with p0=0: (k_b / k_d) * (1 - exp(-k_d * t))
    // = (5.0 / 0.1) * (1 - exp(-0.1 * 2.0)) = 50 * (1 - 0.81873) = 9.063
    double expected_mean = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
    assert(std::abs(final_mean - expected_mean) < 0.1);
    std::cout << "[PASS] test_adaptive_fsp_birth_death (mean=" << final_mean
              << ", expected=" << expected_mean << ")\n";
}

void test_distribution_statistics() {
    const std::vector<markovkit::State> states{{0, 2}, {2, 4}};
    const std::vector<double> probabilities{0.25, 0.75};
    const auto means = markovkit::species_means(states, probabilities);
    assert(means == std::vector<double>({1.5, 3.5}));
    const auto marginal = markovkit::marginal_distribution(states, probabilities, 0);
    assert(marginal.at(0) == 0.25);
    assert(marginal.at(2) == 0.75);
}

int main() {
    test_state_space_expand_compress();
    test_adaptive_fsp_birth_death();
    test_distribution_statistics();
    std::cout << "All FSP tests passed.\n";
    return 0;
}
