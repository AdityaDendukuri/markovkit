#include "markovkit.hpp"
#include "check.hpp"
#include <cmath>
#include <iostream>

void test_state_space_expand_compress() {
    markovkit::reaction_system model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const markovkit::state &, const num::array<double> &r, double) { return r[0]; },
        [](const markovkit::state &x, const num::array<double> &r, double) { return r[1] * x[0]; }};

    cme::state_space sp;
    sp.add_state({0}, 1.0);
    check::that(sp.size() == 1, "sp.size() == 1");

    sp.expand(model, cme::rect_boundary(10), 2);
    check::that(sp.size() == 3, "sp.size() == 3"); // 0, 1, 2

    num::array<double> rates = {5.0, 0.1};
    auto generator = cme::build_generator(sp, model, rates, 0.0, false);
    check::that(generator.matrix.n_rows() == 3, "generator.matrix.n_rows() == 3");
    check::that(generator.diagonal_rates[0] == 5.0, "generator.diagonal_rates[0] == 5.0");

    auto [flux, n_removed] = sp.compress(model, rates, 0.0, 0.5, 1e-6);
    check::that(sp.size() <= 3, "sp.size() <= 3");

    double mass = sp.renormalize();
    check::that(std::abs(mass - 1.0) < 1e-12, "std::abs(mass - 1.0) < 1e-12");
    std::cout << "[PASS] test_state_space_expand_compress\n";
}

void test_adaptive_fsp_birth_death() {
    markovkit::reaction_system model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const markovkit::state &, const num::array<double> &r, double) { return r[0]; },
        [](const markovkit::state &x, const num::array<double> &r, double) { return r[1] * x[0]; }};

    fsp::fsp_problem prob;
    prob.model = model;
    prob.u0 = {0};
    prob.t0 = 0.0;
    prob.tf = 2.0;
    prob.rates = {5.0, 0.1};
    prob.bc = cme::rect_boundary(100);

    fsp::adaptive_fsp_params params;
    params.eps_dt = 0.1;
    params.flux_tolerance = 1e-6;

    auto [sol, diag] = fsp::solve_adaptive_fsp(prob, params);
    check::that(sol.n_snapshots() > 1, "sol.n_snapshots() > 1");

    const num::mat<double> means = sol.mean_trajectory();
    double final_mean = means(means.rows() - 1, 0);
    // Analytical mean for birth-death with p0=0: (k_b / k_d) * (1 - exp(-k_d * t))
    // = (5.0 / 0.1) * (1 - exp(-0.1 * 2.0)) = 50 * (1 - 0.81873) = 9.063
    double expected_mean = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
    check::that(std::abs(final_mean - expected_mean) < 0.1, "std::abs(final_mean - expected_mean) < 0.1");
    std::cout << "[PASS] test_adaptive_fsp_birth_death (mean=" << final_mean
              << ", expected=" << expected_mean << ")\n";
}

void test_distribution_statistics() {
    const num::array<markovkit::state> states{{0, 2}, {2, 4}};
    const num::array<double> probabilities{0.25, 0.75};
    const num::vec<double> means = markovkit::species_means(states, probabilities);
    check::that(means.size() == 2 && means[0] == 1.5 && means[1] == 3.5, "means.size() == 2 && means[0] == 1.5 && means[1] == 3.5");
    const auto marginal = markovkit::marginal_distribution(states, probabilities, 0);
    check::that(marginal.at(0) == 0.25, "marginal.at(0) == 0.25");
    check::that(marginal.at(2) == 0.75, "marginal.at(2) == 0.75");
}

void test_krylov_fsp_pure_death() {
    fsp::fsp_problem problem;
    problem.model.changes = {{-1}};
    problem.model.propensities = {
        [](const markovkit::state &x, const num::array<double> &, double) {
            return static_cast<double>(x[0]);
        }};
    problem.u0 = {5};
    problem.tf = 1.0;
    problem.bc = cme::nonnegative_boundary();
    fsp::krylov_fsp_params options;
    options.eps = 1e-10;
    options.krylov_tol = 1e-12;
    options.max_iter = 100;
    const auto [solution, diagnostics] = fsp::solve_krylov_fsp(problem, options);
    const auto means = solution.mean_trajectory();
    check::that(solution.times.back() == problem.tf, "Krylov FSP reaches final time");
    check::that(std::abs(means(means.rows() - 1, 0) - 5.0 * std::exp(-1.0)) < 1e-8,
                "Krylov FSP matches the exact pure-death mean");
    check::that(std::abs(solution.marginal(solution.n_snapshots() - 1, 0).at(0) -
                         std::pow(1.0 - std::exp(-1.0), 5)) < 1e-8,
                "Krylov FSP matches the exact extinction probability");
}

void test_support_compaction() {
    cme::state_space space;
    space.add_state({0}, 0.1);
    space.add_state({1}, 0.2);
    space.add_state({2}, 0.3);
    space.add_state({3}, 0.4);
    space.remove_states({true, false, true, false});
    check::that(space.states == num::array<markovkit::state>{{1}, {3}},
                "compaction retains state order");
    check::that(space.probs == num::array<double>{0.2, 0.4},
                "probabilities remain aligned with states");
    check::that(space.find({1}) == 0 && space.find({3}) == 1 && space.find({2}) == -1,
                "compaction rebuilds local indices");
    space.remove_states({true, true});
    check::that(space.empty() && space.probs.empty() && space.index.empty(),
                "removing all states clears storage and lookup");
}

int main() {
    test_support_compaction();
    test_krylov_fsp_pure_death();
    test_state_space_expand_compress();
    test_adaptive_fsp_birth_death();
    test_distribution_statistics();
    std::cout << "All FSP tests passed.\n";
    return check::report("fsp");
}
