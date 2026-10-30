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

void test_distribution() {
    distribution p;
    p.add({0}, 0.1);
    p.add({1}, 0.2);
    p.add({2}, 0.3);
    p.add({3}, 0.4);
    p.add({3}, 9.0); // already present: ignored
    check::that(p.size() == 4, "add ignores a state already present");

    const vec<real> m = p.means();
    check::that(std::abs(m[0] - 2.0) < 1e-15, "mean of a four-point distribution");

    p.remove_if([](idx i) { return i % 2 == 0; });
    check::that(p.states() == array<state>{{1}, {3}}, "remove_if keeps state order");
    check::that(p.probabilities() == array<real>{0.2, 0.4}, "probabilities stay aligned");
    check::that(p.find({1}) == 0 && p.find({3}) == 1 && !p.find({2}), "lookup is rebuilt");

    const real old_total = p.normalize();
    check::that(std::abs(old_total - 0.6) < 1e-15, "normalize returns the old total");
    check::that(std::abs(p.marginal(0).at(3) - 2.0 / 3.0) < 1e-15, "marginal after normalize");

    p.remove_if([](idx) { return true; });
    check::that(p.size() == 0 && !p.find({1}), "removing everything clears the lookup");
}

void test_expand_and_generator() {
    const auto model = birth_death(5.0, 0.1);
    distribution p(state{0});
    expand(p, model, box(10), 2);
    check::that(p.size() == 3, "two layers from {0} reach {1} and {2}");

    const generator Q(p, model, boundary::reflecting);
    check::that(Q.rows() == 3, "one row per state");
    check::that(Q.exit_rates()[0] == 5.0, "exit rate of {0} is the birth rate");
    check::that(std::abs(Q.leak_rates()[2] - 5.0) < 1e-15, "birth out of {2} leaks");

    const generator A(p, model, boundary::absorbing);
    check::that(std::abs(A.exit_rates()[2] - 5.2) < 1e-15, "absorbing keeps the leak on Q_jj");

    vec<real> x(3, 0.0), free(3, 0.0), assembled(3, 0.0);
    x[0] = 0.25;
    x[1] = 0.5;
    x[2] = 0.25;
    Q.apply(x, free);
    Q.matrix().apply(x, assembled);
    real column_sum = 0.0;
    for (idx i = 0; i < 3; ++i) {
        check::that(std::abs(free[i] - assembled[i]) < 1e-15, "matrix() matches apply()");
        column_sum += free[i];
    }
    check::that(std::abs(column_sum) < 1e-15, "a reflecting generator conserves probability");
}

void test_adaptive_fsp_birth_death() {
    const auto solution = solve_adaptive_fsp(birth_death(5.0, 0.1), {0}, 2.0,
                                             {.inside = box(100), .step_scale = 0.1});
    check::that(solution.times.size() > 1, "adaptive FSP saves snapshots");
    const auto means = solution.means();
    const real mean = means(means.rows() - 1, 0);
    const real exact = (5.0 / 0.1) * (1.0 - std::exp(-0.1 * 2.0));
    check::that(std::abs(mean - exact) < 0.1, "adaptive FSP matches the birth-death mean");
    std::cout << "adaptive FSP mean " << mean << ", exact " << exact << '\n';
}

void test_krylov_fsp_pure_death() {
    const reaction_network death{{{{-1}, [](const state &x) { return real(x[0]); }}}};
    const auto solution =
        solve_krylov_fsp(death, {5}, 1.0, {.tolerance = 1e-10, .max_steps = 100, .krylov_tol = 1e-12});
    check::that(solution.times.back() == 1.0, "Krylov FSP reaches the final time");
    const auto means = solution.means();
    check::that(std::abs(means(means.rows() - 1, 0) - 5.0 * std::exp(-1.0)) < 1e-8,
                "Krylov FSP matches the pure-death mean");
    check::that(std::abs(solution.snapshots.back().marginal(0).at(0) -
                         std::pow(1.0 - std::exp(-1.0), 5)) < 1e-8,
                "Krylov FSP matches the extinction probability");
}

} // namespace

int main() {
    test_distribution();
    test_expand_and_generator();
    test_adaptive_fsp_birth_death();
    test_krylov_fsp_pure_death();
    return check::report("fsp");
}
