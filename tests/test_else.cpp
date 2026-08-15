#include <markovkit.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

void test_exponential_fit() {
    using namespace markovkit::else_solver;

    num::Vector a{0.5, 0.5};
    num::Vector b{1.0, 2.0};
    double c = 0.5;

    double t = exponential_fit(a, b, c);
    double val = a[0] * std::exp(-t * b[0]) + a[1] * std::exp(-t * b[1]);
    assert(std::abs(val - c) < 1e-6);
    std::cout << "[PASS] ELSE exponential fit (t = " << t << ")\n";
}

void test_discrete_subnetwork() {
    using namespace markovkit::else_solver;

    num::Matrix R(2, 2, 0.0);
    R(0, 0) = -2.0; R(0, 1) = 1.0;
    R(1, 0) = 1.0;  R(1, 1) = -2.0;

    DiscreteReferenceSubnetwork sub;
    sub.set_rate_matrix(R);

    const auto& Z = sub.fundamental_matrix();
    assert(Z.rows() == 2 && Z.cols() == 2);
    assert(Z(0, 0) > 0.0);

    double t_esc = sub.escape_time(0);
    assert(t_esc > 0.0);
    std::cout << "[PASS] ELSE DiscreteReferenceSubnetwork escape time (t_esc = " << t_esc << ")\n";
}

int main() {
    std::cout << "--- Running ELSE Tests ---\n";
    test_exponential_fit();
    test_discrete_subnetwork();
    std::cout << "All ELSE tests passed successfully!\n";
    return 0;
}
