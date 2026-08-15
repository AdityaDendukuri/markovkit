#include <markovkit.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

void test_fess_fourier_resolvent() {
    using namespace markovkit::fess;

    num::Matrix R(2, 2, 0.0);
    R(0, 0) = -2.0; R(0, 1) = 1.0;
    R(1, 0) = 1.0;  R(1, 1) = -2.0;

    num::Vector v{1.0, 0.0};
    double t = 0.5;

    // 1. Compute via Talbot contour resolvent solves using num::resolvent_solve
    num::Vector p_fourier = fourier_subnetwork_solve(R, v, t, 32);

    // 2. Compute via exact matrix exponential action num::expv
    num::operators::DenseOp Rop(R);
    num::Vector p_expv = num::expv(t, Rop, v);

    std::cout << "FESS Talbot Fourier: p[0] = " << p_fourier[0] << ", p[1] = " << p_fourier[1] << "\n";
    std::cout << "Exact expv:          p[0] = " << p_expv[0] << ", p[1] = " << p_expv[1] << "\n";

    assert(std::abs(p_fourier[0] - p_expv[0]) < 1e-3);
    assert(std::abs(p_fourier[1] - p_expv[1]) < 1e-3);

    std::cout << "[PASS] FESS Talbot Fourier contour resolvent solve matches expv!\n";
}

int main() {
    std::cout << "--- Running FESS Resolvent Tests ---\n";
    test_fess_fourier_resolvent();
    std::cout << "All FESS tests passed successfully!\n";
    return 0;
}
