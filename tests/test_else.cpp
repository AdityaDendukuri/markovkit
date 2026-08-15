#include <markovkit.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

void test_subnetwork_fundamental_matrix() {
    using namespace markovkit::else_solver;

    num::Matrix R(2, 2, 0.0);
    R(0, 0) = -2.0; R(0, 1) = 1.0;
    R(1, 0) = 1.0;  R(1, 1) = -2.0;

    Subnetwork sub(R);

    // Z = -R^{-1} = 1/3 * [2 1; 1 2]
    assert(std::abs(sub.Z(0, 0) - 2.0 / 3.0) < 1e-6);
    assert(std::abs(sub.Z(0, 1) - 1.0 / 3.0) < 1e-6);

    std::cout << "[PASS] ELSE Subnetwork fundamental matrix Z = -R^{-1}\n";
}

void test_schur_rank1_shedding() {
    using namespace markovkit::else_solver;

    num::Matrix R(3, 3, 0.0);
    R(0, 0) = -3.0; R(0, 1) = 1.0; R(0, 2) = 1.0;
    R(1, 0) = 1.0;  R(1, 1) = -2.0; R(1, 2) = 0.0;
    R(2, 0) = 1.0;  R(2, 1) = 0.0;  R(2, 2) = -2.0;

    Subnetwork sub(R);
    sub.shed_state(2); // Shed state 2

    assert(sub.R.rows() == 2);
    assert(sub.Z.rows() == 2);

    std::cout << "[PASS] ELSE Schur rank-1 state shedding\n";
}

void test_talbot_fourier_resolvent() {
    using namespace markovkit::else_solver;

    num::Matrix R(2, 2, 0.0);
    R(0, 0) = -2.0; R(0, 1) = 1.0;
    R(1, 0) = 1.0;  R(1, 1) = -2.0;

    num::Vector v{1.0, 0.0};
    double t = 0.5;

    num::Vector p_fourier = fourier_subnetwork_solve(R, v, t, 32);
    num::operators::DenseOp Rop(R);
    num::Vector p_expv = num::expv(t, Rop, v);

    assert(std::abs(p_fourier[0] - p_expv[0]) < 1e-4);
    assert(std::abs(p_fourier[1] - p_expv[1]) < 1e-4);

    std::cout << "[PASS] ELSE Talbot Fourier contour resolvent solve matches expv!\n";
}

void test_ensemble_else() {
    using namespace markovkit::else_solver;

    num::Matrix R(2, 2, 0.0);
    R(0, 0) = -2.0; R(0, 1) = 1.0;
    R(1, 0) = 1.0;  R(1, 1) = -2.0;

    Subnetwork sub(R);

    // Simulate 100 concurrent stochastic trajectories sharing the subnetwork
    auto res = ensemble_else_simulate(100, sub, 1.0, 42);

    assert(res.n_trajectories == 100);
    assert(res.final_states.size() == 100);
    assert(res.exit_times.size() == 100);

    std::cout << "[PASS] Ensemble ELSE (100 parallel trajectories sharing subnetwork)\n";
}

int main() {
    std::cout << "--- Running Unified ELSE Tests ---\n";
    test_subnetwork_fundamental_matrix();
    test_schur_rank1_shedding();
    test_talbot_fourier_resolvent();
    test_ensemble_else();
    std::cout << "All Unified ELSE tests passed successfully!\n";
    return 0;
}
