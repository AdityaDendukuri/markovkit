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

void test_shedding_criteria() {
    using namespace markovkit::else_solver;

    num::Matrix R(3, 3, 0.0);
    R(0, 0) = -3.0; R(0, 1) = 1.0; R(0, 2) = 1.0;
    R(1, 0) = 1.0;  R(1, 1) = -2.0; R(1, 2) = 0.0;
    R(2, 0) = 1.0;  R(2, 1) = 0.0;  R(2, 2) = -2.0;

    Subnetwork sub(R);
    num::Vector entrance{1.0, 0.0, 0.0};

    auto schur_loss = sub.schur_losses(entrance);
    auto time_loss  = sub.cut_time_losses(entrance);
    auto norm_loss  = sub.cut_norm_losses(entrance);

    assert(schur_loss.size() == 3);
    assert(time_loss.size() == 3);
    assert(norm_loss.size() == 3);

    sub.shed_state(2);
    assert(sub.R.rows() == 2);

    std::cout << "[PASS] ELSE Shedding Rules (Schur, CutTime, CutNorm) and Schur rank-1 shedding\n";
}

void test_fess_network() {
    using namespace markovkit::else_solver;
    using namespace fsp;

    ReactionSystem sys;
    sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
    sys.propensities = {
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[0]; },
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[1] * x[0]; }
    };
    std::vector<double> rates = {5.0, 0.1};

    FESSNetwork net(MultiIndex{0});
    net.augment({MultiIndex{0}}, sys, rates, 3);

    assert(net.size() >= 3);
    std::cout << "[PASS] FESSNetwork dynamic state space expansion (" << net.size() << " states)\n";
}

void test_fess_density_evolution() {
    using namespace markovkit::else_solver;
    using namespace fsp;

    ReactionSystem sys;
    sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
    sys.propensities = {
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[0]; },
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[1] * x[0]; }
    };
    std::vector<double> rates = {5.0, 0.1};

    fsp::FlatHashMap<MultiIndex, double> p0;
    p0[MultiIndex{0}] = 1.0;

    auto evol = fess_density_evolution(p0, sys, rates, 3);
    assert(evol.densities.size() == 4);

    std::cout << "[PASS] FESSDensityEvolution continuous density propagation\n";
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
    auto res = ensemble_else_simulate(100, sub, 1.0, 42);

    assert(res.n_trajectories == 100);
    assert(res.final_states.size() == 100);

    std::cout << "[PASS] Ensemble ELSE (100 parallel trajectories sharing subnetwork)\n";
}

int main() {
    std::cout << "--- Running Complete FESS/ELSE Suite Tests ---\n";
    test_subnetwork_fundamental_matrix();
    test_shedding_criteria();
    test_fess_network();
    test_fess_density_evolution();
    test_talbot_fourier_resolvent();
    test_ensemble_else();
    std::cout << "All FESS/ELSE suite tests passed successfully!\n";
    return 0;
}
