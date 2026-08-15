#include <markovkit.hpp>
#include <plot/plot.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "=== MarkovKit: Genetic Toggle Switch FSP & Ensemble ELSE Simulation ===\n\n";

    using namespace fsp;
    using namespace markovkit::else_solver;

    // 1. Define Tian-Burrage Genetic Toggle Switch Reaction System:
    ReactionSystem sys;
    sys.stoich_vecs = {
        MultiIndex{1, 0},
        MultiIndex{-1, 0},
        MultiIndex{0, 1},
        MultiIndex{0, -1}
    };

    double eta = 1.0;
    double alpha1 = 20.0, beta1 = 400.0, K1 = 100.0, d1 = 1.0;
    double s = 0.1, gamma = 1.0;
    double alpha2 = 20.0, beta2 = 400.0, K2 = 100.0, d2 = 1.0;

    double scale = 0.05; // System scale
    std::vector<double> rates = {
        eta,
        alpha1 * scale,
        beta1 * scale,
        K1 * scale,
        d1,
        s * gamma / (1.0 + s),
        alpha2 * scale,
        beta2 * scale,
        K2 * scale,
        d2
    };

    sys.propensities = {
        [](const MultiIndex& x, const std::vector<double>& r, double t) {
            double v = static_cast<double>(x[1]);
            double k3 = r[3] * r[3] * r[3];
            return r[0] * (r[1] + r[2] * k3 / (k3 + v * v * v));
        },
        [](const MultiIndex& x, const std::vector<double>& r, double t) {
            return (r[4] + r[5]) * static_cast<double>(x[0]);
        },
        [](const MultiIndex& x, const std::vector<double>& r, double t) {
            double u = static_cast<double>(x[0]);
            double k8 = r[8] * r[8] * r[8];
            return r[0] * (r[6] + r[7] * k8 / (k8 + u * u * u));
        },
        [](const MultiIndex& x, const std::vector<double>& r, double t) {
            return r[9] * static_cast<double>(x[1]);
        }
    };

    // 2. Set Up Adaptive FSP Simulation
    FSPProblem prob;
    prob.model = sys;
    prob.rates = rates;
    prob.u0 = MultiIndex{20, 2}; // Initial state: U high, V low
    prob.t0 = 0.0;
    prob.tf = 30.0;
    prob.bc = rect_boundary({80, 80});

    AdaptiveFSPParams params;
    params.eps_dt = 0.2;
    params.expansion_depth = 2;
    params.krylov_m = 30;
    params.krylov_tol = 1e-8;
    params.max_iter = 500;

    std::cout << "1. Solving Toggle Switch FSP from t=0 to t=" << prob.tf << "...\n";
    auto [sol, diag] = solve_adaptive_fsp(prob, params);
    std::cout << "   Adaptive FSP completed in " << sol.t.size() << " steps.\n";

    // 3. Extract time series
    std::vector<double> time_points = sol.t;
    std::vector<double> state_sizes;
    for (int sz : sol.state_space_sizes) state_sizes.push_back(static_cast<double>(sz));

    std::vector<double> mean_U, mean_V;
    for (const auto& snap : sol.snapshots) {
        double mu = 0.0, mv = 0.0;
        for (std::size_t i = 0; i < snap.states.size(); ++i) {
            mu += snap.states[i][0] * snap.probs[i];
            mv += snap.states[i][1] * snap.probs[i];
        }
        mean_U.push_back(mu);
        mean_V.push_back(mv);
    }

    // 4. Run Ensemble ELSE simulation
    std::cout << "2. Running Ensemble ELSE simulation over bistable subnetwork...\n";
    num::Matrix R_sub(2, 2, 0.0);
    R_sub(0, 0) = -1.5; R_sub(0, 1) = 0.5;
    R_sub(1, 0) = 0.5;  R_sub(1, 1) = -1.5;

    Subnetwork sub(R_sub);
    auto ens = ensemble_else_simulate(100, sub, 10.0, 42);
    std::cout << "   Ensemble ELSE simulated " << ens.n_trajectories << " parallel stochastic paths.\n";

    std::vector<double> traj_indices(ens.n_trajectories);
    for (int i = 0; i < ens.n_trajectories; ++i) traj_indices[i] = static_cast<double>(i + 1);

    // 5. Generate 4-panel summary plot using num::plt (gnuplot wrapper)
    num::plt::subplot(2, 2);

    // Panel (a): State space size |S|(t)
    num::plt::plot(time_points, state_sizes, "|S|", "lines lw 2");
    num::plt::title("(a) FSP Active State Space Size |S|");
    num::plt::xlabel("Time");
    num::plt::ylabel("|S|");
    num::plt::next();

    // Panel (b): Mean U and V trajectories
    num::plt::plot(time_points, mean_U, "<U>", "lines lw 2");
    num::plt::title("(b) Mean U Population");
    num::plt::xlabel("Time");
    num::plt::ylabel("Mean Count");
    num::plt::next();

    // Panel (c): U-V Phase Space Trajectory
    num::plt::plot(mean_U, mean_V, "Phase Trajectory", "lines lw 2");
    num::plt::title("(c) U-V Phase Space Trajectory");
    num::plt::xlabel("<U>");
    num::plt::ylabel("<V>");
    num::plt::next();

    // Panel (d): Ensemble ELSE Exit Times
    num::plt::plot(traj_indices, ens.exit_times, "Exit Times", "points pt 7 ps 0.8");
    num::plt::title("(d) Ensemble ELSE Exit Times");
    num::plt::xlabel("Trajectory Index");
    num::plt::ylabel("Exit Time");

    std::cout << "Saving Toggle Switch simulation plots to 'toggle_switch_results.png'...\n";
    num::plt::savefig("toggle_switch_results.png");

    std::cout << "Toggle Switch FSP & Ensemble ELSE simulation completed successfully!\n";
    return 0;
}
