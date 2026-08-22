#include "markovkit.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <numerics.hpp>

namespace {

num::Matrix generate_random_markov_generator(num::idx n, double edge_prob, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> weight_dist(0.5, 2.0);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    num::Matrix Q(n, n, 0.0);

    // Ensure connectivity via a random spanning tree
    std::vector<num::idx> connected = {0};
    std::vector<num::idx> remaining;
    for (num::idx i = 1; i < n; ++i) remaining.push_back(i);

    while (!remaining.empty()) {
        std::uniform_int_distribution<std::size_t> conn_dist(0, connected.size() - 1);
        std::uniform_int_distribution<std::size_t> rem_dist(0, remaining.size() - 1);

        num::idx u = connected[conn_dist(rng)];
        std::size_t v_idx = rem_dist(rng);
        num::idx v = remaining[v_idx];

        double w1 = weight_dist(rng);
        double w2 = weight_dist(rng);
        Q(u, v) = w1;
        Q(v, u) = w2;

        connected.push_back(v);
        remaining.erase(remaining.begin() + static_cast<long>(v_idx));
    }

    // Add random off-diagonal jump edges
    for (num::idx i = 0; i < n; ++i) {
        for (num::idx j = i + 1; j < n; ++j) {
            if (Q(i, j) == 0.0 && prob_dist(rng) < edge_prob) {
                Q(i, j) = weight_dist(rng);
                Q(j, i) = weight_dist(rng);
            }
        }
    }

    // Conservative column sums: dp/dt = Q*p
    for (num::idx j = 0; j < n; ++j) {
        double col_sum = 0.0;
        for (num::idx i = 0; i < n; ++i) {
            if (i != j) col_sum += Q(i, j);
        }
        Q(j, j) = -col_sum;
    }

    return Q;
}

num::Vector compute_exact_krylov_fsp(const num::Matrix &Q, const num::Vector &p0, double t) {
    num::operators::DenseOp Q_op(Q);
    return num::expv(t, Q_op, p0, 50, 1e-15);
}

num::Vector compute_talbot_density(const num::HessenbergResolventSolver &solver,
                                   const num::Vector &p0, double t, num::idx nodes) {
    const num::idx n = solver.size();
    std::vector<num::cplx> density(n, num::cplx(0.0, 0.0));

    num::inverse_laplace_accumulate(t, nodes, [&](num::cplx shift, num::cplx weight) {
        auto sol = solver.solve(shift, p0);
        for (num::idx i = 0; i < n; ++i) {
            density[i] += weight * sol[i];
        }
    });

    num::Vector p(n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        p[i] = std::max(0.0, density[i].real());
    }
    num::clip_and_normalize_nonnegative(p);
    return p;
}

std::pair<double, double> state_errors(const num::Vector &p, const num::Vector &p_ref) {
    double l_inf = 0.0;
    double l_1 = 0.0;
    for (num::idx i = 0; i < p.size(); ++i) {
        double diff = std::abs(p[i] - p_ref[i]);
        l_inf = std::max(l_inf, diff);
        l_1 += diff;
    }
    return {l_inf, l_1};
}

} // namespace

int main() {
    using namespace num;

    const idx N = 120;
    std::mt19937_64 rng(12345);

    Matrix Q = generate_random_markov_generator(N, 0.08, rng);
    Vector p0 = unit_vector(N, 0);

    HessenbergResolventSolver hess_solver(Q);

    const double t_target = 1.0;
    Vector p_exact = compute_exact_krylov_fsp(Q, p0, t_target);

    const std::vector<idx> node_counts = {4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    std::vector<double> node_list_dbl;
    std::vector<double> err_inf_list;
    std::vector<double> err_l1_list;

    for (idx M : node_counts) {
        Vector p_talbot = compute_talbot_density(hess_solver, p0, t_target, M);
        auto [l_inf, l_1] = state_errors(p_talbot, p_exact);

        node_list_dbl.push_back(static_cast<double>(M));
        err_inf_list.push_back(std::max(1e-16, l_inf));
        err_l1_list.push_back(std::max(1e-16, l_1));
    }

    const auto time_grid = logspace(-2.0, 1.0, 25);
    std::vector<double> err_t_M8, err_t_M14, err_t_M24;

    for (double t : time_grid) {
        Vector p_ex = compute_exact_krylov_fsp(Q, p0, t);
        Vector p_m8 = compute_talbot_density(hess_solver, p0, t, 8);
        Vector p_m14 = compute_talbot_density(hess_solver, p0, t, 14);
        Vector p_m24 = compute_talbot_density(hess_solver, p0, t, 24);

        err_t_M8.push_back(std::max(1e-16, state_errors(p_m8, p_ex).first));
        err_t_M14.push_back(std::max(1e-16, state_errors(p_m14, p_ex).first));
        err_t_M24.push_back(std::max(1e-16, state_errors(p_m24, p_ex).first));
    }

    plt::subplot(1, 3);

    // Panel 1: Error vs Talbot nodes M
    plt::plot(node_list_dbl, err_inf_list, "L_inf Error", "linespoints pt 7 lw 2 lc rgb '#1f77b4'");
    plt::plot(node_list_dbl, err_l1_list, "L_1 Error", "linespoints pt 5 lw 2 lc rgb '#ff7f0e'");
    plt::title("Talbot Convergence vs Nodes M (t = 1.0)");
    plt::xlabel("Nodes (M)");
    plt::ylabel("Error vs Krylov-FSP");
    plt::semilogy();
    plt::legend();
    plt::next();

    // Panel 2: Error vs Propagation time
    plt::plot(time_grid, err_t_M8, "M = 8", "lines lw 2 lc rgb '#d62728'");
    plt::plot(time_grid, err_t_M14, "M = 14", "lines lw 2 lc rgb '#2ca02c'");
    plt::plot(time_grid, err_t_M24, "M = 24", "lines lw 2 lc rgb '#9467bd'");
    plt::title("Error vs Time t in [0.01, 10.0]");
    plt::xlabel("Time t");
    plt::ylabel("L_inf Error");
    plt::loglog();
    plt::legend();
    plt::next();

    // Panel 3: Sorted state probability density profile
    Vector p_m14_final = compute_talbot_density(hess_solver, p0, t_target, 14);

    std::vector<idx> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](idx a, idx b) {
        return p_exact[a] > p_exact[b];
    });

    std::vector<double> rank_indices(N);
    std::vector<double> exact_sorted(N);
    std::vector<double> talbot_sorted(N);
    for (idx i = 0; i < N; ++i) {
        rank_indices[i] = static_cast<double>(i + 1);
        exact_sorted[i] = p_exact[order[i]];
        talbot_sorted[i] = p_m14_final[order[i]];
    }

    plt::plot(rank_indices, exact_sorted, "Exact FSP", "lines lw 3 lc rgb '#000000'");
    plt::plot(rank_indices, talbot_sorted, "Talbot (M=14)", "points pt 6 ps 0.8 lc rgb '#1f77b4'");
    plt::title("Rank-Ordered Density p(t=1.0)");
    plt::xlabel("State Rank");
    plt::ylabel("Probability p_{(i)}(t)");
    plt::semilogy();
    plt::legend();

    plt::savefig("talbot_vs_fsp_laplacian.png");

    return 0;
}
