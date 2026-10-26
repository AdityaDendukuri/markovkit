#include "plot/plot.hpp"
#include "core/types.hpp"
#include "markovkit.hpp"

int main() {
    // Mutual cubic repression creates two preferred expression states.
    const double alpha = 20.0;
    const double beta = 400.0;
    const double K = 100.0;
    const double d_u = 1.0 + (0.1 / 1.1);
    const double d_v = 1.0;
    const double K3 = K * K * K;

    const num::array<double> rates = {alpha, beta, K3, d_u, alpha, beta, K3, d_v};

    markovkit::reaction_system model;
    model.changes = {
        {1, 0},  // R1: 0 -> U
        {-1, 0}, // R2: U -> 0
        {0, 1},  // R3: 0 -> V
        {0, -1}, // R4: V -> 0
    };
    model.propensities = {
        [](const markovkit::state &x, const num::array<double> &r, double) {
            double v3 = static_cast<double>(x[1]) * x[1] * x[1];
            return r[0] + (r[1] * r[2] / (r[2] + v3));
        },
        [](const markovkit::state &x, const num::array<double> &r, double) { return r[3] * x[0]; },
        [](const markovkit::state &x, const num::array<double> &r, double) {
            double u3 = static_cast<double>(x[0]) * x[0] * x[0];
            return r[4] + (r[5] * r[6] / (r[6] + u3));
        },
        [](const markovkit::state &x, const num::array<double> &r, double) { return r[7] * x[1]; },
    };

    // The distribution starts at the symmetric expression state.
    fsp::fsp_problem prob;
    prob.model = model;
    prob.u0 = {100, 100};
    prob.t0 = 0.0;
    prob.tf = 10.0;
    prob.rates = rates;
    prob.bc = fsp::rect_boundary(500);

    // Absolute pruning keeps the adaptive support compact.
    fsp::adaptive_fsp_params params;
    params.eps_dt = 10.0;
    params.prob_quantile = 0.0;
    params.flux_tolerance = 0.0;
    params.abs_threshold = 1e-6;
    params.expansion_depth = 5;
    params.save_interval = 200;

    // Track the transient distribution on an adaptive state space.
    const auto solution = std::get<0>(fsp::solve_adaptive_fsp(prob, params));
    const auto means = solution.mean_trajectory();

    num::array<double> u, v, sizes;
    for (std::size_t i = 0; i < solution.times.size(); ++i) {
        u.push_back(means(i, 0));
        v.push_back(means(i, 1));
        sizes.push_back(solution.state_space_sizes[i]);
    }

    const int last = solution.n_snapshots() - 1;
    // Reduce the final joint distribution to U and V marginals.
    const auto marginal_u = solution.marginal(last, 0);
    const auto marginal_v = solution.marginal(last, 1);
    num::array<double> counts_u, probs_u, counts_v, probs_v;
    for (int c = 0; c <= 500; ++c) {
        const double pu = marginal_u.contains(c) ? marginal_u.at(c) : 0.0;
        const double pv = marginal_v.contains(c) ? marginal_v.at(c) : 0.0;
        if (pu > 1e-8) {
            counts_u.push_back(c);
            probs_u.push_back(pu);
        }
        if (pv > 1e-8) {
            counts_v.push_back(c);
            probs_v.push_back(pv);
        }
    }

    // Plot transient means, support size, and terminal marginals.
    num::plt::subplot(2, 2);
    num::plt::plot(solution.times, u, "U", "lines lw 2");
    num::plt::plot(solution.times, v, "V", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean");
    num::plt::legend();
    num::plt::next();
    num::plt::plot(solution.times, sizes, "states", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("states");
    num::plt::next();
    num::plt::plot(counts_u, probs_u, "U", "boxes");
    num::plt::xlabel("count");
    num::plt::ylabel("probability");
    num::plt::next();
    num::plt::plot(counts_v, probs_v, "V", "boxes");
    num::plt::xlabel("count");
    num::plt::ylabel("probability");
    num::plt::savefig("fsp_toggle_switch.png");
}
