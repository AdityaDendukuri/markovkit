#include "markovkit.hpp"
#include <vector>

int main() {
    // These scales reproduce the stochastic Oregonator regime.
    const double y1s = 500.0, y2s = 1000.0, y3s = 2000.0;
    const double mu1s = 2000.0, mu2s = 50000.0;

    const std::vector<double> rates = {
        mu1s / y2s,               // k1 = 2.0
        mu2s / (y1s * y2s),       // k2 = 0.1
        (mu1s + mu2s) / y1s,      // k3 = 104.0
        2.0 * mu1s / (y1s * y1s), // k4 = 0.016
        (mu1s + mu2s) / y3s,      // k5 = 26.0
    };

    // Five reactions couple the three oscillator species.
    markovkit::ReactionSystem model;
    model.changes = {
        {1, -1, 0},  // R1
        {-1, -1, 0}, // R2
        {1, 0, 1},   // R3
        {-2, 0, 0},  // R4
        {0, 1, -1},  // R5
    };
    model.propensities = {
        [](const markovkit::State &x, const std::vector<double> &r, double) { return r[0] * x[1]; },
        [](const markovkit::State &x, const std::vector<double> &r, double) {
            return r[1] * x[0] * x[1];
        },
        [](const markovkit::State &x, const std::vector<double> &r, double) { return r[2] * x[0]; },
        [](const markovkit::State &x, const std::vector<double> &r, double) {
            return x[0] < 2 ? 0.0 : r[3] * x[0] * (x[0] - 1) * 0.5;
        },
        [](const markovkit::State &x, const std::vector<double> &r, double) { return r[4] * x[2]; },
    };

    // The FSP starts from the deterministic reference counts.
    fsp::FSPProblem prob;
    prob.model = model;
    prob.u0 = {500, 1000, 2000};
    prob.t0 = 0.0;
    prob.tf = 2.0;
    prob.rates = rates;
    prob.bc = fsp::rect_boundary(50000);

    // SSA-guided expansion limits growth of the active state space.
    fsp::AdaptiveFSPParams params;
    params.eps_dt = 1.0;
    params.prob_quantile = 0.1;
    params.flux_tolerance = 1.0;
    params.expansion_depth = 1;
    params.expand_method = fsp::ExpandMethod::ssa;
    params.save_interval = 1000;

    // Evolve an adaptively sampled finite state space.
    const auto solution = std::get<0>(fsp::solve_adaptive_fsp(prob, params));
    const auto means = solution.mean_trajectory();

    std::vector<double> x, y, z, sizes;
    // Extract observable curves from saved snapshots.
    for (std::size_t i = 0; i < solution.times.size(); ++i) {
        x.push_back(means[i][0]);
        y.push_back(means[i][1]);
        z.push_back(means[i][2]);
        sizes.push_back(solution.state_space_sizes[i]);
    }

    // Plot the three means alongside active state-space size.
    num::plt::subplot(2, 2);
    num::plt::plot(solution.times, x, "X", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean X");
    num::plt::next();
    num::plt::plot(solution.times, y, "Y", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean Y");
    num::plt::next();
    num::plt::plot(solution.times, z, "Z", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("mean Z");
    num::plt::next();
    num::plt::plot(solution.times, sizes, "states", "lines lw 2");
    num::plt::xlabel("time");
    num::plt::ylabel("states");
    num::plt::savefig("fsp_oregonator.png");
}
