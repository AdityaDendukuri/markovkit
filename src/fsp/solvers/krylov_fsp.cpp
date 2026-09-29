#include "fsp/solvers/krylov_fsp.hpp"
#include "cme/generator.hpp"
#include "cme/state_space.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fsp {

std::tuple<fsp_solution, krylov_fsp_diagnostics> solve_krylov_fsp(const fsp_problem &prob,
                                                                  const krylov_fsp_params &params) {
    const markovkit::reaction_system &model = prob.model;
    const num::array<double> &rates = prob.rates;
    const cme::boundary_condition &bc = prob.bc;
    double t0 = prob.t0;
    double tf = prob.tf;
    int n_species = static_cast<int>(prob.u0.size());

    cme::state_space sp;
    sp.add_state(prob.u0, 1.0);

    // Save the initial point mass as the first snapshot.
    fsp_solution sol;
    sol.n_species = n_species;
    sol.times.push_back(t0);
    sol.snapshots.push_back({sp.states, num::vec(sp.probs)});
    sol.state_space_sizes.push_back(sp.size());

    krylov_fsp_diagnostics diag;
    diag.reject_count = 0;
    diag.total_iters = 0;

    double t = t0;
    double tau = params.tau_init;
    double droptol = params.droptol;
    int accepted_steps = 0;

    while (t < tf && diag.total_iters < params.max_iter) {
        ++diag.total_iters;

        tau = std::min(tau, tf - t);

        // Expand before propagation so newly reachable probability has support.
        sp.expand(model, bc, params.r);

        // The absorbing diagonal turns missing probability into an error estimate.
        auto assembled = cme::build_generator(sp, model, rates, t, true);
        auto &A = assembled.matrix;
        int n = sp.size();

        // Propagate on the current finite state space.
        num::vec p_in(sp.probs);

        num::vec w = num::expv(tau, A, p_in, params.krylov_m, params.krylov_tol);

        // Accept only if retained mass satisfies the Vo--Sidje error budget.
        double mass = 0.0;
        for (const double value : w) {
            mass += std::max(0.0, value);
        }

        double required_mass = 1.0 - (params.eps * ((t + tau - t0) / (tf - t0)));

        if (mass < required_mass && tau > 1e-14) {
            // Retry with a shorter interval when too much mass escapes.
            tau *= 0.5;
            ++diag.reject_count;
            continue;
        }

        // Commit the accepted probability vector.
        num::vec nonnegative = w;
        for (double &value : nonnegative) {
            value = std::max(0.0, value);
        }
        sp.probs.resize(n);
        num::copy_to(nonnegative, sp.probs);

        // Remove a state only when both its probability and local derivative are small.
        num::vec dp_dt(static_cast<num::idx>(n), 0.0);
        num::operators::sparse_op(A).apply(w, dp_dt);

        num::array<bool> remove_mask(n, false);
        for (int i = 0; i < n; ++i) {
            if (sp.probs[i] < droptol &&
                std::abs(dp_dt[static_cast<num::idx>(i)]) < params.droptol_deriv) {
                remove_mask[i] = true;
            }
        }
        sp.remove_states(remove_mask);
        sp.renormalize();

        t += tau;
        ++accepted_steps;

        diag.tau_log.push_back(tau);
        diag.mass_log.push_back(mass);
        diag.time_log.push_back(t);
        diag.size_log.push_back(sp.size());

        if (params.log_interval > 0 && accepted_steps % params.log_interval == 0) {
            if (params.progress_cb) {
                params.progress_cb(accepted_steps, t, tau, sp.size(), mass);
            }
        }

        if (accepted_steps % params.save_interval == 0 || t >= tf) {
            // Save only configured checkpoints and the final state.
            sol.times.push_back(t);
            sol.snapshots.push_back({sp.states, num::vec(sp.probs)});
            sol.state_space_sizes.push_back(sp.size());
        }

        // Grow the next trial interval after a successful step.
        tau = std::min(tau * params.growth_factor, tf - t);
    }

    if (sol.times.back() < t - (1e-15 * std::abs(t))) {
        sol.times.push_back(t);
        sol.snapshots.push_back({sp.states, num::vec(sp.probs)});
        sol.state_space_sizes.push_back(sp.size());
    }

    return {std::move(sol), std::move(diag)};
}

} // namespace fsp
