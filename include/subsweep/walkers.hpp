// Subnetwork sweeping with retained walker identities.
#pragma once

#include "stats/selection.hpp"
#include "stochastic/rng.hpp"
#include "subsweep/exit.hpp"
#include "subsweep/laplace.hpp"
#include "subsweep/selection.hpp"
#include "subsweep/sweep.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace subsweep {

template <typename State>
struct trajectory {
    array<real> times;
    array<State> states;
    // with keep_subnetworks: the subnetwork, i^(r) and j^(r) of each sweep r
    array<idx> subnetwork;
    array<idx> i;
    array<idx> j;

    [[nodiscard]] idx sweeps() const {
        if (states.size() < 2) {
            return 0;
        }
        // a trajectory closed at the final time repeats its last state
        bool closed = states.back() == states[states.size() - 2];
        return closed ? states.size() - 2 : states.size() - 1;
    }
};

// the walkers, and with keep_subnetworks the R and global states of each sweep
template <typename State>
struct walker_sweep_result {
    array<trajectory<State>> trajectories;
    array<num::spmat> R;
    array<array<idx>> global_state;
    array<State> states;
};

template <typename State, detail::oracle<State> Rates, typename SelectD, typename ComputeZ>
[[nodiscard]] walker_sweep_result<State>
sweep_walkers(const Rates &rates, const State &initial, idx walkers, real final_time,
              SelectD select_D, ComputeZ compute_Z, idx sweep_limit, unsigned seed,
              bool keep_subnetworks = false) {
    if (walkers == 0 || !(final_time > 0.0)) {
        throw std::invalid_argument("walker subsweep needs walkers and a positive final time");
    }

    walker_sweep_result<State> sequence;
    array<trajectory<State>> &trajectories = sequence.trajectories;
    trajectories.resize(walkers);
    unordered_map<State, idx> global_row;
    array<State> current(walkers, initial);
    array<real> time(walkers, 0.0);
    array<idx> sweep_count(walkers, 0);
    array<num::rng> random;

    for (idx walker = 0; walker < walkers; ++walker) {
        random.emplace_back(seed + static_cast<unsigned>(walker));
        trajectories[walker].times.push_back(0.0);
        trajectories[walker].states.push_back(initial);
    }

    using factor = factor_of<ComputeZ, State>;
    // D and u are empty, and Z is absent, before the first sweep
    subnetwork<State> D;
    std::optional<factor> Z;
    num::vec<real> u;
    while (true) {
        array<idx> active = num::filter(walkers, [&](idx walker) {
            bool before_final_time = time[walker] < final_time;
            bool below_sweep_limit = sweep_count[walker] < sweep_limit;
            return before_final_time && below_sweep_limit;
        });
        if (active.empty()) {
            break;
        }

        // the distinct current states i_1, ..., i_d, weighted by their walker counts
        auto [entrance, entrance_of] = num::group_by(
            active.size(), [&](idx a) -> const State & { return current[active[a]]; });
        array<std::pair<State, real>> entrances;
        for (idx k = 0; k < entrance.size(); ++k) {
            auto walkers_at_k = std::count(entrance_of.begin(), entrance_of.end(), k);
            entrances.push_back({entrance[k], static_cast<real>(walkers_at_k) / active.size()});
        }

        const subnetwork<State> *previous_D = D.size() ? &D : nullptr;
        const num::vec<real> *previous_u = u.size() ? &u : nullptr;
        update_subnetwork(D, select_D(entrances, previous_D, previous_u), rates);
        update_factor(Z, D, compute_Z);
        if (keep_subnetworks) {
            sequence.R.push_back(D.matrix());
            array<idx> &global = sequence.global_state.emplace_back();
            for (idx row = 0; row < D.size(); ++row) {
                auto [entry, added] = global_row.emplace(D.state(row), sequence.states.size());
                if (added) {
                    sequence.states.push_back(D.state(row));
                }
                global.push_back(entry->second);
            }
        }

        // R^T U = E and R^T V = U give U(j, k) = Z_{i_k j} and V(j, k) = (Z^2)_{i_k j}
        num::mat<real> E(D.size(), entrance.size(), 0.0);
        for (idx k = 0; k < entrance.size(); ++k) {
            auto i = D.find(entrance[k]);
            if (!i) {
                throw std::invalid_argument("each entrance must belong to the subnetwork");
            }
            E(*i, k) = 1.0;
        }
        num::mat<real> U = num::solve(num::transpose(*Z), E);
        num::mat<real> V = num::solve(num::transpose(*Z), U);

        // u = Z^T rho, the occupation times of the entrances
        u = num::vec<real>(D.size(), 0.0);
        for (idx k = 0; k < entrance.size(); ++k) {
            for (idx j = 0; j < D.size(); ++j) {
                u[j] += entrances[k].second * U(j, k);
            }
        }

        // Pr(X_{tau-} = j | X_0 = i_k) = Z_{i_k j} w_j, with w = R 1
        num::vec<real> w = D.exit_rates();
        array<num::vec<real>> last_interior_probabilities;
        for (idx k = 0; k < entrance.size(); ++k) {
            num::vec<real> Z_row(D.size(), 0.0);
            for (idx j = 0; j < D.size(); ++j) {
                Z_row[j] = U(j, k);
            }
            last_interior_probabilities.push_back(last_interior_law(Z_row, w));
        }

        for (idx a = 0; a < active.size(); ++a) {
            // Sample the last interior state and add its conditional expected sweep duration.
            idx walker = active[a];
            idx k = entrance_of[a];
            idx j = sample_last_interior_state(last_interior_probabilities[k], random[walker]);
            if (keep_subnetworks) {
                trajectories[walker].subnetwork.push_back(sequence.R.size() - 1);
                trajectories[walker].i.push_back(*D.find(entrance[k]));
                trajectories[walker].j.push_back(j);
            }
            real expected_duration = V(j, k) / U(j, k);
            if (!(expected_duration > 0.0) || time[walker] + expected_duration >= final_time) {
                time[walker] = final_time;
                continue;
            }

            time[walker] += expected_duration;
            current[walker] = sample_first_exterior_state(rates, D, j, random[walker]);
            ++sweep_count[walker];
            trajectories[walker].times.push_back(time[walker]);
            trajectories[walker].states.push_back(current[walker]);
        }
    }

    // close each trajectory that stopped before the final time without reaching the limit
    for (idx walker = 0; walker < walkers; ++walker) {
        if (sweep_count[walker] < sweep_limit && trajectories[walker].times.back() < final_time) {
            trajectories[walker].times.push_back(final_time);
            trajectories[walker].states.push_back(trajectories[walker].states.back());
        }
    }
    return sequence;
}

// Retain `walkers` individual state sequences from `initial` until `final_time`.
template <typename State, detail::oracle<State> Rates>
[[nodiscard]] walker_sweep_result<State> subsweep_walkers(const Rates &rates, const State &initial,
                                                          idx walkers, real final_time,
                                                          const sweep_settings &settings = {}) {
    return with_regime<State, Rates>(settings.regime, rates, [&](auto compute_Z) {
        return sweep_walkers(rates, initial, walkers, final_time,
                             capacity_selection<State>(rates, settings), compute_Z,
                             settings.sweep_limit, settings.seed, settings.keep_subnetworks);
    });
}

// the density at the contour `nodes` along the sampled exits: with R z = e_j,
// d = z_i and q^T (sI + R) = e_i^T, G_r = q_j / d and F_r(x) = q_x z_x / d, and
// each walker gives sum_r (prod_{m<r} G_m) F_r
template <typename State>
[[nodiscard]] distribution<State> reconstruct_density(const walker_sweep_result<State> &sequence,
                                                      view<const num::contour_node> nodes,
                                                      bool normalize = true) {
    if (sequence.R.empty() || nodes.empty()) {
        throw std::invalid_argument(
            "walker reconstruction needs retained subnetworks and contour nodes");
    }
    const array<trajectory<State>> &walkers = sequence.trajectories;
    // the (walker, r) visits of each subnetwork, in the order they happened
    array<array<std::pair<idx, idx>>> visits(sequence.R.size());
    for (idx walker = 0; walker < walkers.size(); ++walker) {
        for (idx r = 0; r < walkers[walker].subnetwork.size(); ++r) {
            visits[walkers[walker].subnetwork[r]].push_back({walker, r});
        }
    }
    // g = prod_{m<r} G_m(s) for each walker at each node
    array<array<num::cplx>> g(walkers.size(), array<num::cplx>(nodes.size(), 1.0));
    distribution<State> result{sequence.states, num::vec<real>(sequence.states.size(), 0.0)};

    for (idx k = 0; k < sequence.R.size(); ++k) {
        if (visits[k].empty()) {
            continue;
        }
        const num::spmat &R = sequence.R[k];
        const idx n = R.n_rows();
        // z = R^{-1} e_j and d = z_i for each visit
        auto Z = num::lu(R, num::sparse);
        array<num::vec<real>> z;
        array<real> d;
        for (auto [walker, r] : visits[k]) {
            num::vec<real> e_j(n, 0.0);
            e_j[walkers[walker].j[r]] = 1.0;
            z.push_back(num::solve(Z, e_j));
            d.push_back(z.back()[walkers[walker].i[r]]);
        }
        // q = (sI + R^T)^{-1} e_i, the resolvent of -R^T at each node
        num::auto_resolvent resolvent(num::scaled(num::transpose(R), -1.0));
        for (idx node = 0; node < nodes.size(); ++node) {
            const auto &[s, weight] = nodes[node];
            const auto F = num::shift(resolvent, s);
            for (idx v = 0; v < visits[k].size(); ++v) {
                auto [walker, r] = visits[k][v];
                num::vec<num::cplx> e_i(n, 0.0), q;
                e_i[walkers[walker].i[r]] = 1.0;
                num::solve(F, e_i, q);
                num::cplx before = g[walker][node];
                for (idx x = 0; x < n; ++x) {
                    num::cplx F = q[x] * z[v][x] / d[v];
                    result.probability[sequence.global_state[k][x]] +=
                        (weight * before * F).real() / static_cast<real>(walkers.size());
                }
                g[walker][node] = before * q[walkers[walker].j[r]] / d[v];
            }
        }
    }
    clip_density(result.probability, normalize);
    return result;
}

} // namespace subsweep
