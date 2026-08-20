#include "else/else_solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace else_sim {

namespace {

std::size_t sample_discrete(std::mt19937& rng, const std::vector<double>& weights) {
  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
  return dist(rng);
}

void prune_subnetwork(fsp::StateSpace& sp,
                      const Subnetwork& sub,
                      const num::Vector& start,
                      const fsp::FlatHashMap<MultiIndex, int>& protected_states,
                      num::idx capacity) {
  if (sp.size() <= static_cast<int>(capacity)) return;

  const auto u = sub.occupation(start);
  const auto visits = sub.entry_counts(u, start);

  std::vector<std::pair<double, int>> ranked;
  for (int i = 0; i < sp.size(); ++i) {
    if (!protected_states.count(sp.states[i])) {
      ranked.push_back({visits[static_cast<num::idx>(i)], i});
    }
  }
  std::sort(ranked.begin(), ranked.end());

  int excess = sp.size() - static_cast<int>(capacity);
  std::vector<bool> remove(sp.size(), false);
  for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(excess), ranked.size()); ++i) {
    remove[ranked[i].second] = true;
  }
  sp.remove_states(remove);
}

} // namespace

ELSEStepResult else_step(fsp::StateSpace& workspace,
                         const MultiIndex& state,
                         const fsp::ReactionSystem& model,
                         const std::vector<double>& rates,
                         const ELSEOptions& options,
                         std::mt19937& rng) {
  workspace.add_state(state, 1.0);
  workspace.expand(model, fsp::rect_boundary(1000000), options.expansion_depth);

  if (workspace.size() > static_cast<int>(options.capacity)) {
    Subnetwork sub_temp(workspace, model, rates);
    num::Vector start_vec(workspace.size(), 0.0);
    start_vec[workspace.find(state)] = 1.0;
    fsp::FlatHashMap<MultiIndex, int> prot;
    prot[state] = 1;
    prune_subnetwork(workspace, sub_temp, start_vec, prot, options.capacity);
  }

  Subnetwork sub(workspace, model, rates);
  if (sub.boundary_states().empty()) {
    return {state, state, 0.0, 0.0, true};
  }

  num::Matrix starts(sub.size(), 1, 0.0);
  starts(sub.find(state), 0) = 1.0;
  const auto moments = sub.moments(starts);

  const auto p_exit = sub.escape_probabilities(moments, 0);
  if (p_exit.empty()) return {state, state, 0.0, 0.0, true};

  std::size_t chosen_boundary_idx = sample_discrete(rng, p_exit);
  num::idx exit_state_idx = sub.boundary_states()[chosen_boundary_idx];
  MultiIndex preexit = sub.states()[exit_state_idx];

  double waited = sub.conditional_mean_time(moments, exit_state_idx, 0);
  double mean_time = 0.0;
  for (num::idx i = 0; i < moments.size(); ++i) mean_time += moments.first(i, 0);

  // Find outward transitions from preexit
  std::vector<MultiIndex> targets;
  std::vector<double> outward_rates;
  for (const auto& edge : sub.exit_edges()) {
    if (edge.source == exit_state_idx) {
      targets.push_back(sub.destinations()[edge.destination]);
      outward_rates.push_back(edge.rate);
    }
  }

  if (targets.empty() || !(waited > 0.0)) {
    return {state, state, 0.0, 0.0, true};
  }

  std::size_t dest_chosen = sample_discrete(rng, outward_rates);
  MultiIndex next = targets[dest_chosen];

  return {next, preexit, waited, mean_time, false};
}

Trajectory else_trajectory(const fsp::ReactionSystem& model,
                           const std::vector<double>& rates,
                           const MultiIndex& initial,
                           double t0,
                           double tf,
                           ELSEOptions options,
                           int seed) {
  std::mt19937 rng(seed);
  Trajectory path;
  path.t.push_back(t0);
  path.states.push_back(initial);

  fsp::StateSpace ws;
  MultiIndex current = initial;
  double t = t0;

  while (t < tf) {
    auto res = else_step(ws, current, model, rates, options, rng);
    if (res.terminated || !(res.waiting_time > 0.0)) break;
    t += res.waiting_time;
    if (t > tf) t = tf;
    current = res.next_state;
    path.t.push_back(t);
    path.states.push_back(current);
  }
  return path;
}

EnsembleResult else_ensemble(const fsp::ReactionSystem& model,
                             const std::vector<double>& rates,
                             const MultiIndex& initial,
                             std::size_t num_trajectories,
                             double t0,
                             double tf,
                             ELSEOptions options,
                             int seed) {
  std::vector<double> now(num_trajectories, t0);
  std::vector<MultiIndex> here(num_trajectories, initial);
  std::vector<std::vector<MultiIndex>> paths(num_trajectories);
  std::vector<std::mt19937> rng;

  for (std::size_t i = 0; i < num_trajectories; ++i) {
    rng.emplace_back(seed + static_cast<int>(i));
    paths[i].push_back(initial);
  }

  fsp::StateSpace ws;
  ws.add_state(initial);

  while (true) {
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < num_trajectories; ++i) {
      if (now[i] < tf) active.push_back(i);
    }
    if (active.empty()) break;

    std::vector<MultiIndex> occupied;
    fsp::FlatHashMap<MultiIndex, num::idx> col_of;
    fsp::FlatHashMap<MultiIndex, int> protected_states;

    for (std::size_t i : active) {
      if (!col_of.count(here[i])) {
        col_of[here[i]] = occupied.size();
        occupied.push_back(here[i]);
        protected_states[here[i]] = 1;
      }
    }

    for (const auto& s : occupied) ws.add_state(s);
    ws.expand(model, fsp::rect_boundary(1000000), options.expansion_depth);

    if (ws.size() > static_cast<int>(options.capacity)) {
      Subnetwork sub_temp(ws, model, rates);
      num::Vector mixture(ws.size(), 0.0);
      for (std::size_t i : active) {
        int idx = ws.find(here[i]);
        if (idx >= 0) mixture[static_cast<num::idx>(idx)] += 1.0 / active.size();
      }
      prune_subnetwork(ws, sub_temp, mixture, protected_states, options.capacity);
    }

    Subnetwork sub(ws, model, rates);
    if (sub.boundary_states().empty()) break;

    num::Matrix starts(sub.size(), occupied.size(), 0.0);
    for (std::size_t c = 0; c < occupied.size(); ++c) {
      int idx = sub.find(occupied[c]);
      if (idx >= 0) starts(static_cast<num::idx>(idx), c) = 1.0;
    }
    const auto moments = sub.moments(starts);

    std::vector<std::vector<double>> exit_laws;
    exit_laws.reserve(occupied.size());
    for (std::size_t c = 0; c < occupied.size(); ++c) {
      exit_laws.push_back(sub.escape_probabilities(moments, c));
    }

    for (std::size_t i : active) {
      num::idx c = col_of[here[i]];
      const auto& p_exit = exit_laws[c];
      if (p_exit.empty()) {
        now[i] = tf;
        continue;
      }

      std::size_t chosen_b = sample_discrete(rng[i], p_exit);
      num::idx exit_state_idx = sub.boundary_states()[chosen_b];
      double waited = sub.conditional_mean_time(moments, exit_state_idx, c);

      if (!(waited > 0.0) || now[i] + waited >= tf) {
        now[i] = tf;
        continue;
      }

      std::vector<MultiIndex> targets;
      std::vector<double> outward;
      for (const auto& edge : sub.exit_edges()) {
        if (edge.source == exit_state_idx) {
          targets.push_back(sub.destinations()[edge.destination]);
          outward.push_back(edge.rate);
        }
      }
      if (targets.empty()) {
        now[i] = tf;
        continue;
      }

      here[i] = targets[sample_discrete(rng[i], outward)];
      now[i] += waited;
      paths[i].push_back(here[i]);
    }
  }

  EnsembleResult res;
  res.t = now;
  res.paths = std::move(paths);
  return res;
}

void EnsembleResult::save_csv(const std::string& filename) const {
  FILE* fp = std::fopen(filename.c_str(), "w");
  if (!fp) return;
  std::fprintf(fp, "trajectory,time");
  if (!paths.empty() && !paths[0].empty()) {
    for (std::size_t s = 0; s < paths[0][0].size(); ++s) {
      std::fprintf(fp, ",X%zu", s);
    }
  }
  std::fprintf(fp, "\n");

  for (std::size_t i = 0; i < paths.size(); ++i) {
    for (std::size_t k = 0; k < paths[i].size(); ++k) {
      std::fprintf(fp, "%zu,%.6f", i, (k < t.size() ? t[i] : 0.0));
      for (std::size_t s = 0; s < paths[i][k].size(); ++s) {
        std::fprintf(fp, ",%d", paths[i][k][s]);
      }
      std::fprintf(fp, "\n");
    }
  }
  std::fclose(fp);
}

std::vector<DensityStep> else_density(const fsp::ReactionSystem& model,
                                      const std::vector<double>& rates,
                                      const MultiIndex& initial,
                                      int steps,
                                      ELSEOptions options) {
  std::vector<DensityStep> history;
  fsp::StateSpace ws;
  ws.add_state(initial, 1.0);
  std::map<MultiIndex, double> density{{initial, 1.0}};
  double elapsed = 0.0;

  for (int step = 0; step < steps; ++step) {
    double total_mass = 0.0;
    for (const auto& [st, p] : density) total_mass += p;
    if (!(total_mass > options.tolerance)) break;

    std::map<MultiIndex, double> normalized;
    for (const auto& [st, p] : density) normalized[st] = p / total_mass;

    for (const auto& [st, p] : normalized) ws.add_state(st);
    ws.expand(model, fsp::rect_boundary(1000000), options.expansion_depth);

    Subnetwork sub(ws, model, rates);
    num::Vector start_vec(sub.size(), 0.0);
    for (const auto& [st, p] : normalized) {
      int idx = sub.find(st);
      if (idx >= 0) start_vec[static_cast<num::idx>(idx)] = p;
    }

    const auto u = sub.occupation(start_vec);
    const auto flux = sub.exit_flux(u);

    double dt = 0.0;
    for (num::idx i = 0; i < u.size(); ++i) dt += u[i];
    elapsed += dt;

    std::map<MultiIndex, double> next;
    for (std::size_t d = 0; d < flux.size(); ++d) {
      if (flux[d] > options.tolerance) {
        next[sub.destinations()[d]] += flux[d];
      }
    }

    DensityStep ds;
    ds.states = sub.states();
    for (num::idx i = 0; i < sub.size(); ++i) ds.index[sub.states()[i]] = static_cast<int>(i);
    ds.generator = sub.generator();
    ds.exit_rates = sub.exit_rates();
    ds.destinations = sub.destinations();
    ds.edges = sub.exit_edges();
    ds.start = normalized;
    ds.mass = total_mass;
    ds.elapsed = elapsed;
    history.push_back(std::move(ds));

    if (next.empty() || elapsed >= options.horizon) break;
    density = std::move(next);
  }
  return history;
}

} // namespace else_sim
