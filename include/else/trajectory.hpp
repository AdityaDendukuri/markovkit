#pragma once
#include "else/subnetwork.hpp"
#include "markovkit/trajectory.hpp"
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace else_sim {

struct ELSEOptions {
  num::idx capacity = 60;   // Maximum states retained in a subnetwork.
  int expansion_depth = 1;  // Reaction-neighbor shells added before pruning.
  double tolerance = 1e-12; // Boundary mass below this value is discarded.
  std::size_t maximum_steps = std::numeric_limits<std::size_t>::max();
};

/// Simulate one ELSE path; equivalent to an ensemble of size one.
markovkit::Trajectory else_trajectory(const markovkit::ReactionSystem& model,
                                      const std::vector<double>& rates,
                                      const markovkit::State& initial,
                                      double t0,
                                      double tf,
                                      ELSEOptions options = {},
                                      int seed = 42);

/// Simulate paths sharing each subnetwork factorization and entrance solves.
std::vector<markovkit::Trajectory> else_ensemble(const markovkit::ReactionSystem& model,
                                                 const std::vector<double>& rates,
                                                 const markovkit::State& initial,
                                                 std::size_t num_trajectories,
                                                 double t0,
                                                 double tf,
                                                 ELSEOptions options = {},
                                                 int seed = 42);

/// Build the subnetwork chain used by deterministic density inversion.
std::vector<cme::RestrictedGenerator> density_subnetworks(const markovkit::ReactionSystem& model,
                                                          const std::vector<double>& rates,
                                                          const markovkit::State& initial,
                                                          int count,
                                                          ELSEOptions options = {});

} // namespace else_sim
