#pragma once
#include "else/else_solver.hpp"
#include "else/subnetwork.hpp"
#include <map>
#include <memory>
#include <numerics.hpp>
#include <vector>

namespace else_sim {

struct DensitySolution {
  std::vector<markovkit::State> states;
  std::vector<double> probability;
};

/// Invert the Laplace-domain density propagated through an ELSE subnetwork chain.
/// Small shifted systems use dense LU; larger systems use sparse UMFPACK.
class TalbotDensitySolver {
public:
  explicit TalbotDensitySolver(std::vector<DensitySubnetwork> subnetworks);
  ~TalbotDensitySolver();
  TalbotDensitySolver(TalbotDensitySolver&&) noexcept;
  TalbotDensitySolver& operator=(TalbotDensitySolver&&) noexcept;

  [[nodiscard]] const std::vector<markovkit::State>& states() const;

  [[nodiscard]] DensitySolution
  solve(const markovkit::State& initial, double time, num::idx nodes = 14);
  [[nodiscard]] DensitySolution
  solve(const std::map<markovkit::State, double>& initial, double time, num::idx nodes = 14);
  [[nodiscard]] std::vector<DensitySolution>
  solve(const std::vector<markovkit::State>& initial_states, double time, num::idx nodes = 14);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace else_sim
