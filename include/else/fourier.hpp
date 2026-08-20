#pragma once
#include "else/else_solver.hpp"
#include "else/subnetwork.hpp"
#include <map>
#include <memory>
#include <numerics.hpp>
#include <vector>

namespace else_sim {

// Shifted Fourier nodes: s_k = shift + i * k * pi / period
std::vector<num::cplx> fourier_shifts(int modes, double period, double shift);

// Coefficients matrix across states and modes
struct FourierCoefficients {
  num::idx states = 0;
  int modes = 0;
  double period = 0.0;
  double shift = 0.0;
  std::vector<num::cplx> values;

  FourierCoefficients() = default;
  FourierCoefficients(num::idx state_count, int mode_count, double period, double shift);

  num::cplx& operator()(num::idx state, int mode) {
    return values[(state * (modes + 1)) + mode];
  }
  num::cplx operator()(num::idx state, int mode) const {
    return values[(state * (modes + 1)) + mode];
  }
};

// Transform of density steps across Fourier modes
FourierCoefficients density_transform(const std::vector<DensityStep>& steps,
                                      int modes,
                                      double period,
                                      double shift,
                                      std::vector<MultiIndex>* states_out = nullptr);

// Invert Fourier series at time t with Fejer damping
std::vector<double>
invert_fourier(const FourierCoefficients& coefficients, double time, bool normalize = true);

// Talbot contour propagation: exp(R*t) * v via shifted solves
num::Vector
propagate_talbot(const Subnetwork& sub, const num::Vector& v, double time, num::idx nodes = 32);

struct DensitySolution {
  std::vector<MultiIndex> states;
  std::vector<double> probability;
};

// Talbot inversion of density propagated through one or more ELSE steps.
// The same implementation is used for CME and graph-Laplacian steps.
class TalbotDensitySolver {
public:
  explicit TalbotDensitySolver(std::vector<DensityStep> steps);
  ~TalbotDensitySolver();
  TalbotDensitySolver(TalbotDensitySolver&&) noexcept;
  TalbotDensitySolver& operator=(TalbotDensitySolver&&) noexcept;

  [[nodiscard]] const std::vector<MultiIndex>& states() const;

  [[nodiscard]] DensitySolution solve(const MultiIndex& initial, double time, num::idx nodes = 14);
  [[nodiscard]] DensitySolution
  solve(const std::map<MultiIndex, double>& initial, double time, num::idx nodes = 14);
  [[nodiscard]] std::vector<DensitySolution>
  solve(const std::vector<MultiIndex>& initial_states, double time, num::idx nodes = 14);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace else_sim
