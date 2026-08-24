#pragma once
#include "else/subnetwork.hpp"
#include "else/trajectory.hpp"
#include <map>
#include <memory>
#include <numerics.hpp>
#include <vector>

namespace else_sim {

struct DensitySolution {
    std::vector<markovkit::State> states;
    std::vector<double> probability;
};

struct ContourNodeStatus {
    double time;
    num::idx node_index;
    num::idx total_nodes;
    num::cplx shift;
};

using ContourObserver = std::function<void(const ContourNodeStatus &)>;
using DensityObserver = ContourObserver;
using TalbotNodeStatus = ContourNodeStatus;

/// Options for numerical contour inversion of the Laplace-domain density.
struct ContourOptions {
    num::idx nodes = 14;
};

/// Invert the Laplace-domain density propagated through an ELSE subnetwork chain
/// via shifted resolvent linear solves on an inversion contour.
class LaplaceDensitySolver {
  public:
    explicit LaplaceDensitySolver(std::vector<cme::RestrictedGenerator> subnetworks);
    ~LaplaceDensitySolver();
    LaplaceDensitySolver(LaplaceDensitySolver &&) noexcept;
    LaplaceDensitySolver &operator=(LaplaceDensitySolver &&) noexcept;

    [[nodiscard]] const std::vector<markovkit::State> &states() const;

    [[nodiscard]] DensitySolution solve(const markovkit::State &initial, double time,
                                        num::idx nodes = 14,
                                        ContourObserver observer = nullptr);
    [[nodiscard]] DensitySolution solve(const markovkit::State &initial, double time,
                                        ContourOptions options,
                                        ContourObserver observer = nullptr);
    [[nodiscard]] DensitySolution solve(const std::map<markovkit::State, double> &initial,
                                        double time, num::idx nodes = 14,
                                        ContourObserver observer = nullptr);
    [[nodiscard]] DensitySolution solve(const std::map<markovkit::State, double> &initial,
                                        double time, ContourOptions options,
                                        ContourObserver observer = nullptr);
    [[nodiscard]] std::vector<DensitySolution>
    solve(const std::vector<markovkit::State> &initial_states, double time, num::idx nodes = 14,
          ContourObserver observer = nullptr);
    [[nodiscard]] std::vector<DensitySolution>
    solve(const std::vector<markovkit::State> &initial_states, double time, ContourOptions options,
          ContourObserver observer = nullptr);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using TalbotDensitySolver = LaplaceDensitySolver;

} // namespace else_sim
