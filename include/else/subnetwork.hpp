#pragma once
#include "cme/reversible.hpp"
#include "markovkit/boundary_transition.hpp"
#include <memory>
#include <numerics.hpp>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace else_sim {

/// Occupation and time-weighted occupation from a shared factorization of -R.
struct OccupationIntegrals {
    num::Matrix occupation;               // Integral of p(t) dt.
    num::Matrix time_weighted_occupation; // Integral of t p(t) dt.

    [[nodiscard]] num::idx size() const { return occupation.rows(); }
    [[nodiscard]] num::idx starts() const { return occupation.cols(); }
};

/// Restricted generator, boundary transitions, and occupation solver.
class Subnetwork {
  public:
    explicit Subnetwork(cme::RestrictedGenerator restricted);
    explicit Subnetwork(cme::ReversibleGenerator reversible);
    ~Subnetwork();
    Subnetwork(Subnetwork &&) noexcept;
    Subnetwork &operator=(Subnetwork &&) noexcept;

    [[nodiscard]] num::idx size() const { return states_.size(); }
    [[nodiscard]] const std::vector<markovkit::State> &states() const { return states_; }
    [[nodiscard]] const std::vector<num::idx> &boundary_states() const { return boundary_states_; }
    [[nodiscard]] const std::vector<markovkit::BoundaryTransition> &boundary() const {
        return boundary_;
    }
    [[nodiscard]] const num::SparseMatrix &generator() const { return R_; }
    [[nodiscard]] bool is_reversible() const;
    [[nodiscard]] std::span<const double> stationary_weights() const;
    [[nodiscard]] const num::Vector &inverse_column_sums() const;

    [[nodiscard]] int find(const markovkit::State &x) const;

    // Solve (-R)u = start; u is the expected time spent in each state.
    [[nodiscard]] num::Vector occupation(const num::Vector &start) const;

    // Compute occupation integrals using one factorization.
    [[nodiscard]] OccupationIntegrals occupation_integrals(const num::Matrix &starts) const;

    [[nodiscard]] std::vector<double> exit_probabilities(const num::Vector &occupation) const;
    [[nodiscard]] std::vector<double> exit_probabilities(const OccupationIntegrals &occupation,
                                                         num::idx column = 0) const;

    [[nodiscard]] double conditional_exit_time(const OccupationIntegrals &occupation,
                                               num::idx boundary_state, num::idx column = 0) const;

    [[nodiscard]] std::vector<double> expected_entries(const num::Vector &occupation,
                                                       const num::Vector &start) const;

    // Exact mean-exit-time loss from removing each state.
    [[nodiscard]] std::vector<double> cut_time_losses(const num::Vector &occupation) const;

    // Exact singleton losses for a selected candidate set.
    [[nodiscard]] std::vector<double> cut_time_losses(const num::Vector &occupation,
                                                      std::span<const num::idx> candidates) const;

    /// @brief Result of evaluating cut-time loss with floating-point error tracking.
    struct CutTimeLossResult {
        double loss = 0.0;             ///< Mean holding-time loss \f$\Delta \tau = \tau - \tau_{S^c}\f$
        double estimated_error = 0.0;   ///< Component-wise backward residual \f$\|Z_{SS} c - u_S\|_\infty\f$
        bool naive_fallback_used = false; ///< True if naive reduced scratch refactor was used
    };

    /// @brief Exact combined loss from removing a state block via the Woodbury formula,
    /// with automatic component-wise residual tracking and fallback to the naive reduced solve.
    [[nodiscard]] double cut_time_loss(const num::Vector &occupation,
                                       std::span<const num::idx> removed_states,
                                       double tolerance = 1e-6) const;

    /// @brief Detailed cut-time loss evaluation returning loss, floating-point error, and fallback diagnostics.
    [[nodiscard]] CutTimeLossResult cut_time_loss_with_diagnostics(
        const num::Vector &occupation, std::span<const num::idx> removed_states,
        double tolerance = 1e-6) const;

    /// @brief Direct ground-truth naive method: refactorizes the reduced generator from scratch.
    [[nodiscard]] double naive_cut_time_loss(const num::Vector &occupation,
                                             std::span<const num::idx> removed_states) const;

  private:
    void initialize(cme::RestrictedGenerator restricted);

    std::vector<markovkit::State> states_;
    std::unordered_map<markovkit::State, int> index_;
    num::SparseMatrix R_{0, 0, {}, {}, {0}};
    std::vector<double> total_rates_;
    std::vector<double> exit_rates_;
    std::vector<num::idx> boundary_states_;
    std::vector<markovkit::BoundaryTransition> boundary_;

    struct SolverImpl;
    std::unique_ptr<SolverImpl> solver_;
};

} // namespace else_sim
