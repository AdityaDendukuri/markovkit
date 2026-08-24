#include "else/subnetwork.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace else_sim {

struct Subnetwork::SolverImpl {
    // General subnetworks use an automatically selected dense or sparse solver.
    explicit SolverImpl(const num::SparseMatrix &operator_matrix)
        : general_solver(std::make_unique<num::AutoLinearSolver>(operator_matrix)) {}

    SolverImpl(num::Matrix symmetric, std::vector<double> stationary_sqrt)
        : weights(std::move(stationary_sqrt)) {
        // Reversible subnetworks reduce to a positive-definite Cholesky solve.
        cholesky_factor = num::cholesky(num::linalg::assume_spd(std::move(symmetric)));
        if (!cholesky_factor->success) {
            throw std::invalid_argument("reversible subnetwork operator is not positive definite");
        }
    }

    // solver vars
    std::unique_ptr<num::AutoLinearSolver> general_solver; // picks LU or sparse LU etc..
    std::optional<num::CholeskyResult> cholesky_factor; // cholesky would be non-empty if symmetric
    std::vector<double> weights;

    // inverse diagonal vars
    mutable num::InverseDiagonalWorkspace inverse_workspace;
    mutable std::optional<num::Vector> inverse_column_sums_cache;
    mutable std::vector<double> inverse_diagonal_cache;

    void solve(const num::Vector &b, num::Vector &x) const {
        if (cholesky_factor) {
            // Convert between generator and symmetric coordinates around the solve.
            num::Vector transformed_b = b;
            num::divide_elements(transformed_b, weights);
            num::Vector transformed_x(b.size(), 0.0);
            num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
            x = std::move(transformed_x);
            num::scale_elements(x, weights);
            return;
        }
        general_solver->solve(b, x);
    }

    void solve(const num::Matrix &right_hand_side, num::Matrix &solution) const {
        if (cholesky_factor) {
            num::Matrix transformed = right_hand_side;
            num::divide_rows(transformed, weights);
            num::cholesky_solve(*cholesky_factor, transformed, solution);
            num::scale_rows(solution, weights);
            return;
        }
        general_solver->solve(right_hand_side, solution);
    }

    void solve_transpose(const num::Vector &b, num::Vector &x) const {
        if (cholesky_factor) {
            // The transpose solve reverses the similarity scaling.
            num::Vector transformed_b = b;
            num::scale_elements(transformed_b, weights);
            num::Vector transformed_x(b.size(), 0.0);
            num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
            x = std::move(transformed_x);
            num::divide_elements(x, weights);
            return;
        }
        general_solver->solve_transpose(b, x);
    }

    [[nodiscard]] const num::Vector &inverse_column_sums() const {
        if (!inverse_column_sums_cache) {
            const num::idx n = cholesky_factor ? cholesky_factor->L.rows() : general_solver->size();
            num::Vector ones(n, 1.0);
            num::Vector sums(n, 0.0);
            solve_transpose(ones, sums);
            inverse_column_sums_cache = std::move(sums);
        }
        return *inverse_column_sums_cache;
    }

    void inverse_diagonal(std::span<const num::idx> candidates,
                          std::vector<double> &diagonal) const {
        if (diagonal.size() != candidates.size()) {
            throw std::invalid_argument("candidate and inverse-diagonal sizes must match");
        }
        const num::idx n = cholesky_factor ? cholesky_factor->L.rows() : general_solver->size();
        if (inverse_diagonal_cache.empty()) {
            inverse_diagonal_cache.assign(n, std::numeric_limits<double>::quiet_NaN());
        }

        std::vector<num::idx> missing;
        for (num::idx candidate : candidates) {
            if (candidate >= n) {
                throw std::out_of_range("cut-time candidate is out of range");
            }
            if (std::isnan(inverse_diagonal_cache[candidate])) {
                missing.push_back(candidate);
            }
        }
        std::sort(missing.begin(), missing.end());
        missing.erase(std::unique(missing.begin(), missing.end()), missing.end());

        if (!missing.empty()) {
            num::Vector values(missing.size(), 0.0);
            if (cholesky_factor) {
                num::selected_inverse(*cholesky_factor, missing, missing, values,
                                      inverse_workspace);
            } else {
                num::selected_inverse(*general_solver, missing, missing, values, inverse_workspace);
            }
            for (num::idx index = 0; index < missing.size(); ++index) {
                inverse_diagonal_cache[missing[index]] = values[index];
            }
        }

        for (num::idx index = 0; index < candidates.size(); ++index) {
            diagonal[index] = inverse_diagonal_cache[candidates[index]];
        }
    }

    // =========================================================================
    // Woodbury Principal Block Solve: Z_{SS} c = u_S
    // =========================================================================
    // Solves the reduced system against the principal inverse block Z_{SS} = [(-R)^{-1}]_{S,S}.
    // In reversible coordinates, factors (S^{-1})_{SS} with Cholesky in O(|S|^3) time.
    void solve_principal_inverse(std::span<const num::idx> indices,
                                 const num::Vector &right_hand_side,
                                 num::Vector &solution,
                                 num::Matrix *extracted_block = nullptr) const {
        if (right_hand_side.size() != indices.size() || solution.size() != indices.size()) {
            throw std::invalid_argument("principal inverse solve dimensions must match");
        }

        num::Matrix block;
        if (cholesky_factor) {
            num::inverse_principal_block(*cholesky_factor, indices, block, inverse_workspace);
            if (extracted_block) {
                *extracted_block = block;
            }
            const auto block_factor = num::cholesky(num::assume_spd(std::move(block)));
            if (!block_factor.success) {
                throw std::runtime_error("cut-time inverse block is not positive definite");
            }

            // Solve in symmetric coordinates, then restore generator coordinates.
            num::Vector transformed = right_hand_side;
            for (num::idx index = 0; index < indices.size(); ++index) {
                transformed[index] /= weights[indices[index]];
            }
            num::cholesky_solve(block_factor, transformed, solution);
            for (num::idx index = 0; index < indices.size(); ++index) {
                solution[index] *= weights[indices[index]];
            }
            return;
        }

        num::inverse_principal_block(*general_solver, indices, block, inverse_workspace);
        if (extracted_block) {
            *extracted_block = block;
        }
        const auto block_factor = num::lu(num::assume_square(std::move(block)));
        if (block_factor.singular) {
            throw std::runtime_error("cut-time inverse block is singular");
        }
        num::lu_solve(block_factor, right_hand_side, solution);
    }
};

Subnetwork::Subnetwork(cme::RestrictedGenerator restricted) {
    initialize(std::move(restricted));
    solver_ = std::make_unique<SolverImpl>(num::scaled(R_, -1.0));
}

Subnetwork::Subnetwork(cme::ReversibleGenerator reversible) {
    auto weights = std::move(reversible.stationary_sqrt);
    auto symmetric = std::move(reversible.symmetric_operator);
    initialize(std::move(reversible.restricted));
    solver_ = std::make_unique<SolverImpl>(std::move(symmetric), std::move(weights));
}

void Subnetwork::initialize(cme::RestrictedGenerator restricted) {
    // Cache the restricted generator and its boundary exit rates.
    states_ = std::move(restricted.states);
    index_ = std::move(restricted.index);
    R_ = std::move(restricted.generator);
    total_rates_.assign(states_.size(), 0.0);
    exit_rates_.assign(states_.size(), 0.0);
    boundary_ = std::move(restricted.boundary);
    if (R_.n_rows() != states_.size() || R_.n_cols() != states_.size()) {
        throw std::invalid_argument("subnetwork generator dimensions do not match its states");
    }
    for (num::idx state = 0; state < states_.size(); ++state) {
        if (!index_.contains(states_[state])) {
            index_[states_[state]] = static_cast<int>(state);
        }
        total_rates_[state] = -R_(state, state);
    }
    for (const auto &transition : boundary_) {
        if (transition.source >= states_.size()) {
            throw std::invalid_argument("subnetwork boundary source is out of range");
        }
        exit_rates_[transition.source] += transition.rate;
    }
    for (num::idx state = 0; state < states_.size(); ++state) {
        if (exit_rates_[state] > 0.0) {
            boundary_states_.push_back(state);
        }
    }
}

Subnetwork::~Subnetwork() = default;
Subnetwork::Subnetwork(Subnetwork &&) noexcept = default;
Subnetwork &Subnetwork::operator=(Subnetwork &&) noexcept = default;

int Subnetwork::find(const markovkit::State &x) const {
    auto it = index_.find(x);
    return it != index_.end() ? it->second : -1;
}

bool Subnetwork::is_reversible() const {
    return solver_ && solver_->cholesky_factor.has_value();
}

const num::Vector &Subnetwork::inverse_column_sums() const {
    return solver_->inverse_column_sums();
}

num::Vector Subnetwork::occupation(const num::Vector &start) const {
    // Solving -R u = p0 integrates the transient state probabilities.
    num::Vector u(size(), 0.0);
    solver_->solve(start, u);
    return u;
}

OccupationIntegrals Subnetwork::occupation_integrals(const num::Matrix &starts) const {
    OccupationIntegrals integrals;
    integrals.occupation = num::Matrix(size(), starts.cols(), 0.0);
    integrals.time_weighted_occupation = num::Matrix(size(), starts.cols(), 0.0);
    solver_->solve(starts, integrals.occupation);
    // Applying (-R)^-1 twice gives integral t p(t) dt without forming the inverse.
    solver_->solve(integrals.occupation, integrals.time_weighted_occupation);
    return integrals;
}

std::vector<double> Subnetwork::exit_probabilities(const num::Vector &occupancy) const {
    // Integrated boundary flux gives the exit-state distribution.
    std::vector<double> probs;
    probs.reserve(boundary_states_.size());
    for (num::idx b : boundary_states_) {
        double p = exit_rates_[b] * std::max(0.0, occupancy[b]);
        probs.push_back(p);
    }
    num::clip_and_normalize_nonnegative(probs);
    return probs;
}

std::vector<double> Subnetwork::exit_probabilities(const OccupationIntegrals &integrals,
                                                   num::idx col) const {
    std::vector<double> probs;
    probs.reserve(boundary_states_.size());
    for (num::idx b : boundary_states_) {
        double p = exit_rates_[b] * std::max(0.0, integrals.occupation(b, col));
        probs.push_back(p);
    }
    num::clip_and_normalize_nonnegative(probs);
    return probs;
}

double Subnetwork::conditional_exit_time(const OccupationIntegrals &integrals,
                                         num::idx exit_state_idx, num::idx col) const {
    // Divide the first time moment by the exit state's occupation.
    double u = integrals.occupation(exit_state_idx, col);
    if (!(u > 0.0)) {
        return 0.0;
    }
    return std::max(0.0, integrals.time_weighted_occupation(exit_state_idx, col)) / u;
}

std::vector<double> Subnetwork::expected_entries(const num::Vector &occupancy,
                                                 const num::Vector &start) const {
    std::vector<double> counts(size(), 0.0);
    for (num::idx j = 0; j < size(); ++j) {
        // Departures equal rate times occupation; subtract the initial entrance.
        counts[j] = std::max(0.0, (total_rates_[j] * occupancy[j]) - start[j]);
    }
    return counts;
}

std::vector<double> Subnetwork::cut_time_losses(const num::Vector &occupancy) const {
    const num::idx n = size();
    std::vector<num::idx> candidates(n);
    std::iota(candidates.begin(), candidates.end(), num::idx{0});
    return cut_time_losses(occupancy, candidates);
}

std::vector<double> Subnetwork::cut_time_losses(const num::Vector &occupancy,
                                                std::span<const num::idx> candidates) const {
    if (occupancy.size() != size()) {
        throw std::invalid_argument("cut-time occupation size must match the subnetwork");
    }
    const auto &column_sums = solver_->inverse_column_sums();
    std::vector<double> diagonal(candidates.size(), 0.0);
    solver_->inverse_diagonal(candidates, diagonal);

    std::vector<double> losses(candidates.size(), 0.0);
    for (num::idx index = 0; index < candidates.size(); ++index) {
        const num::idx candidate = candidates[index];
        if (diagonal[index] > 1e-15) {
            losses[index] = occupancy[candidate] * (column_sums[candidate] / diagonal[index]);
        }
    }
    return losses;
}

// =============================================================================
// Direct Ground-Truth Naive Method: Scratch Reduced Generator Factorization
// =============================================================================
// Explicitly forms the (n - |S|) x (n - |S|) reduced generator submatrix
// -R_{S^c, S^c}, solves for reduced transient occupation u_{reduced}, and
// computes exact ground-truth loss: \Delta \tau = 1^T u - 1^T u_{reduced}.
double Subnetwork::naive_cut_time_loss(const num::Vector &occupancy,
                                       std::span<const num::idx> removed_states) const {
    const num::idx n = size();
    const num::idx r = removed_states.size();
    if (r == 0) {
        return 0.0;
    }
    if (r >= n) {
        double total = 0.0;
        for (num::idx i = 0; i < n; ++i) {
            total += occupancy[i];
        }
        return total;
    }

    std::vector<bool> is_removed(n, false);
    for (num::idx s : removed_states) {
        if (s < n) {
            is_removed[s] = true;
        }
    }

    std::vector<num::idx> kept;
    kept.reserve(n - r);
    for (num::idx i = 0; i < n; ++i) {
        if (!is_removed[i]) {
            kept.push_back(i);
        }
    }
    const num::idx m = kept.size();

    // 1. Build reduced generator submatrix -R_reduced
    num::Matrix R_red(m, m, 0.0);
    for (num::idx i = 0; i < m; ++i) {
        for (num::idx j = 0; j < m; ++j) {
            R_red(i, j) = -R_(kept[i], kept[j]);
        }
    }

    // 2. Compute entrance vector p_0 = (-R) * occupancy
    num::Vector p_red(m, 0.0);
    for (num::idx i = 0; i < m; ++i) {
        const num::idx orig_i = kept[i];
        double p_val = 0.0;
        for (num::idx j = 0; j < n; ++j) {
            p_val += -R_(orig_i, j) * occupancy[j];
        }
        p_red[i] = std::max(0.0, p_val);
    }

    // 3. Solve (-R_red) u_red = p_red directly
    num::Vector u_red(m, 0.0);
    auto lu_red = num::lu(num::assume_square(std::move(R_red)));
    if (lu_red.singular) {
        return 0.0;
    }
    num::lu_solve(lu_red, p_red, u_red);

    // 4. Loss = sum(occupancy) - sum(u_red)
    double tau_orig = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        tau_orig += occupancy[i];
    }
    double tau_red = 0.0;
    for (num::idx i = 0; i < m; ++i) {
        tau_red += std::max(0.0, u_red[i]);
    }
    return std::max(0.0, tau_orig - tau_red);
}

// =============================================================================
// Woodbury Principal Block Cut-Time Update with Component-Wise Residual Tracking
// =============================================================================
// Evaluates the exact holding time loss: \ell_S = q_S^T * (Z_{SS})^{-1} * u_S.
// Computes backward error ||Z_{SS} c - u_S||_inf. If the residual exceeds
// tolerance (e.g. ill-conditioning), it falls back to naive_cut_time_loss.
Subnetwork::CutTimeLossResult
Subnetwork::cut_time_loss_with_diagnostics(const num::Vector &occupancy,
                                           std::span<const num::idx> removed_states,
                                           double tolerance) const {
    if (occupancy.size() != size()) {
        throw std::invalid_argument("cut-time occupation size must match the subnetwork");
    }
    if (removed_states.empty()) {
        return {.loss = 0.0, .estimated_error = 0.0, .naive_fallback_used = false};
    }
    if (removed_states.size() == 1) {
        double single_loss = cut_time_losses(occupancy, removed_states).front();
        return {.loss = single_loss, .estimated_error = 0.0, .naive_fallback_used = false};
    }

    const num::idx r = removed_states.size();
    num::Vector u_S(r, 0.0);
    for (num::idx i = 0; i < r; ++i) {
        u_S[i] = occupancy[removed_states[i]];
    }

    num::Vector coefficients(r, 0.0);
    num::Matrix Z_SS;
    bool fast_success = false;
    double residual_norm = 0.0;

    try {
        solver_->solve_principal_inverse(removed_states, u_S, coefficients, &Z_SS);

        // Component-wise backward residual: || Z_SS * c - u_S ||_inf / ||u_S||_inf
        double max_res = 0.0;
        double max_rhs = 0.0;
        for (num::idx i = 0; i < r; ++i) {
            max_rhs = std::max(max_rhs, std::abs(u_S[i]));
            double reconstructed = 0.0;
            for (num::idx j = 0; j < r; ++j) {
                reconstructed += Z_SS(i, j) * coefficients[j];
            }
            max_res = std::max(max_res, std::abs(reconstructed - u_S[i]));
        }
        residual_norm = max_res / std::max(1e-12, max_rhs);

        if (residual_norm <= tolerance && std::isfinite(residual_norm)) {
            fast_success = true;
        }
    } catch (...) {
        fast_success = false;
        residual_norm = std::numeric_limits<double>::infinity();
    }

    if (fast_success) {
        const auto &column_sums = solver_->inverse_column_sums();
        double loss = 0.0;
        for (num::idx i = 0; i < r; ++i) {
            loss += column_sums[removed_states[i]] * coefficients[i];
        }
        if (std::isfinite(loss) && loss >= 0.0) {
            return {.loss = loss, .estimated_error = residual_norm, .naive_fallback_used = false};
        }
    }

    // Residual threshold breached or non-finite -> Recompute via naive ground-truth solve
    double naive_loss = naive_cut_time_loss(occupancy, removed_states);
    return {.loss = naive_loss, .estimated_error = residual_norm, .naive_fallback_used = true};
}

double Subnetwork::cut_time_loss(const num::Vector &occupancy,
                                 std::span<const num::idx> removed_states,
                                 double tolerance) const {
    return cut_time_loss_with_diagnostics(occupancy, removed_states, tolerance).loss;
}

} // namespace else_sim
