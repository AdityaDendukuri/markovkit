/// @file rdme/population_multigrid.hpp
/// @brief Population-conserving geometric multigrid for structured-grid RDME diffusion.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/factorization/lu.hpp"
#include "rdme/diffusion_generator.hpp"
#include "rdme/reaction_state_space.hpp"
#include "structures/graph/structured_grid.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rdme {

/// Lift one factor-two Cartesian voxel aggregation to a fixed-population state space.
///
/// Restriction maps every fine configuration to its aggregate coarse configuration.
/// Prolongation distributes each coarse population independently and uniformly among
/// the fine children. Consequently R P = I, both maps preserve total probability,
/// and neither map changes a species total.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
class PopulationTransfer {
  public:
    using graph_type = num::structured_grid_graph<D, Weight, Index>;

    PopulationTransfer(const graph_type &fine_graph, const DiffusionStateSpace &fine_space,
                       num::array<int> species_totals)
        : coarse_graph_(make_coarse_graph(fine_graph)),
          coarse_space_(static_cast<std::size_t>(coarse_graph_.n_vertices()),
                        std::move(species_totals)),
          fine_to_coarse_(fine_space.size(), 0), prolongation_weights_(fine_space.size(), 0.0) {
        if (fine_space.layout().voxel_count() !=
            static_cast<std::size_t>(fine_graph.n_vertices())) {
            throw std::invalid_argument("RDME transfer grid and state space do not match");
        }
        build(fine_graph, fine_space);
    }

    [[nodiscard]] num::idx fine_size() const noexcept { return fine_to_coarse_.size(); }
    [[nodiscard]] num::idx coarse_size() const noexcept { return coarse_space_.size(); }
    [[nodiscard]] const graph_type &coarse_graph() const noexcept { return coarse_graph_; }
    [[nodiscard]] const DiffusionStateSpace &coarse_state_space() const noexcept {
        return coarse_space_;
    }
    [[nodiscard]] const num::array<num::idx> &fine_to_coarse() const noexcept {
        return fine_to_coarse_;
    }
    [[nodiscard]] const num::array<double> &prolongation_weights() const noexcept {
        return prolongation_weights_;
    }
    [[nodiscard]] const num::array<Index> &voxel_parent() const noexcept { return voxel_parent_; }

    void restrict(std::span<const double> fine, std::span<double> coarse) const {
        check_dimensions(fine.size(), coarse.size());
        std::fill(coarse.begin(), coarse.end(), 0.0);
        for (num::idx state = 0; state < fine_size(); ++state) {
            coarse[fine_to_coarse_[state]] += fine[state];
        }
    }

    void prolong(std::span<const double> coarse, std::span<double> fine) const {
        check_dimensions(fine.size(), coarse.size());
        for (num::idx state = 0; state < fine_size(); ++state) {
            fine[state] = prolongation_weights_[state] * coarse[fine_to_coarse_[state]];
        }
    }

  private:
    [[nodiscard]] static graph_type make_coarse_graph(const graph_type &fine) {
        auto dimensions = fine.dimensions();
        auto spacing = fine.spacing();
        bool reduced = false;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (dimensions[axis] == 1)
                continue;
            if (dimensions[axis] % 2 != 0) {
                throw std::invalid_argument(
                    "RDME factor-two coarsening requires even nontrivial grid dimensions");
            }
            dimensions[axis] /= 2;
            spacing[axis] *= Weight{2};
            reduced = true;
        }
        if (!reduced) {
            throw std::invalid_argument("RDME multigrid requires at least one coarsenable axis");
        }
        return graph_type(dimensions, spacing, fine.boundaries());
    }

    void build(const graph_type &fine_graph, const DiffusionStateSpace &fine_space) {
        const auto &fine_layout = fine_space.layout();
        const auto &coarse_layout = coarse_space_.layout();
        voxel_parent_.resize(static_cast<std::size_t>(fine_graph.n_vertices()));
        num::array<std::size_t> child_count(static_cast<std::size_t>(coarse_graph_.n_vertices()),
                                            0);

        for (Index fine_voxel = 0; fine_voxel < fine_graph.n_vertices(); ++fine_voxel) {
            auto coordinate = fine_graph.coordinate(fine_voxel);
            for (std::size_t axis = 0; axis < D; ++axis) {
                if (fine_graph.dimensions()[axis] > 1)
                    coordinate[axis] /= 2;
            }
            const Index parent = coarse_graph_.vertex(coordinate);
            voxel_parent_[static_cast<std::size_t>(fine_voxel)] = parent;
            ++child_count[static_cast<std::size_t>(parent)];
        }

        num::array<double> coarse_weight_sum(coarse_size(), 0.0);
        for (num::idx fine_state = 0; fine_state < fine_space.size(); ++fine_state) {
            const Configuration &fine_configuration = fine_space[fine_state];
            Configuration coarse_configuration(coarse_layout.size(), 0);
            for (std::size_t fine_voxel = 0; fine_voxel < fine_layout.voxel_count(); ++fine_voxel) {
                const std::size_t coarse_voxel =
                    static_cast<std::size_t>(voxel_parent_[fine_voxel]);
                for (std::size_t species = 0; species < fine_layout.species_count(); ++species) {
                    coarse_configuration[coarse_layout.index(coarse_voxel, species)] +=
                        fine_configuration[fine_layout.index(fine_voxel, species)];
                }
            }

            const num::idx coarse_state = coarse_space_.find(coarse_configuration);
            if (coarse_state == DiffusionStateSpace::npos) {
                throw std::logic_error("RDME aggregation changed a conserved population");
            }
            fine_to_coarse_[fine_state] = coarse_state;

            double log_weight = 0.0;
            for (std::size_t coarse_voxel = 0; coarse_voxel < coarse_layout.voxel_count();
                 ++coarse_voxel) {
                const double children = static_cast<double>(child_count[coarse_voxel]);
                for (std::size_t species = 0; species < coarse_layout.species_count(); ++species) {
                    const int total =
                        coarse_configuration[coarse_layout.index(coarse_voxel, species)];
                    log_weight += std::lgamma(static_cast<double>(total) + 1.0) -
                                  static_cast<double>(total) * std::log(children);
                }
            }
            for (std::size_t fine_voxel = 0; fine_voxel < fine_layout.voxel_count(); ++fine_voxel) {
                for (std::size_t species = 0; species < fine_layout.species_count(); ++species) {
                    log_weight -= std::lgamma(
                        static_cast<double>(
                            fine_configuration[fine_layout.index(fine_voxel, species)]) +
                        1.0);
                }
            }
            const double weight = std::exp(log_weight);
            prolongation_weights_[fine_state] = weight;
            coarse_weight_sum[coarse_state] += weight;
        }

        for (num::idx fine_state = 0; fine_state < fine_space.size(); ++fine_state) {
            prolongation_weights_[fine_state] /= coarse_weight_sum[fine_to_coarse_[fine_state]];
        }
    }

    void check_dimensions(std::size_t fine, std::size_t coarse) const {
        if (fine != fine_size() || coarse != coarse_size()) {
            throw std::invalid_argument("RDME population transfer vector size mismatch");
        }
    }

    graph_type coarse_graph_;
    DiffusionStateSpace coarse_space_;
    num::array<Index> voxel_parent_;
    num::array<num::idx> fine_to_coarse_;
    num::array<double> prolongation_weights_;
};

enum class PopulationSmoother { weighted_jacobi, aggregate_block };

/// A recursive geometric V-cycle for a shifted fixed-sector diffusion operator.
///
/// The historical name is retained for source compatibility. Intermediate levels
/// are matrix-free; only a terminal configuration space below `coarse_max_states`
/// is assembled and factored. On a uniform factor-two grid, doubling diffusivity
/// on every coarse level is exactly the conservative Petrov--Galerkin operator R A P.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
class TwoLevelPopulationMultigrid {
  public:
    using graph_type = num::structured_grid_graph<D, Weight, Index>;
    using domain_type = num::vec;
    using codomain_type = num::vec;
    using math_laws = num::math::type_list<num::law::linear_map>;

    TwoLevelPopulationMultigrid(VoxelDomain<graph_type> domain, num::array<int> species_totals,
                                double shift, std::size_t pre_sweeps = 2,
                                std::size_t post_sweeps = 2, double jacobi_weight = 2.0 / 3.0,
                                num::idx coarse_max_states = 4, std::size_t max_levels = 32,
                                std::size_t coarse_cycles = 1, std::size_t sweep_growth = 0,
                                PopulationSmoother smoother = PopulationSmoother::weighted_jacobi)
        : TwoLevelPopulationMultigrid(PrivateTag{}, std::move(domain), std::move(species_totals),
                                      shift, pre_sweeps, post_sweeps, jacobi_weight,
                                      coarse_max_states, max_levels, coarse_cycles, sweep_growth,
                                      smoother, 1) {}

    [[nodiscard]] num::idx rows() const noexcept { return fine_generator_.rows(); }
    [[nodiscard]] num::idx cols() const noexcept { return fine_generator_.cols(); }
    [[nodiscard]] double shift() const noexcept { return shift_; }
    [[nodiscard]] std::size_t n_levels() const noexcept {
        return coarse_level_ ? 1 + coarse_level_->n_levels() : 1;
    }
    [[nodiscard]] double grid_complexity() const noexcept {
        return static_cast<double>(hierarchy_states()) / static_cast<double>(rows());
    }
    [[nodiscard]] double operator_complexity() const noexcept {
        return static_cast<double>(hierarchy_nonzeros()) / static_cast<double>(level_nonzeros());
    }
    [[nodiscard]] bool has_coarse_level() const noexcept {
        return static_cast<bool>(coarse_level_);
    }
    [[nodiscard]] const TwoLevelPopulationMultigrid *coarse_level() const noexcept {
        return coarse_level_.get();
    }
    [[nodiscard]] const DiffusionGenerator<graph_type> &fine_generator() const noexcept {
        return fine_generator_;
    }
    [[nodiscard]] const PopulationTransfer<D, Weight, Index> &transfer() const {
        if (!transfer_)
            throw std::logic_error("terminal RDME multigrid level has no transfer");
        return *transfer_;
    }
    [[nodiscard]] const num::mat &coarse_inverse() const noexcept {
        return coarse_level_ ? coarse_level_->coarse_inverse() : exact_inverse_;
    }

    void apply_system(const num::vec &x, num::vec &y) const {
        if (x.size() != rows() || y.size() != rows()) {
            throw std::invalid_argument("RDME shifted diffusion vector size mismatch");
        }
        fine_generator_.apply(x, generator_image_);
        for (num::idx state = 0; state < rows(); ++state) {
            y[state] = shift_ * x[state] - generator_image_[state];
        }
    }

    /// Apply the level smoother to an existing iterate. This is primarily useful for
    /// diagnostics; a zero right-hand side exposes its error-propagation operator.
    void relax(const num::vec &rhs, num::vec &solution, std::size_t sweeps = 1,
               bool reverse = false) const {
        if (rhs.size() != rows() || solution.size() != rows()) {
            throw std::invalid_argument("RDME smoother vector size mismatch");
        }
        rhs_ = rhs;
        solution_ = solution;
        smooth(sweeps, reverse);
        solution = solution_;
    }

    void apply(const num::vec &x, num::vec &y) const { apply_block(x, y); }

    void apply_block(std::span<const double> x, std::span<double> y) const {
        if (x.size() != rows() || y.size() != rows()) {
            throw std::invalid_argument("RDME multigrid vector size mismatch");
        }
        std::copy(x.begin(), x.end(), rhs_.begin());
        std::fill(solution_.begin(), solution_.end(), 0.0);
        smooth(pre_sweeps_, false);

        apply_system(solution_, system_image_);
        for (num::idx state = 0; state < rows(); ++state) {
            residual_[state] = rhs_[state] - system_image_[state];
        }
        if (coarse_level_) {
            transfer_->restrict(residual_, coarse_rhs_);
            std::fill(coarse_solution_.begin(), coarse_solution_.end(), 0.0);
            for (std::size_t cycle = 0; cycle < coarse_cycles_; ++cycle) {
                if (cycle == 0) {
                    coarse_residual_ = coarse_rhs_;
                } else {
                    coarse_level_->apply_system(coarse_solution_, coarse_system_image_);
                    for (num::idx state = 0; state < coarse_rhs_.size(); ++state)
                        coarse_residual_[state] = coarse_rhs_[state] - coarse_system_image_[state];
                }
                coarse_level_->apply_block(coarse_residual_, coarse_correction_);
                for (num::idx state = 0; state < coarse_solution_.size(); ++state)
                    coarse_solution_[state] += coarse_correction_[state];
            }
            transfer_->prolong(coarse_solution_, correction_);
        } else {
            multiply_exact_inverse(residual_, correction_);
        }
        for (num::idx state = 0; state < rows(); ++state)
            solution_[state] += correction_[state];

        smooth(post_sweeps_, true);
        std::copy(solution_.begin(), solution_.end(), y.begin());
    }

  private:
    struct PrivateTag {};

    TwoLevelPopulationMultigrid(PrivateTag, VoxelDomain<graph_type> domain,
                                num::array<int> species_totals, double shift,
                                std::size_t pre_sweeps, std::size_t post_sweeps,
                                double jacobi_weight, num::idx coarse_max_states,
                                std::size_t max_levels, std::size_t coarse_cycles,
                                std::size_t sweep_growth, PopulationSmoother smoother,
                                std::size_t level)
        : fine_generator_(std::move(domain), species_totals), shift_(checked_shift(shift)),
          pre_sweeps_(pre_sweeps), post_sweeps_(post_sweeps),
          jacobi_weight_(checked_jacobi_weight(jacobi_weight)), coarse_cycles_(coarse_cycles),
          smoother_(smoother), rhs_(rows(), 0.0), solution_(rows(), 0.0),
          system_image_(rows(), 0.0), generator_image_(rows(), 0.0), residual_(rows(), 0.0),
          correction_(rows(), 0.0) {
        if (coarse_max_states == 0) {
            throw std::invalid_argument("RDME coarse state threshold must be positive");
        }
        if (max_levels == 0) {
            throw std::invalid_argument("RDME maximum multigrid levels must be positive");
        }
        if (coarse_cycles == 0) {
            throw std::invalid_argument("RDME coarse cycle count must be positive");
        }
        if (rows() > coarse_max_states && level < max_levels &&
            can_coarsen(fine_generator_.domain().graph())) {
            transfer_ = std::make_unique<PopulationTransfer<D, Weight, Index>>(
                fine_generator_.domain().graph(), fine_generator_.state_space(), species_totals);
            num::array<double> coarse_diffusivities(
                fine_generator_.domain().diffusivities().begin(),
                fine_generator_.domain().diffusivities().end());
            for (double &diffusivity : coarse_diffusivities)
                diffusivity *= 2.0;
            VoxelDomain<graph_type> coarse_domain(
                transfer_->coarse_graph(),
                std::vector<double>(coarse_diffusivities.begin(), coarse_diffusivities.end()));
            coarse_level_ =
                std::unique_ptr<TwoLevelPopulationMultigrid>(new TwoLevelPopulationMultigrid(
                    PrivateTag{}, std::move(coarse_domain), std::move(species_totals), shift_,
                    pre_sweeps_ + sweep_growth, post_sweeps_ + sweep_growth, jacobi_weight_,
                    coarse_max_states, max_levels, coarse_cycles_, sweep_growth, smoother_,
                    level + 1));
            coarse_rhs_ = num::vec(transfer_->coarse_size(), 0.0);
            coarse_solution_ = num::vec(transfer_->coarse_size(), 0.0);
            coarse_system_image_ = num::vec(transfer_->coarse_size(), 0.0);
            coarse_residual_ = num::vec(transfer_->coarse_size(), 0.0);
            coarse_correction_ = num::vec(transfer_->coarse_size(), 0.0);
            if (smoother_ == PopulationSmoother::aggregate_block)
                build_aggregate_blocks();
        } else {
            exact_inverse_ = build_exact_inverse();
        }
    }

    [[nodiscard]] static double checked_shift(double shift) {
        if (!(shift > 0.0) || !std::isfinite(shift)) {
            throw std::invalid_argument("RDME multigrid shift must be finite and positive");
        }
        return shift;
    }

    [[nodiscard]] static double checked_jacobi_weight(double weight) {
        if (!(weight > 0.0) || !(weight <= 1.0)) {
            throw std::invalid_argument("RDME Jacobi weight must lie in (0, 1]");
        }
        return weight;
    }

    [[nodiscard]] static bool can_coarsen(const graph_type &graph) {
        bool reduced = false;
        for (Index dimension : graph.dimensions()) {
            if (dimension == 1)
                continue;
            if (dimension % 2 != 0)
                return false;
            reduced = true;
        }
        return reduced;
    }

    [[nodiscard]] std::size_t level_nonzeros() const noexcept {
        return static_cast<std::size_t>(rows()) + fine_generator_.transitions().size();
    }

    [[nodiscard]] std::size_t hierarchy_states() const noexcept {
        return static_cast<std::size_t>(rows()) +
               (coarse_level_ ? coarse_level_->hierarchy_states() : 0);
    }

    [[nodiscard]] std::size_t hierarchy_nonzeros() const noexcept {
        return level_nonzeros() + (coarse_level_ ? coarse_level_->hierarchy_nonzeros() : 0);
    }

    [[nodiscard]] num::mat build_exact_inverse() {
        num::mat matrix(rows(), rows(), 0.0);
        num::vec basis(rows(), 0.0);
        num::vec image(rows(), 0.0);
        num::vec generator_image(rows(), 0.0);

        for (num::idx column = 0; column < rows(); ++column) {
            std::fill(basis.begin(), basis.end(), 0.0);
            basis[column] = 1.0;
            fine_generator_.apply(basis, generator_image);
            for (num::idx state = 0; state < rows(); ++state) {
                image[state] = shift_ * basis[state] - generator_image[state];
                matrix(state, column) = image[state];
            }
        }

        const num::lu_result factor = num::unsafe::lu(matrix);
        if (factor.singular) {
            throw std::runtime_error("RDME multigrid terminal operator is singular");
        }
        return num::lu_inv(factor);
    }

    struct Block {
        num::array<num::idx> states;
        num::lu_result factor;
        mutable num::vec rhs;
        mutable num::vec correction;
    };

    using BlockPartition = std::vector<Block>;

    void build_aggregate_blocks() {
        const auto &space = fine_generator_.state_space();
        const auto &layout = space.layout();
        const auto &voxel_parent = transfer_->voxel_parent();
        const std::size_t aggregate_count = transfer_->coarse_state_space().layout().voxel_count();
        block_partitions_.resize(aggregate_count);

        for (std::size_t aggregate = 0; aggregate < aggregate_count; ++aggregate) {
            std::unordered_map<Configuration, num::array<num::idx>, ConfigurationHash> groups;
            for (num::idx state = 0; state < rows(); ++state) {
                Configuration signature = space[state];
                for (std::size_t voxel = 0; voxel < layout.voxel_count(); ++voxel) {
                    if (static_cast<std::size_t>(voxel_parent[voxel]) != aggregate)
                        continue;
                    for (std::size_t species = 0; species < layout.species_count(); ++species)
                        signature[layout.index(voxel, species)] = 0;
                }
                groups[std::move(signature)].push_back(state);
            }

            auto &partition = block_partitions_[aggregate];
            partition.reserve(groups.size());
            num::array<num::idx> local_index(rows(), DiffusionStateSpace::npos);
            for (auto &[signature, states] : groups) {
                (void)signature;
                if (states.size() == 1)
                    continue;
                for (num::idx local = 0; local < states.size(); ++local)
                    local_index[states[local]] = local;

                num::mat matrix(states.size(), states.size(), 0.0);
                for (num::idx local_source = 0; local_source < states.size(); ++local_source) {
                    const num::idx source = states[local_source];
                    matrix(local_source, local_source) =
                        shift_ + fine_generator_.exit_rates()[source];
                    for (num::idx entry = fine_generator_.source_offsets()[source];
                         entry < fine_generator_.source_offsets()[source + 1]; ++entry) {
                        const auto &transition = fine_generator_.transitions()[entry];
                        const num::idx local_destination = local_index[transition.destination];
                        if (local_destination != DiffusionStateSpace::npos)
                            matrix(local_destination, local_source) -= transition.rate;
                    }
                }
                num::lu_result factor = num::unsafe::lu(matrix);
                if (factor.singular)
                    throw std::runtime_error("RDME aggregate smoother block is singular");
                const num::idx block_size = states.size();
                partition.push_back({std::move(states), std::move(factor),
                                     num::vec(block_size, 0.0), num::vec(block_size, 0.0)});
                for (num::idx state : partition.back().states)
                    local_index[state] = DiffusionStateSpace::npos;
            }
        }
    }

    void smooth_jacobi(std::size_t sweeps) const {
        for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
            apply_system(solution_, system_image_);
            for (num::idx state = 0; state < rows(); ++state) {
                const double diagonal = shift_ + fine_generator_.exit_rates()[state];
                solution_[state] +=
                    jacobi_weight_ * (rhs_[state] - system_image_[state]) / diagonal;
            }
        }
    }

    void smooth_blocks(std::size_t sweeps, bool reverse) const {
        apply_system(solution_, system_image_);
        for (num::idx state = 0; state < rows(); ++state)
            residual_[state] = rhs_[state] - system_image_[state];

        for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
            for (std::size_t step = 0; step < block_partitions_.size(); ++step) {
                const std::size_t aggregate = reverse ? block_partitions_.size() - step - 1 : step;
                for (Block &block : block_partitions_[aggregate]) {
                    for (num::idx local = 0; local < block.states.size(); ++local) {
                        const num::idx state = block.states[local];
                        block.rhs[local] = residual_[state];
                    }
                    num::lu_solve(block.factor, block.rhs, block.correction);
                }
                for (const Block &block : block_partitions_[aggregate]) {
                    for (num::idx local = 0; local < block.states.size(); ++local)
                        solution_[block.states[local]] += block.correction[local];
                    for (num::idx local = 0; local < block.states.size(); ++local) {
                        const num::idx source = block.states[local];
                        const double value = block.correction[local];
                        residual_[source] -=
                            (shift_ + fine_generator_.exit_rates()[source]) * value;
                        for (num::idx entry = fine_generator_.source_offsets()[source];
                             entry < fine_generator_.source_offsets()[source + 1]; ++entry) {
                            const auto &transition = fine_generator_.transitions()[entry];
                            residual_[transition.destination] += transition.rate * value;
                        }
                    }
                }
            }
        }
    }

    void smooth(std::size_t sweeps, bool reverse = false) const {
        if (smoother_ == PopulationSmoother::aggregate_block && !block_partitions_.empty())
            smooth_blocks(sweeps, reverse);
        else
            smooth_jacobi(sweeps);
    }

    void multiply_exact_inverse(std::span<const double> x, std::span<double> y) const {
        std::fill(y.begin(), y.end(), 0.0);
        for (num::idx row = 0; row < rows(); ++row) {
            for (num::idx column = 0; column < rows(); ++column) {
                y[row] += exact_inverse_(row, column) * x[column];
            }
        }
    }

    DiffusionGenerator<graph_type> fine_generator_;
    double shift_;
    std::size_t pre_sweeps_;
    std::size_t post_sweeps_;
    double jacobi_weight_;
    std::size_t coarse_cycles_;
    PopulationSmoother smoother_;
    std::unique_ptr<PopulationTransfer<D, Weight, Index>> transfer_;
    std::unique_ptr<TwoLevelPopulationMultigrid> coarse_level_;
    num::mat exact_inverse_;
    mutable num::vec rhs_;
    mutable num::vec solution_;
    mutable num::vec system_image_;
    mutable num::vec generator_image_;
    mutable num::vec residual_;
    mutable num::vec correction_;
    mutable num::vec coarse_rhs_;
    mutable num::vec coarse_solution_;
    mutable num::vec coarse_system_image_;
    mutable num::vec coarse_residual_;
    mutable num::vec coarse_correction_;
    mutable std::vector<BlockPartition> block_partitions_;
};

/// Preferred name for the recursive hierarchy. The older two-level name remains
/// available because it was part of the first experimental API.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
using PopulationMultigrid = TwoLevelPopulationMultigrid<D, Weight, Index>;

/// Block preconditioner applying one population multigrid V-cycle independently
/// to every contiguous sector of a reaction state space. Reaction couplings are
/// intentionally left to the outer nonsymmetric Krylov iteration.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
class SectorDiffusionMultigrid {
  public:
    using graph_type = num::structured_grid_graph<D, Weight, Index>;
    using domain_type = num::vec;
    using codomain_type = num::vec;
    using math_laws = num::math::type_list<num::law::linear_map>;

    SectorDiffusionMultigrid(const VoxelDomain<graph_type> &domain,
                             const ReactionStateSpace &state_space, double shift,
                             std::size_t pre_sweeps = 2, std::size_t post_sweeps = 2,
                             double jacobi_weight = 2.0 / 3.0,
                             PopulationSmoother smoother = PopulationSmoother::weighted_jacobi)
        : sector_offsets_(state_space.sector_offsets()) {
        levels_.reserve(state_space.n_sectors());
        for (const auto &totals : state_space.sector_totals()) {
            levels_.emplace_back(domain, totals, shift, pre_sweeps, post_sweeps, jacobi_weight, 4,
                                 32, 1, 0, smoother);
        }
    }

    [[nodiscard]] num::idx rows() const noexcept { return sector_offsets_.back(); }
    [[nodiscard]] num::idx cols() const noexcept { return rows(); }
    [[nodiscard]] std::size_t n_sectors() const noexcept { return levels_.size(); }
    [[nodiscard]] std::size_t max_hierarchy_levels() const noexcept {
        std::size_t result = 0;
        for (const auto &level : levels_)
            result = std::max(result, level.n_levels());
        return result;
    }

    void apply(const num::vec &x, num::vec &y) const {
        if (x.size() != rows() || y.size() != rows()) {
            throw std::invalid_argument("RDME sector multigrid vector size mismatch");
        }
        for (std::size_t sector = 0; sector < levels_.size(); ++sector) {
            const num::idx offset = sector_offsets_[sector];
            const num::idx size = sector_offsets_[sector + 1] - offset;
            levels_[sector].apply_block(std::span<const double>(x.data() + offset, size),
                                        std::span<double>(y.data() + offset, size));
        }
    }

  private:
    num::array<num::idx> sector_offsets_;
    num::array<TwoLevelPopulationMultigrid<D, Weight, Index>> levels_;
};

template <std::size_t D, typename Weight, std::integral Index>
TwoLevelPopulationMultigrid(VoxelDomain<num::structured_grid_graph<D, Weight, Index>>,
                            num::array<int>, double, std::size_t, std::size_t, double)
    -> TwoLevelPopulationMultigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
TwoLevelPopulationMultigrid(VoxelDomain<num::structured_grid_graph<D, Weight, Index>>,
                            num::array<int>, double)
    -> TwoLevelPopulationMultigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
SectorDiffusionMultigrid(const VoxelDomain<num::structured_grid_graph<D, Weight, Index>> &,
                         const ReactionStateSpace &, double, std::size_t, std::size_t, double)
    -> SectorDiffusionMultigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
SectorDiffusionMultigrid(const VoxelDomain<num::structured_grid_graph<D, Weight, Index>> &,
                         const ReactionStateSpace &, double, std::size_t, std::size_t, double,
                         PopulationSmoother) -> SectorDiffusionMultigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
SectorDiffusionMultigrid(const VoxelDomain<num::structured_grid_graph<D, Weight, Index>> &,
                         const ReactionStateSpace &, double)
    -> SectorDiffusionMultigrid<D, Weight, Index>;

} // namespace rdme
