/// @file rdme/population_multigrid.hpp
/// @brief Population-conserving geometric multigrid for structured-grid RDME diffusion.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
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
#include <stdexcept>
#include <utility>

namespace rdme {

/// Lift one factor-two Cartesian voxel aggregation to a fixed-population state space.
///
/// Restriction maps every fine configuration to its aggregate coarse configuration.
/// Prolongation distributes each coarse population independently and uniformly among
/// the fine children. Consequently R P = I, both maps preserve total probability,
/// and neither map changes a species total.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
class population_transfer {
  public:
    using graph_type = num::structured_grid_graph<D, Weight, Index>;

    population_transfer(const graph_type &fine_graph, const diffusion_state_space &fine_space,
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
    [[nodiscard]] const diffusion_state_space &coarse_state_space() const noexcept {
        return coarse_space_;
    }
    [[nodiscard]] const num::array<num::idx> &fine_to_coarse() const noexcept {
        return fine_to_coarse_;
    }
    [[nodiscard]] const num::array<double> &prolongation_weights() const noexcept {
        return prolongation_weights_;
    }
    [[nodiscard]] const num::array<Index> &voxel_parent() const noexcept { return voxel_parent_; }

    void restrict(num::view<const double> fine, num::view<double> coarse) const {
        check_dimensions(fine.size(), coarse.size());
        std::fill(coarse.begin(), coarse.end(), 0.0);
        for (num::idx state = 0; state < fine_size(); ++state) {
            coarse[fine_to_coarse_[state]] += fine[state];
        }
    }

    void prolong(num::view<const double> coarse, num::view<double> fine) const {
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

    void build(const graph_type &fine_graph, const diffusion_state_space &fine_space) {
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
            const configuration &fine_configuration = fine_space[fine_state];
            configuration coarse_configuration(coarse_layout.size(), 0);
            for (std::size_t fine_voxel = 0; fine_voxel < fine_layout.voxel_count(); ++fine_voxel) {
                const std::size_t coarse_voxel =
                    static_cast<std::size_t>(voxel_parent_[fine_voxel]);
                for (std::size_t species = 0; species < fine_layout.species_count(); ++species) {
                    coarse_configuration[coarse_layout.index(coarse_voxel, species)] +=
                        fine_configuration[fine_layout.index(fine_voxel, species)];
                }
            }

            const num::idx coarse_state = coarse_space_.find(coarse_configuration);
            if (coarse_state == diffusion_state_space::npos) {
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
    diffusion_state_space coarse_space_;
    num::array<Index> voxel_parent_;
    num::array<num::idx> fine_to_coarse_;
    num::array<double> prolongation_weights_;
};

enum class population_smoother { weighted_jacobi, aggregate_block };

/// A recursive geometric V-cycle for a shifted fixed-sector diffusion operator.
///
/// Intermediate levels are matrix-free; only a terminal configuration space below
/// `coarse_max_states` is assembled and factored. On a uniform factor-two grid, doubling
/// diffusivity on every coarse level is exactly the conservative Petrov--Galerkin operator R A P.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
class population_multigrid {
  public:
    using graph_type = num::structured_grid_graph<D, Weight, Index>;
    using domain_type = num::vec<double>;
    using codomain_type = num::vec<double>;

    population_multigrid(voxel_domain<graph_type> domain, num::array<int> species_totals,
                         double shift, std::size_t pre_sweeps = 2, std::size_t post_sweeps = 2,
                         double jacobi_weight = 2.0 / 3.0, num::idx coarse_max_states = 4,
                         std::size_t max_levels = 32, std::size_t coarse_cycles = 1,
                         std::size_t sweep_growth = 0,
                         population_smoother smoother = population_smoother::weighted_jacobi)
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
        if (rows() > coarse_max_states && max_levels > 1 &&
            can_coarsen(fine_generator_.domain().graph())) {
            transfer_ = std::make_unique<population_transfer<D, Weight, Index>>(
                fine_generator_.domain().graph(), fine_generator_.state_space(), species_totals);
            auto coarse_diffusivities = fine_generator_.domain().diffusivities();
            for (double &diffusivity : coarse_diffusivities)
                diffusivity *= 2.0;
            voxel_domain<graph_type> coarse_domain(transfer_->coarse_graph(),
                                                   std::move(coarse_diffusivities));
            coarse_level_ = std::make_unique<population_multigrid>(
                std::move(coarse_domain), std::move(species_totals), shift_,
                pre_sweeps_ + sweep_growth, post_sweeps_ + sweep_growth, jacobi_weight_,
                coarse_max_states, max_levels - 1, coarse_cycles_, sweep_growth, smoother_);
            coarse_rhs_ = num::vec<double>(transfer_->coarse_size(), 0.0);
            coarse_solution_ = num::vec<double>(transfer_->coarse_size(), 0.0);
            coarse_system_image_ = num::vec<double>(transfer_->coarse_size(), 0.0);
            coarse_residual_ = num::vec<double>(transfer_->coarse_size(), 0.0);
            coarse_correction_ = num::vec<double>(transfer_->coarse_size(), 0.0);
            if (smoother_ == population_smoother::aggregate_block)
                build_aggregate_blocks();
        } else {
            terminal_factor_ = factor_terminal_system();
        }
    }

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
    [[nodiscard]] const population_multigrid *coarse_level() const noexcept {
        return coarse_level_.get();
    }
    [[nodiscard]] const diffusion_generator<graph_type> &fine_generator() const noexcept {
        return fine_generator_;
    }
    [[nodiscard]] const population_transfer<D, Weight, Index> &transfer() const {
        if (!transfer_)
            throw std::logic_error("terminal RDME multigrid level has no transfer");
        return *transfer_;
    }

    void apply_system(const num::vec<double> &x, num::vec<double> &y) const {
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
    void relax(const num::vec<double> &rhs, num::vec<double> &solution, std::size_t sweeps = 1,
               bool reverse = false) const {
        if (rhs.size() != rows() || solution.size() != rows()) {
            throw std::invalid_argument("RDME smoother vector size mismatch");
        }
        rhs_ = rhs;
        solution_ = solution;
        smooth(sweeps, reverse);
        solution = solution_;
    }

    void apply(const num::vec<double> &x, num::vec<double> &y) const { apply_block(x, y); }

    void apply_block(num::view<const double> x, num::view<double> y) const {
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
            num::solve(terminal_factor_, residual_, correction_);
        }
        for (num::idx state = 0; state < rows(); ++state)
            solution_[state] += correction_[state];

        smooth(post_sweeps_, true);
        std::copy(solution_.begin(), solution_.end(), y.begin());
    }

  private:
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

    [[nodiscard]] num::lu_result<double> factor_terminal_system() const {
        num::mat<double> matrix(rows(), rows(), 0.0);
        for (num::idx source = 0; source < rows(); ++source) {
            matrix(source, source) = shift_ + fine_generator_.exit_rates()[source];
            for (num::idx entry = fine_generator_.source_offsets()[source];
                 entry < fine_generator_.source_offsets()[source + 1]; ++entry) {
                const auto &jump = fine_generator_.transitions()[entry];
                matrix(jump.destination, source) -= jump.rate;
            }
        }
        auto factor = num::lu(matrix);
        if (factor.singular)
            throw std::runtime_error("RDME multigrid terminal operator is singular");
        return factor;
    }

    struct block {
        num::array<num::idx> states;
        num::lu_result<double> factor;
        mutable num::vec<double> rhs;
        mutable num::vec<double> correction;
    };

    using block_partition = num::array<block>;

    void build_aggregate_blocks() {
        const auto &space = fine_generator_.state_space();
        const auto &layout = space.layout();
        const auto &voxel_parent = transfer_->voxel_parent();
        const std::size_t aggregate_count = transfer_->coarse_state_space().layout().voxel_count();
        block_partitions_.resize(aggregate_count);

        for (std::size_t aggregate = 0; aggregate < aggregate_count; ++aggregate) {
            num::unordered_map<configuration, num::array<num::idx>, markovkit::state_hash> groups;
            for (num::idx state = 0; state < rows(); ++state) {
                configuration signature = space[state];
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
            num::array<num::idx> local_index(rows(), diffusion_state_space::npos);
            for (auto &[signature, states] : groups) {
                (void)signature;
                if (states.size() == 1)
                    continue;
                for (num::idx local = 0; local < states.size(); ++local)
                    local_index[states[local]] = local;

                num::mat<double> matrix(states.size(), states.size(), 0.0);
                for (num::idx local_source = 0; local_source < states.size(); ++local_source) {
                    const num::idx source = states[local_source];
                    matrix(local_source, local_source) =
                        shift_ + fine_generator_.exit_rates()[source];
                    for (num::idx entry = fine_generator_.source_offsets()[source];
                         entry < fine_generator_.source_offsets()[source + 1]; ++entry) {
                        const auto &transition = fine_generator_.transitions()[entry];
                        const num::idx local_destination = local_index[transition.destination];
                        if (local_destination != diffusion_state_space::npos)
                            matrix(local_destination, local_source) -= transition.rate;
                    }
                }
                num::lu_result<double> factor = num::lu(matrix);
                if (factor.singular)
                    throw std::runtime_error("RDME aggregate smoother block is singular");
                const num::idx block_size = states.size();
                partition.push_back({std::move(states), std::move(factor),
                                     num::vec<double>(block_size, 0.0),
                                     num::vec<double>(block_size, 0.0)});
                for (num::idx state : partition.back().states)
                    local_index[state] = diffusion_state_space::npos;
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
                for (block &block : block_partitions_[aggregate]) {
                    for (num::idx local = 0; local < block.states.size(); ++local) {
                        const num::idx state = block.states[local];
                        block.rhs[local] = residual_[state];
                    }
                    num::solve(block.factor, block.rhs, block.correction);
                }
                for (const block &block : block_partitions_[aggregate]) {
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
        if (smoother_ == population_smoother::aggregate_block && !block_partitions_.empty())
            smooth_blocks(sweeps, reverse);
        else
            smooth_jacobi(sweeps);
    }

    diffusion_generator<graph_type> fine_generator_;
    double shift_;
    std::size_t pre_sweeps_;
    std::size_t post_sweeps_;
    double jacobi_weight_;
    std::size_t coarse_cycles_;
    population_smoother smoother_;
    std::unique_ptr<population_transfer<D, Weight, Index>> transfer_;
    std::unique_ptr<population_multigrid> coarse_level_;
    num::lu_result<double> terminal_factor_;
    mutable num::vec<double> rhs_;
    mutable num::vec<double> solution_;
    mutable num::vec<double> system_image_;
    mutable num::vec<double> generator_image_;
    mutable num::vec<double> residual_;
    mutable num::vec<double> correction_;
    mutable num::vec<double> coarse_rhs_;
    mutable num::vec<double> coarse_solution_;
    mutable num::vec<double> coarse_system_image_;
    mutable num::vec<double> coarse_residual_;
    mutable num::vec<double> coarse_correction_;
    mutable num::array<block_partition> block_partitions_;
};

/// Block preconditioner applying one population multigrid V-cycle independently
/// to every contiguous sector of a reaction state space. Reaction couplings are
/// intentionally left to the outer nonsymmetric Krylov iteration.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
class sector_diffusion_multigrid {
  public:
    using graph_type = num::structured_grid_graph<D, Weight, Index>;
    using domain_type = num::vec<double>;
    using codomain_type = num::vec<double>;

    sector_diffusion_multigrid(const voxel_domain<graph_type> &domain,
                               const reaction_state_space &state_space, double shift,
                               std::size_t pre_sweeps = 2, std::size_t post_sweeps = 2,
                               double jacobi_weight = 2.0 / 3.0,
                               population_smoother smoother = population_smoother::weighted_jacobi)
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

    void apply(const num::vec<double> &x, num::vec<double> &y) const {
        if (x.size() != rows() || y.size() != rows()) {
            throw std::invalid_argument("RDME sector multigrid vector size mismatch");
        }
        for (std::size_t sector = 0; sector < levels_.size(); ++sector) {
            const num::idx offset = sector_offsets_[sector];
            const num::idx size = sector_offsets_[sector + 1] - offset;
            levels_[sector].apply_block(num::view<const double>(x.data() + offset, size),
                                        num::view<double>(y.data() + offset, size));
        }
    }

  private:
    num::array<num::idx> sector_offsets_;
    num::array<population_multigrid<D, Weight, Index>> levels_;
};

template <std::size_t D, typename Weight, std::integral Index>
population_multigrid(voxel_domain<num::structured_grid_graph<D, Weight, Index>>, num::array<int>,
                     double, std::size_t, std::size_t, double)
    -> population_multigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
population_multigrid(voxel_domain<num::structured_grid_graph<D, Weight, Index>>, num::array<int>,
                     double) -> population_multigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
sector_diffusion_multigrid(const voxel_domain<num::structured_grid_graph<D, Weight, Index>> &,
                           const reaction_state_space &, double, std::size_t, std::size_t, double)
    -> sector_diffusion_multigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
sector_diffusion_multigrid(const voxel_domain<num::structured_grid_graph<D, Weight, Index>> &,
                           const reaction_state_space &, double, std::size_t, std::size_t, double,
                           population_smoother) -> sector_diffusion_multigrid<D, Weight, Index>;

template <std::size_t D, typename Weight, std::integral Index>
sector_diffusion_multigrid(const voxel_domain<num::structured_grid_graph<D, Weight, Index>> &,
                           const reaction_state_space &, double)
    -> sector_diffusion_multigrid<D, Weight, Index>;

// Original experimental name.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
using two_level_population_multigrid = population_multigrid<D, Weight, Index>;

} // namespace rdme
