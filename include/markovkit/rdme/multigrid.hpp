// geometric multigrid for the shifted diffusion operator on population sectors of a grid
#pragma once

#include "container/matrix.hpp"
#include "linear/factorization/lu.hpp"
#include "markovkit/generator.hpp"
#include "markovkit/rdme/population_space.hpp"
#include "markovkit/rdme/reaction_diffusion.hpp"
#include "structures/graph/structured_grid.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace markovkit {

template <std::size_t D>
using grid = num::structured_grid_graph<D>;

// halve every axis longer than one cell, or nothing when an axis is odd or none is long
template <std::size_t D>
std::optional<grid<D>> coarsen(const grid<D> &fine) {
    auto dimensions = fine.dimensions();
    auto spacing = fine.spacing();
    bool reduced = false;
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (dimensions[axis] == 1) {
            continue;
        }
        if (dimensions[axis] % 2 != 0) {
            return std::nullopt;
        }
        dimensions[axis] /= 2;
        spacing[axis] *= 2.0;
        reduced = true;
    }
    return reduced ? std::optional<grid<D>>(grid<D>(dimensions, spacing, fine.boundaries()))
                   : std::nullopt;
}

// Restriction adds each fine state into the coarse state with the same count in each coarse
// voxel; prolongation splits each coarse state multinomially among the fine voxels it covers.
// Both preserve probability and species totals, and R P = I.
template <std::size_t D>
class population_transfer {
  public:
    population_transfer(const grid<D> &fine_grid, const grid<D> &coarse_grid,
                        const population_sector &fine, const population_sector &coarse)
        : parent_(fine.n_voxels()), coarse_of_(fine.size()), weight_(fine.size()),
          coarse_size_(coarse.size()) {
        const idx S = fine.n_species();
        array<idx> children(coarse.n_voxels(), 0);
        for (idx v = 0; v < fine.n_voxels(); ++v) {
            auto coordinate = fine_grid.coordinate(v);
            for (std::size_t axis = 0; axis < D; ++axis) {
                if (fine_grid.dimensions()[axis] > 1) {
                    coordinate[axis] /= 2;
                }
            }
            parent_[v] = coarse_grid.vertex(coordinate);
            ++children[parent_[v]];
        }

        // weight = multinomial probability of the fine state given its coarse state
        array<real> total(coarse.size(), 0.0);
        for (idx i = 0; i < fine.size(); ++i) {
            state c(coarse.n_voxels() * S, 0);
            real log_weight = 0.0;
            for (idx v = 0; v < fine.n_voxels(); ++v) {
                for (idx s = 0; s < S; ++s) {
                    const int n = fine[i][(v * S) + s];
                    c[(parent_[v] * S) + s] += n;
                    log_weight -= std::lgamma(n + 1.0);
                }
            }
            for (idx a = 0; a < coarse.n_voxels(); ++a) {
                for (idx s = 0; s < S; ++s) {
                    const int n = c[(a * S) + s];
                    log_weight += std::lgamma(n + 1.0) - (n * std::log(real(children[a])));
                }
            }
            coarse_of_[i] = *coarse.find(c);
            weight_[i] = std::exp(log_weight);
            total[coarse_of_[i]] += weight_[i];
        }
        for (idx i = 0; i < fine.size(); ++i) {
            weight_[i] /= total[coarse_of_[i]];
        }
    }

    [[nodiscard]] idx fine_size() const { return coarse_of_.size(); }
    [[nodiscard]] idx coarse_size() const { return coarse_size_; }

    // the coarse voxel containing each fine voxel
    [[nodiscard]] const array<idx> &parent() const { return parent_; }

    void restrict(view<const real> fine, view<real> coarse) const {
        std::fill(coarse.begin(), coarse.end(), 0.0);
        for (idx i = 0; i < fine_size(); ++i) {
            coarse[coarse_of_[i]] += fine[i];
        }
    }

    void prolong(view<const real> coarse, view<real> fine) const {
        for (idx i = 0; i < fine_size(); ++i) {
            fine[i] = weight_[i] * coarse[coarse_of_[i]];
        }
    }

  private:
    array<idx> parent_;
    array<idx> coarse_of_;
    array<real> weight_;
    idx coarse_size_;
};

enum class relaxation : std::uint8_t {
    jacobi,         // weighted point Jacobi
    aggregate_block // exact solves on states that differ only inside one coarse voxel
};

struct multigrid_options {
    int pre_sweeps = 2;
    int post_sweeps = 2;
    real jacobi_weight = 2.0 / 3.0;
    idx coarsest_size = 4; // factor a level directly once it has at most this many states
    int max_levels = 32;
    int coarse_cycles = 1; // 1 for a V-cycle, 2 for a W-cycle
    int sweep_growth = 0;  // extra sweeps on each coarser level
    relaxation smoother = relaxation::jacobi;
};

// One multigrid cycle for A = shift I - Q on one population sector, as a preconditioner.
// On a factor-two grid, doubling the diffusivity gives exactly the Petrov-Galerkin coarse
// operator R A P, so every level is matrix-free except the coarsest, which is factored.
template <std::size_t D>
class population_multigrid {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    population_multigrid(const voxel_domain<grid<D>> &domain, const array<int> &totals,
                         real shift, multigrid_options options = {})
        : space_(domain.n_voxels(), totals),
          Q_(space_, reaction_diffusion<grid<D>>{domain, {}}), shift_(shift), options_(options),
          rhs_(rows()), x_(rows()), r_(rows()), e_(rows()) {
        if (!(shift > 0.0)) {
            throw std::invalid_argument("population_multigrid: shift must be positive");
        }
        auto coarse_grid = coarsen(domain.graph);
        if (rows() <= options.coarsest_size || options.max_levels <= 1 || !coarse_grid) {
            array<idx> all(rows());
            std::iota(all.begin(), all.end(), 0);
            terminal_ = factor(all, all);
            return;
        }

        array<real> doubled = domain.diffusivities;
        for (real &d : doubled) {
            d *= 2.0;
        }
        multigrid_options coarse_options = options;
        coarse_options.max_levels -= 1;
        coarse_options.pre_sweeps += options.sweep_growth;
        coarse_options.post_sweeps += options.sweep_growth;
        coarse_ = std::make_unique<population_multigrid>(
            voxel_domain<grid<D>>(*coarse_grid, std::move(doubled)), totals, shift,
            coarse_options);
        transfer_ = std::make_unique<population_transfer<D>>(domain.graph, *coarse_grid, space_,
                                                             coarse_->space_);
        const idx n = coarse_->rows();
        coarse_rhs_ = coarse_x_ = coarse_r_ = coarse_e_ = vec<real>(n, 0.0);
        if (options.smoother == relaxation::aggregate_block) {
            build_blocks();
        }
    }

    [[nodiscard]] idx rows() const { return space_.size(); }
    [[nodiscard]] idx cols() const { return space_.size(); }
    [[nodiscard]] real shift() const { return shift_; }
    [[nodiscard]] const population_sector &space() const { return space_; }
    [[nodiscard]] const generator &fine_generator() const { return Q_; }
    [[nodiscard]] const population_multigrid *coarse_level() const { return coarse_.get(); }
    [[nodiscard]] const population_transfer<D> &transfer() const { return *transfer_; }

    [[nodiscard]] idx n_levels() const { return coarse_ ? 1 + coarse_->n_levels() : 1; }
    [[nodiscard]] real grid_complexity() const { return real(total_states()) / real(rows()); }
    [[nodiscard]] real operator_complexity() const {
        return real(total_nonzeros()) / real(rows() + Q_.n_jumps());
    }

    // y = A x
    void apply_system(view<const real> x, view<real> y) const { Q_.apply_shifted(shift_, x, y); }

    // one cycle from a zero initial guess: x ~ A^{-1} b
    void apply(const vec<real> &b, vec<real> &x) const { cycle(b, x); }

    void cycle(view<const real> b, view<real> x) const {
        std::copy(b.begin(), b.end(), rhs_.begin());
        std::fill(x_.begin(), x_.end(), 0.0);
        smooth(options_.pre_sweeps, false);
        residual();
        if (coarse_) {
            transfer_->restrict(r_, coarse_rhs_);
            std::fill(coarse_x_.begin(), coarse_x_.end(), 0.0);
            for (int c = 0; c < options_.coarse_cycles; ++c) {
                coarse_->apply_system(coarse_x_, coarse_r_);
                for (idx i = 0; i < coarse_r_.size(); ++i) {
                    coarse_r_[i] = coarse_rhs_[i] - coarse_r_[i];
                }
                coarse_->cycle(coarse_r_, coarse_e_);
                for (idx i = 0; i < coarse_x_.size(); ++i) {
                    coarse_x_[i] += coarse_e_[i];
                }
            }
            transfer_->prolong(coarse_x_, e_);
        } else {
            num::solve(terminal_, r_, e_);
        }
        for (idx i = 0; i < rows(); ++i) {
            x_[i] += e_[i];
        }
        smooth(options_.post_sweeps, true);
        std::copy(x_.begin(), x_.end(), x.begin());
    }

    // `sweeps` smoother sweeps on A x = rhs, starting from x
    void relax(const vec<real> &rhs, vec<real> &x, int sweeps = 1, bool reverse = false) const {
        rhs_ = rhs;
        x_ = x;
        smooth(sweeps, reverse);
        x = x_;
    }

  private:
    struct block {
        array<idx> states;
        num::lu_result<real> lu;
        mutable vec<real> rhs, e;
    };

    // A restricted to `states`, dense and factored; position[j] is the place of state j in
    // `states`, or at least states.size() when j is not among them
    [[nodiscard]] num::lu_result<real> factor(const array<idx> &states,
                                              const array<idx> &position) const {
        const idx n = states.size();
        num::mat<real> A(n, n, 0.0);
        for (idx k = 0; k < n; ++k) {
            A(k, k) = shift_ + Q_.exit_rates()[states[k]];
            for (const auto &e : Q_.jumps(states[k])) {
                if (position[e.to] < n) {
                    A(position[e.to], k) -= e.rate;
                }
            }
        }
        auto lu = num::lu(A);
        if (lu.singular) {
            throw std::runtime_error("population_multigrid: singular block");
        }
        return lu;
    }

    // group the states that agree outside each coarse voxel; each group is one block
    void build_blocks() {
        const idx S = space_.n_species();
        const auto &parent = transfer_->parent();
        array<idx> position(rows(), rows());
        blocks_.resize(coarse_->space().n_voxels());
        for (idx a = 0; a < blocks_.size(); ++a) {
            num::unordered_map<state, array<idx>, state_hash> groups;
            for (idx i = 0; i < rows(); ++i) {
                state outside = space_[i];
                for (idx v = 0; v < space_.n_voxels(); ++v) {
                    if (parent[v] == a) {
                        std::fill_n(outside.begin() + (v * S), S, 0);
                    }
                }
                groups[std::move(outside)].push_back(i);
            }
            for (auto &[outside, members] : groups) {
                if (members.size() == 1) {
                    continue;
                }
                for (idx k = 0; k < members.size(); ++k) {
                    position[members[k]] = k;
                }
                auto lu = factor(members, position);
                for (idx i : members) {
                    position[i] = rows();
                }
                const idx n = members.size();
                blocks_[a].push_back({std::move(members), std::move(lu), vec<real>(n, 0.0),
                                      vec<real>(n, 0.0)});
            }
        }
    }

    // r = rhs - A x
    void residual() const {
        apply_system(x_, r_);
        for (idx i = 0; i < rows(); ++i) {
            r_[i] = rhs_[i] - r_[i];
        }
    }

    void smooth(int sweeps, bool reverse) const {
        if (blocks_.empty()) {
            for (int sweep = 0; sweep < sweeps; ++sweep) {
                residual();
                for (idx i = 0; i < rows(); ++i) {
                    x_[i] += options_.jacobi_weight * r_[i] / (shift_ + Q_.exit_rates()[i]);
                }
            }
            return;
        }
        // block Gauss-Seidel over aggregates, updating the residual only where it changes
        residual();
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (idx step = 0; step < blocks_.size(); ++step) {
                const auto &aggregate = blocks_[reverse ? blocks_.size() - 1 - step : step];
                for (const block &b : aggregate) {
                    for (idx k = 0; k < b.states.size(); ++k) {
                        b.rhs[k] = r_[b.states[k]];
                    }
                    num::solve(b.lu, b.rhs, b.e);
                }
                for (const block &b : aggregate) {
                    for (idx k = 0; k < b.states.size(); ++k) {
                        const idx j = b.states[k];
                        x_[j] += b.e[k];
                        r_[j] -= (shift_ + Q_.exit_rates()[j]) * b.e[k];
                        for (const auto &e : Q_.jumps(j)) {
                            r_[e.to] += e.rate * b.e[k];
                        }
                    }
                }
            }
        }
    }

    [[nodiscard]] idx total_states() const {
        return rows() + (coarse_ ? coarse_->total_states() : 0);
    }
    [[nodiscard]] idx total_nonzeros() const {
        return rows() + Q_.n_jumps() + (coarse_ ? coarse_->total_nonzeros() : 0);
    }

    population_sector space_;
    generator Q_;
    real shift_;
    multigrid_options options_;
    std::unique_ptr<population_multigrid> coarse_;
    std::unique_ptr<population_transfer<D>> transfer_;
    num::lu_result<real> terminal_;
    array<array<block>> blocks_;
    mutable vec<real> rhs_, x_, r_, e_;
    mutable vec<real> coarse_rhs_, coarse_x_, coarse_r_, coarse_e_;
};

template <std::size_t D>
population_multigrid(voxel_domain<grid<D>>, array<int>, real) -> population_multigrid<D>;
template <std::size_t D>
population_multigrid(voxel_domain<grid<D>>, array<int>, real, multigrid_options)
    -> population_multigrid<D>;

// one population multigrid cycle per sector of a box, applied block by block;
// the reactions coupling sectors are left to the outer Krylov solver
template <std::size_t D>
class sector_multigrid {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    sector_multigrid(const voxel_domain<grid<D>> &domain, const population_box &box, real shift,
                     multigrid_options options = {}) {
        for (idx k = 0; k < box.n_sectors(); ++k) {
            levels_.emplace_back(domain, box.sector_totals(k), shift, options);
            begin_.push_back(box.sector_begin(k));
        }
        begin_.push_back(box.size());
    }

    [[nodiscard]] idx rows() const { return begin_.back(); }
    [[nodiscard]] idx cols() const { return begin_.back(); }
    [[nodiscard]] idx n_sectors() const { return levels_.size(); }
    [[nodiscard]] idx max_levels() const {
        idx result = 0;
        for (const auto &level : levels_) {
            result = std::max(result, level.n_levels());
        }
        return result;
    }

    void apply(const vec<real> &b, vec<real> &x) const {
        for (idx k = 0; k < levels_.size(); ++k) {
            const idx n = begin_[k + 1] - begin_[k];
            levels_[k].cycle(view<const real>(b.data() + begin_[k], n),
                             view<real>(x.data() + begin_[k], n));
        }
    }

  private:
    array<population_multigrid<D>> levels_;
    array<idx> begin_;
};

template <std::size_t D>
sector_multigrid(voxel_domain<grid<D>>, const population_box &, real) -> sector_multigrid<D>;
template <std::size_t D>
sector_multigrid(voxel_domain<grid<D>>, const population_box &, real, multigrid_options)
    -> sector_multigrid<D>;

} // namespace markovkit
