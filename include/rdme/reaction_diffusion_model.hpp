/// @file rdme/reaction_diffusion_model.hpp
/// @brief Local reactions coupled to a spatial RDME diffusion domain.
#pragma once

#include "core/types.hpp"
#include "rdme/voxel_domain.hpp"
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>

namespace rdme {

using LocalPropensity =
    std::function<double(std::span<const int>, std::span<const double>, double, double)>;

/// One reaction acting independently in every voxel.
///
/// The propensity receives local molecule counts, model parameters, voxel
/// volume, and time. Stoichiometry has one entry per species.
struct LocalReaction {
    num::array<int> change;
    LocalPropensity propensity;
};

template <VoxelGraph Graph>
class ReactionDiffusionModel {
  public:
    ReactionDiffusionModel(VoxelDomain<Graph> domain, num::array<LocalReaction> reactions)
        : domain_(std::move(domain)), reactions_(std::move(reactions)) {
        for (const auto &reaction : reactions_) {
            if (reaction.change.size() != domain_.n_species()) {
                throw std::invalid_argument(
                    "RDME reaction stoichiometry must match the species count");
            }
            if (!reaction.propensity) {
                throw std::invalid_argument("RDME reactions require a propensity function");
            }
        }
    }

    [[nodiscard]] const VoxelDomain<Graph> &domain() const noexcept { return domain_; }
    [[nodiscard]] const num::array<LocalReaction> &reactions() const noexcept { return reactions_; }
    [[nodiscard]] std::size_t n_species() const noexcept { return domain_.n_species(); }
    [[nodiscard]] std::size_t n_reactions() const noexcept { return reactions_.size(); }

  private:
    VoxelDomain<Graph> domain_;
    num::array<LocalReaction> reactions_;
};

template <class Graph>
ReactionDiffusionModel(VoxelDomain<Graph>, num::array<LocalReaction>)
    -> ReactionDiffusionModel<Graph>;

} // namespace rdme
