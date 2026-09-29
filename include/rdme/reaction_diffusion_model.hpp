/// @file rdme/reaction_diffusion_model.hpp
/// @brief Local reactions coupled to a spatial RDME diffusion domain.
#pragma once

#include "core/types.hpp"
#include "rdme/voxel_domain.hpp"
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>

namespace rdme {

using local_propensity =
    std::function<double(num::view<const int>, num::view<const double>, double, double)>;

/// One reaction acting independently in every voxel.
///
/// The propensity receives local molecule counts, model parameters, voxel
/// volume, and time. Stoichiometry has one entry per species.
struct local_reaction {
    num::array<int> change;
    local_propensity propensity;
};

template <voxel_graph Graph>
class reaction_diffusion_model {
  public:
    reaction_diffusion_model(voxel_domain<Graph> domain, num::array<local_reaction> reactions)
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

    [[nodiscard]] const voxel_domain<Graph> &domain() const noexcept { return domain_; }
    [[nodiscard]] const num::array<local_reaction> &reactions() const noexcept {
        return reactions_;
    }
    [[nodiscard]] std::size_t n_species() const noexcept { return domain_.n_species(); }
    [[nodiscard]] std::size_t n_reactions() const noexcept { return reactions_.size(); }

  private:
    voxel_domain<Graph> domain_;
    num::array<local_reaction> reactions_;
};

template <class Graph>
reaction_diffusion_model(voxel_domain<Graph>, num::array<local_reaction>)
    -> reaction_diffusion_model<Graph>;

} // namespace rdme
