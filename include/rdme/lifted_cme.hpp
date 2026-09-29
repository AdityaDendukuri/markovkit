/// @file rdme/lifted_cme.hpp
/// @brief Exact reference lift of an RDME into MarkovKit's CME representation.
#pragma once

#include "cme/reaction_system.hpp"
#include "core/types.hpp"
#include "rdme/state_layout.hpp"
#include "rdme/voxel_domain.hpp"
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace rdme {

/// Lift identical local reactions and graph diffusion jumps into one CME system.
///
/// This explicit event lift is intended as a correctness oracle and as a bridge
/// to the existing FSP solvers. Its event count is V*R + S*sum(degree). Large
/// RDME calculations should use the forthcoming matrix-free operator instead.
template <voxel_graph Graph>
[[nodiscard]] cme::reaction_system lift_to_cme(const voxel_domain<Graph> &domain,
                                               const cme::reaction_system &local_reactions) {
    local_reactions.validate();
    const std::size_t voxel_count = static_cast<std::size_t>(domain.n_voxels());
    const std::size_t species_count = domain.n_species();
    const state_layout layout(voxel_count, species_count);
    if (layout.size() > markovkit::state::k_max_dim) {
        throw std::invalid_argument(
            "exact RDME-to-CME lift exceeds MarkovKit's fixed-capacity reference state");
    }

    for (const auto &change : local_reactions.changes) {
        if (change.size() != species_count) {
            throw std::invalid_argument(
                "local RDME reaction stoichiometry must match the species count");
        }
    }

    cme::reaction_system result;
    std::size_t directed_edges = 0;
    for (typename Graph::index_type voxel = 0; voxel < domain.n_voxels(); ++voxel) {
        directed_edges += domain.graph().neighbors(voxel).size();
    }
    result.changes.reserve(voxel_count * local_reactions.size() + species_count * directed_edges);
    result.propensities.reserve(result.changes.capacity());

    for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
        for (std::size_t reaction = 0; reaction < local_reactions.size(); ++reaction) {
            markovkit::state lifted_change(layout.size(), 0);
            for (std::size_t species = 0; species < species_count; ++species) {
                lifted_change[layout.index(voxel, species)] =
                    local_reactions.changes[reaction][species];
            }
            result.changes.push_back(std::move(lifted_change));

            const auto local_propensity = local_reactions.propensities[reaction];
            result.propensities.push_back(
                [layout, voxel, local_propensity](const markovkit::state &state,
                                                  const num::array<double> &rates, double time) {
                    layout.validate(state);
                    markovkit::state local_state(layout.species_count(), 0);
                    for (std::size_t species = 0; species < layout.species_count(); ++species) {
                        local_state[species] = state[layout.index(voxel, species)];
                    }
                    return local_propensity(local_state, rates, time);
                });
        }
    }

    for (typename Graph::index_type source = 0; source < domain.n_voxels(); ++source) {
        for (const auto &edge : domain.graph().neighbors(source)) {
            const auto destination = static_cast<typename Graph::index_type>(edge.to);
            for (std::size_t species = 0; species < species_count; ++species) {
                markovkit::state lifted_change(layout.size(), 0);
                lifted_change[layout.index(static_cast<std::size_t>(source), species)] = -1;
                lifted_change[layout.index(static_cast<std::size_t>(destination), species)] = 1;
                result.changes.push_back(std::move(lifted_change));

                const double rate = domain.jump_rate(species, source, edge);
                const std::size_t source_index =
                    layout.index(static_cast<std::size_t>(source), species);
                result.propensities.push_back([source_index, rate](const markovkit::state &state,
                                                                   const num::array<double> &,
                                                                   double) {
                    if (state.size() <= source_index) {
                        throw std::invalid_argument(
                            "RDME state size does not match the lifted CME");
                    }
                    return rate * static_cast<double>(state[source_index]);
                });
            }
        }
    }

    return result;
}

} // namespace rdme
