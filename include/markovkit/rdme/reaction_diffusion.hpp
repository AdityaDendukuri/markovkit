// the reaction-diffusion master equation as a markov_model
#pragma once

#include "markovkit/chain.hpp"
#include <cmath>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <utility>

namespace markovkit {

// a graph of voxels whose edge weights are face transmissibilities
template <typename Graph>
concept voxel_graph = requires(const Graph &graph, typename Graph::index_type v) {
    {graph.n_vertices()}->std::convertible_to<idx>;
    {graph.neighbors(v).begin()->to}->std::convertible_to<idx>;
    {graph.neighbors(v).begin()->weight}->std::convertible_to<real>;
};

// RDME geometry: a molecule of species s jumps from voxel i along edge e at rate
// diffusivities[s] * e.weight / volumes[i]
template <voxel_graph Graph>
struct voxel_domain {
    Graph graph;
    array<real> volumes;
    array<real> diffusivities;

    voxel_domain(Graph g, array<real> voxel_volumes, array<real> species_diffusivities)
        : graph(std::move(g)), volumes(std::move(voxel_volumes)),
          diffusivities(std::move(species_diffusivities)) {
        if (volumes.size() != n_voxels()) {
            throw std::invalid_argument("voxel_domain: one volume per voxel");
        }
        for (real v : volumes) {
            if (!(v > 0.0) || !std::isfinite(v)) {
                throw std::invalid_argument("voxel_domain: volumes must be positive");
            }
        }
    }

    // uniform volumes, from a grid that knows its cell volume
    voxel_domain(Graph g, array<real> species_diffusivities) requires requires(const Graph &grid) {
        grid.cell_volume();
    }
        : voxel_domain(g, array<real>(g.n_vertices(), static_cast<real>(g.cell_volume())),
                       std::move(species_diffusivities)) {}

    [[nodiscard]] idx n_voxels() const { return graph.n_vertices(); }
    [[nodiscard]] idx n_species() const { return diffusivities.size(); }

    template <typename Edge>
    [[nodiscard]] real jump_rate(idx species, idx from, const Edge &edge) const {
        return diffusivities[species] * static_cast<real>(edge.weight) / volumes[from];
    }
};

template <typename Graph>
voxel_domain(Graph, array<real>, array<real>) -> voxel_domain<Graph>;
template <typename Graph>
voxel_domain(Graph, array<real>) -> voxel_domain<Graph>;

// a reaction inside one voxel; its rate sees that voxel's counts and volume
struct local_reaction {
    state change;
    std::function<real(view<const int> counts, real volume)> rate;
};

// every local reaction in every voxel, plus diffusion between neighboring voxels.
// States are voxel-major: x[index(v, s)] is the count of species s in voxel v.
template <voxel_graph Graph>
struct reaction_diffusion {
    voxel_domain<Graph> domain;
    array<local_reaction> reactions;

    [[nodiscard]] idx n_voxels() const { return domain.n_voxels(); }
    [[nodiscard]] idx n_species() const { return domain.n_species(); }
    [[nodiscard]] idx index(idx voxel, idx species) const {
        return (voxel * n_species()) + species;
    }

    [[nodiscard]] array<transition> transitions(const state &x) const {
        array<transition> result;
        const idx S = n_species();
        for (idx v = 0; v < n_voxels(); ++v) {
            for (const auto &edge : domain.graph.neighbors(v)) {
                for (idx s = 0; s < S; ++s) {
                    const int n = x[index(v, s)];
                    if (n == 0) {
                        continue;
                    }
                    state y = x;
                    --y[index(v, s)];
                    ++y[index(static_cast<idx>(edge.to), s)];
                    result.push_back({std::move(y), n * domain.jump_rate(s, v, edge)});
                }
            }
        }
        for (idx v = 0; v < n_voxels(); ++v) {
            const view<const int> counts(x.data() + index(v, 0), S);
            for (const local_reaction &r : reactions) {
                const real rate = r.rate(counts, domain.volumes[v]);
                if (!(rate > 0.0)) {
                    continue;
                }
                state y = x;
                bool valid = true;
                for (idx s = 0; s < S; ++s) {
                    y[index(v, s)] += r.change[s];
                    valid = valid && y[index(v, s)] >= 0;
                }
                if (valid) {
                    result.push_back({std::move(y), rate});
                }
            }
        }
        return result;
    }
};

template <typename Graph>
reaction_diffusion(voxel_domain<Graph>, array<local_reaction>) -> reaction_diffusion<Graph>;

} // namespace markovkit
