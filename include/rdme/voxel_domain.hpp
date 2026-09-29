/// @file rdme/voxel_domain.hpp
/// @brief Spatial geometry and diffusion data for an RDME model.
#pragma once

#include "core/types.hpp"
#include "structures/concepts.hpp"
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace rdme {

template <class Graph>
concept voxel_graph = requires(const Graph &graph, typename Graph::index_type vertex) {
    requires num::concepts::incidence_structure<Graph, typename Graph::index_type>;
    {graph.neighbors(vertex)};
    {graph.neighbors(vertex).begin()->to}->std::convertible_to<typename Graph::index_type>;
    {graph.neighbors(vertex).begin()->weight}->std::convertible_to<double>;
};

template <class Graph>
concept uniform_cell_graph = requires(const Graph &graph) {
    {graph.cell_volume()}->std::convertible_to<double>;
};

/// Spatial graph and species diffusion constants for an RDME.
///
/// An edge weight is a geometric face transmissibility. The first-order jump
/// rate per molecule from voxel i along edge e is
/// `diffusivity[species] * e.weight / volume[i]`. The graph may be an implicit
/// structured grid or an explicitly stored mesh graph.
template <voxel_graph Graph>
class voxel_domain {
  public:
    using graph_type = Graph;
    using index_type = typename Graph::index_type;

    voxel_domain(Graph graph, num::array<double> voxel_volumes, num::array<double> diffusivities)
        : graph_(std::move(graph)), voxel_volumes_(std::move(voxel_volumes)),
          diffusivities_(std::move(diffusivities)) {
        validate();
    }

    /// Construct a uniform-volume domain from a graph that exposes cell_volume().
    voxel_domain(Graph graph, num::array<double> diffusivities) requires uniform_cell_graph<Graph>
        : graph_(std::move(graph)),
          voxel_volumes_(static_cast<std::size_t>(graph_.n_vertices()),
                         static_cast<double>(graph_.cell_volume())),
          diffusivities_(std::move(diffusivities)) {
        validate();
    }

    [[nodiscard]] const Graph &graph() const noexcept { return graph_; }
    [[nodiscard]] index_type n_voxels() const noexcept { return graph_.n_vertices(); }
    [[nodiscard]] std::size_t n_species() const noexcept { return diffusivities_.size(); }
    [[nodiscard]] const num::array<double> &voxel_volumes() const noexcept {
        return voxel_volumes_;
    }
    [[nodiscard]] const num::array<double> &diffusivities() const noexcept {
        return diffusivities_;
    }

    [[nodiscard]] double voxel_volume(index_type voxel) const {
        check_voxel(voxel);
        return voxel_volumes_[static_cast<std::size_t>(voxel)];
    }

    [[nodiscard]] double diffusivity(std::size_t species) const {
        check_species(species);
        return diffusivities_[species];
    }

    template <class Edge>
    [[nodiscard]] double jump_rate(std::size_t species, index_type source, const Edge &edge) const {
        check_species(species);
        check_voxel(source);
        check_voxel(static_cast<index_type>(edge.to));
        return diffusivities_[species] * static_cast<double>(edge.weight) /
               voxel_volumes_[static_cast<std::size_t>(source)];
    }

    [[nodiscard]] double outgoing_jump_rate(std::size_t species, index_type source) const {
        double result = 0.0;
        for (const auto &edge : graph_.neighbors(source)) {
            result += jump_rate(species, source, edge);
        }
        return result;
    }

  private:
    void validate() const {
        if (voxel_volumes_.size() != static_cast<std::size_t>(graph_.n_vertices())) {
            throw std::invalid_argument("RDME voxel volume count must match the graph");
        }
        for (double volume : voxel_volumes_) {
            if (!(volume > 0.0) || !std::isfinite(volume)) {
                throw std::invalid_argument("RDME voxel volumes must be finite and positive");
            }
        }
        for (double coefficient : diffusivities_) {
            if (coefficient < 0.0 || !std::isfinite(coefficient)) {
                throw std::invalid_argument(
                    "RDME diffusion coefficients must be finite and nonnegative");
            }
        }
    }

    void check_voxel(index_type voxel) const {
        if (voxel < 0 || voxel >= graph_.n_vertices()) {
            throw std::out_of_range("RDME voxel index is outside the domain");
        }
    }

    void check_species(std::size_t species) const {
        if (species >= diffusivities_.size()) {
            throw std::out_of_range("RDME species index is outside the domain");
        }
    }

    Graph graph_;
    num::array<double> voxel_volumes_;
    num::array<double> diffusivities_;
};

template <class Graph>
voxel_domain(Graph, num::array<double>, num::array<double>) -> voxel_domain<Graph>;

template <class Graph>
voxel_domain(Graph, num::array<double>) -> voxel_domain<Graph>;

} // namespace rdme
