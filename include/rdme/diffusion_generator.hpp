/// @file rdme/diffusion_generator.hpp
/// @brief Cached matrix-free diffusion generator on a conserved RDME sector.
#pragma once

#include "container/vector.hpp"
#include "linear/sparse/sparse.hpp"
#include "rdme/diffusion_state_space.hpp"
#include "rdme/voxel_domain.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace rdme {

template <voxel_graph Graph>
class diffusion_generator {
  public:
    using domain_type = num::vec<double>;
    using codomain_type = num::vec<double>;

    struct transition {
        num::idx destination{};
        double rate{};
    };

    diffusion_generator(voxel_domain<Graph> domain, num::array<int> species_totals)
        : domain_(std::move(domain)),
          state_space_(static_cast<std::size_t>(domain_.n_voxels()), std::move(species_totals)),
          source_offsets_(state_space_.size() + 1, 0), exit_rates_(state_space_.size(), 0.0) {
        build_cache();
    }

    [[nodiscard]] num::idx rows() const noexcept { return state_space_.size(); }
    [[nodiscard]] num::idx cols() const noexcept { return state_space_.size(); }
    [[nodiscard]] const voxel_domain<Graph> &domain() const noexcept { return domain_; }
    [[nodiscard]] const diffusion_state_space &state_space() const noexcept { return state_space_; }
    [[nodiscard]] const num::array<num::idx> &source_offsets() const noexcept {
        return source_offsets_;
    }
    [[nodiscard]] const num::array<transition> &transitions() const noexcept {
        return transitions_;
    }
    [[nodiscard]] const num::array<double> &exit_rates() const noexcept { return exit_rates_; }

    /// Apply the column-oriented generator y <- Q x without allocating.
    void apply(const num::vec<double> &x, num::vec<double> &y) const {
        if (x.size() != rows() || y.size() != rows()) {
            throw std::invalid_argument("RDME diffusion generator vector size mismatch");
        }
        std::fill(y.begin(), y.end(), 0.0);
        for (num::idx source = 0; source < rows(); ++source) {
            const double probability = x[source];
            y[source] -= exit_rates_[source] * probability;
            for (num::idx entry = source_offsets_[source]; entry < source_offsets_[source + 1];
                 ++entry) {
                const auto &transition = transitions_[entry];
                y[transition.destination] += transition.rate * probability;
            }
        }
    }

    [[nodiscard]] num::spmat to_sparse() const {
        num::array<num::idx> matrix_rows;
        num::array<num::idx> matrix_columns;
        num::array<double> values;
        matrix_rows.reserve(transitions_.size() + rows());
        matrix_columns.reserve(matrix_rows.capacity());
        values.reserve(matrix_rows.capacity());

        for (num::idx source = 0; source < rows(); ++source) {
            matrix_rows.push_back(source);
            matrix_columns.push_back(source);
            values.push_back(-exit_rates_[source]);
            for (num::idx entry = source_offsets_[source]; entry < source_offsets_[source + 1];
                 ++entry) {
                matrix_rows.push_back(transitions_[entry].destination);
                matrix_columns.push_back(source);
                values.push_back(transitions_[entry].rate);
            }
        }
        return num::spmat::from_triplets(rows(), cols(), matrix_rows, matrix_columns, values);
    }

  private:
    void build_cache() {
        const auto &layout = state_space_.layout();
        for (num::idx source_state = 0; source_state < state_space_.size(); ++source_state) {
            const configuration &state = state_space_[source_state];
            for (typename Graph::index_type source_voxel = 0; source_voxel < domain_.n_voxels();
                 ++source_voxel) {
                for (const auto &edge : domain_.graph().neighbors(source_voxel)) {
                    const auto destination_voxel = static_cast<typename Graph::index_type>(edge.to);
                    for (std::size_t species = 0; species < domain_.n_species(); ++species) {
                        const std::size_t source_coordinate =
                            layout.index(static_cast<std::size_t>(source_voxel), species);
                        const int molecules = state[source_coordinate];
                        if (molecules == 0)
                            continue;

                        configuration destination = state;
                        --destination[source_coordinate];
                        ++destination[layout.index(static_cast<std::size_t>(destination_voxel),
                                                   species)];
                        const num::idx destination_state = state_space_.find(destination);
                        if (destination_state == diffusion_state_space::npos) {
                            throw std::logic_error(
                                "diffusion transition left its conserved RDME sector");
                        }

                        const double rate = static_cast<double>(molecules) *
                                            domain_.jump_rate(species, source_voxel, edge);
                        transitions_.push_back({destination_state, rate});
                        exit_rates_[source_state] += rate;
                    }
                }
            }
            source_offsets_[source_state + 1] = transitions_.size();
        }
    }

    voxel_domain<Graph> domain_;
    diffusion_state_space state_space_;
    num::array<num::idx> source_offsets_;
    num::array<transition> transitions_;
    num::array<double> exit_rates_;
};

template <class Graph>
diffusion_generator(voxel_domain<Graph>, num::array<int>) -> diffusion_generator<Graph>;

} // namespace rdme
