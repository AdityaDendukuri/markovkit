/// @file rdme/restricted_generator.hpp
/// @brief Cached block-structured reaction-diffusion generator.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include "rdme/diffusion_generator.hpp"
#include "rdme/reaction_diffusion_model.hpp"
#include "rdme/reaction_state_space.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rdme {

enum class boundary_mode { reflecting, absorbing };

template <voxel_graph Graph>
class restricted_generator {
  public:
    using domain_type = num::vec;
    using codomain_type = num::vec;
    using math_laws = num::math::type_list<num::law::linear_map>;

    struct transition {
        num::idx destination{};
        double rate{};
    };

    restricted_generator(reaction_diffusion_model<Graph> model, num::array<int> max_totals,
                         num::array<double> parameters = {}, double time = 0.0,
                         boundary_mode mode = boundary_mode::absorbing)
        : model_(std::move(model)),
          state_space_(static_cast<std::size_t>(model_.domain().n_voxels()), std::move(max_totals)),
          parameters_(std::move(parameters)), time_(time), boundary_mode_(mode),
          source_offsets_(state_space_.size() + 1, 0),
          diffusion_exit_rates_(state_space_.size(), 0.0), exit_rates_(state_space_.size(), 0.0) {
        build_topology();
        refresh(parameters_, time_);
    }

    [[nodiscard]] num::idx rows() const noexcept { return state_space_.size(); }
    [[nodiscard]] num::idx cols() const noexcept { return state_space_.size(); }
    [[nodiscard]] const reaction_diffusion_model<Graph> &model() const noexcept { return model_; }
    [[nodiscard]] const reaction_state_space &state_space() const noexcept { return state_space_; }
    [[nodiscard]] const num::array<num::idx> &source_offsets() const noexcept {
        return source_offsets_;
    }
    [[nodiscard]] const num::array<transition> &transitions() const noexcept {
        return transitions_;
    }
    [[nodiscard]] const num::array<double> &exit_rates() const noexcept { return exit_rates_; }
    [[nodiscard]] double time() const noexcept { return time_; }

    /// Re-evaluate reaction propensities without rebuilding states or transition destinations.
    void refresh(num::view<const double> parameters, double time) {
        if (parameters.size() != parameters_.size()) {
            throw std::invalid_argument(
                "RDME parameter count cannot change after generator construction");
        }
        if (parameters.data() != parameters_.data()) {
            std::copy(parameters.begin(), parameters.end(), parameters_.begin());
        }
        time_ = time;
        exit_rates_ = diffusion_exit_rates_;
        const auto &layout = state_space_.layout();

        for (const reaction_event &event : reaction_events_) {
            const configuration &state = state_space_[event.source];
            const auto local_counts = num::view<const int>(
                state.data() + layout.index(event.voxel, 0), layout.species_count());
            const auto &reaction = model_.reactions()[event.reaction];
            const double rate = reaction.propensity(
                local_counts, parameters_, model_.domain().voxel_volume(event.voxel), time_);
            if (rate < 0.0 || !std::isfinite(rate)) {
                throw std::domain_error(
                    "RDME reaction propensities must be finite and nonnegative");
            }
            exit_rates_[event.source] += rate;
            if (event.transition != no_transition) {
                transitions_[event.transition].rate = rate;
            }
        }
    }

    void apply(const num::vec &x, num::vec &y) const {
        if (x.size() != rows() || y.size() != rows()) {
            throw std::invalid_argument("RDME restricted generator vector size mismatch");
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

    [[nodiscard]] const num::vec &apply(const num::vec &x, generator_workspace &workspace) const {
        apply(x, workspace.output());
        return workspace.output();
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
    static constexpr num::idx no_transition = std::numeric_limits<num::idx>::max();

    struct reaction_event {
        num::idx source{};
        num::idx transition{no_transition};
        std::size_t voxel{};
        std::size_t reaction{};
    };

    void build_topology() {
        const auto &domain = model_.domain();
        const auto &layout = state_space_.layout();
        for (num::idx source_state = 0; source_state < state_space_.size(); ++source_state) {
            const configuration &state = state_space_[source_state];

            for (typename Graph::index_type source_voxel = 0; source_voxel < domain.n_voxels();
                 ++source_voxel) {
                for (const auto &edge : domain.graph().neighbors(source_voxel)) {
                    const auto destination_voxel = static_cast<typename Graph::index_type>(edge.to);
                    for (std::size_t species = 0; species < domain.n_species(); ++species) {
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
                        if (destination_state == reaction_state_space::npos) {
                            throw std::logic_error("diffusion left its RDME population sector");
                        }
                        const double rate = static_cast<double>(molecules) *
                                            domain.jump_rate(species, source_voxel, edge);
                        transitions_.push_back({destination_state, rate});
                        diffusion_exit_rates_[source_state] += rate;
                    }
                }
            }

            for (std::size_t voxel = 0; voxel < layout.voxel_count(); ++voxel) {
                for (std::size_t reaction_index = 0; reaction_index < model_.n_reactions();
                     ++reaction_index) {
                    configuration destination = state;
                    bool physically_valid = true;
                    for (std::size_t species = 0; species < layout.species_count(); ++species) {
                        int &count = destination[layout.index(voxel, species)];
                        count += model_.reactions()[reaction_index].change[species];
                        physically_valid = physically_valid && count >= 0;
                    }
                    if (!physically_valid)
                        continue;

                    const num::idx destination_state = state_space_.find(destination);
                    if (destination_state != reaction_state_space::npos) {
                        const num::idx transition = transitions_.size();
                        transitions_.push_back({destination_state, 0.0});
                        reaction_events_.push_back(
                            {source_state, transition, voxel, reaction_index});
                    } else if (boundary_mode_ == boundary_mode::absorbing) {
                        reaction_events_.push_back(
                            {source_state, no_transition, voxel, reaction_index});
                    }
                }
            }
            source_offsets_[source_state + 1] = transitions_.size();
        }
    }

    reaction_diffusion_model<Graph> model_;
    reaction_state_space state_space_;
    num::array<double> parameters_;
    double time_ = 0.0;
    boundary_mode boundary_mode_ = boundary_mode::absorbing;
    num::array<num::idx> source_offsets_;
    num::array<transition> transitions_;
    num::array<reaction_event> reaction_events_;
    num::array<double> diffusion_exit_rates_;
    num::array<double> exit_rates_;
};

template <class Graph>
restricted_generator(reaction_diffusion_model<Graph>, num::array<int>, num::array<double>, double,
                     boundary_mode) -> restricted_generator<Graph>;

} // namespace rdme
