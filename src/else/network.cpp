/// @file src/else/network.cpp
/// @brief Compiled FESSNetwork methods.
#include "markovkit/else/network.hpp"

namespace markovkit::else_solver {

FESSNetwork::FESSNetwork(fsp::MultiIndex initial_state) {
    augment(initial_state);
}

FESSNetwork::FESSNetwork(std::vector<fsp::MultiIndex> initial_states) {
    for (const auto& s : initial_states) augment(s);
}

int FESSNetwork::augment(const fsp::MultiIndex& state) {
    auto it = index.find(state);
    if (it != index.end()) return it->second;

    int new_idx = static_cast<int>(states.size());
    states.push_back(state);
    index[state] = new_idx;
    return new_idx;
}

void FESSNetwork::augment(const std::vector<fsp::MultiIndex>& entrances,
                            const fsp::ReactionSystem& model,
                            const std::vector<double>& rates,
                            int depth) {
    std::vector<fsp::MultiIndex> frontier = entrances;
    for (const auto& s : frontier) augment(s);

    for (int d = 0; d < depth; ++d) {
        std::vector<fsp::MultiIndex> next_frontier;
        for (const auto& state : frontier) {
            for (const auto& change : model.stoich_vecs) {
                fsp::MultiIndex dest = state;
                bool valid = true;
                for (std::size_t i = 0; i < state.size(); ++i) {
                    dest[i] += change[i];
                    if (dest[i] < 0) { valid = false; break; }
                }
                if (!valid) continue;
                augment(dest);
                next_frontier.push_back(dest);
            }
        }
        frontier = std::move(next_frontier);
    }
}

} // namespace markovkit::else_solver
