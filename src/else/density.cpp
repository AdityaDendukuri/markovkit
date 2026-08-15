/// @file src/else/density.cpp
/// @brief Compiled FESSDensityEvolution methods.
#include "markovkit/else/density.hpp"

namespace markovkit::else_solver {

FESSDensityEvolution fess_density_evolution(
    const fsp::FlatHashMap<fsp::MultiIndex, double>& initial_density,
    const fsp::ReactionSystem& model,
    const std::vector<double>& rates,
    int steps,
    FESSOptions options
) {
    FESSDensityEvolution evol;
    evol.densities.push_back(initial_density);
    evol.expected_times.push_back(0.0);
    evol.support_sizes.push_back(static_cast<int>(initial_density.size()));
    evol.network_sizes.push_back(static_cast<int>(initial_density.size()));

    auto current_density = initial_density;
    double current_time = 0.0;

    for (int step = 0; step < steps; ++step) {
        std::vector<fsp::MultiIndex> entrances;
        for (const auto& kv : current_density) entrances.push_back(kv.first);

        FESSNetwork net(entrances);
        net.augment(entrances, model, rates, options.expansion_depth);

        int n = net.size();
        num::Matrix R(n, n, 0.0);
        for (int c = 0; c < n; ++c) {
            const auto& state = net.states[c];
            double total_rate = 0.0;
            for (std::size_t r = 0; r < model.propensities.size(); ++r) {
                double rate = model.propensities[r](state, rates, current_time);
                if (rate <= 0.0) continue;
                total_rate += rate;

                fsp::MultiIndex dest = state;
                for (std::size_t i = 0; i < state.size(); ++i) dest[i] += model.stoich_vecs[r][i];

                auto it = net.index.find(dest);
                if (it != net.index.end()) {
                    R(it->second, c) += rate;
                }
            }
            R(c, c) = -total_rate;
        }

        Subnetwork sub(R);
        num::Vector entrance_vec(n, 0.0);
        for (int c = 0; c < n; ++c) {
            auto it = current_density.find(net.states[c]);
            if (it != current_density.end()) entrance_vec[c] = it->second;
        }

        double time_inc = 0.0;
        num::Vector u = sub.occupation(entrance_vec);
        for (num::idx i = 0; i < n; ++i) time_inc += u[i];

        current_time += time_inc;
        evol.densities.push_back(current_density);
        evol.expected_times.push_back(current_time);
        evol.support_sizes.push_back(static_cast<int>(current_density.size()));
        evol.network_sizes.push_back(n);
    }

    return evol;
}

} // namespace markovkit::else_solver
