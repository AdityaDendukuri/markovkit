#include "markovkit.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

// Helper to compute set overlap and Jaccard similarity between two sets of shed states
double compute_jaccard(const std::vector<num::idx> &setA, const std::vector<num::idx> &setB) {
    if (setA.empty() && setB.empty()) return 100.0;
    std::unordered_set<num::idx> sa(setA.begin(), setA.end());
    std::unordered_set<num::idx> sb(setB.begin(), setB.end());
    std::size_t intersection_size = 0;
    for (num::idx x : sa) {
        if (sb.contains(x)) {
            intersection_size++;
        }
    }
    std::unordered_set<num::idx> union_set = sa;
    union_set.insert(sb.begin(), sb.end());
    return (static_cast<double>(intersection_size) / static_cast<double>(union_set.size())) * 100.0;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "       ELSE STATE SHEDDING: SELECTION CONCORDANCE EXPERIMENT           \n";
    std::cout << "========================================================================\n\n";

    constexpr num::idx n = 40;
    constexpr double birth_rate = 12.0;
    constexpr double death_rate = 1.0;

    // 1. Define Immigration-Death reaction network
    markovkit::ReactionSystem model{
        .changes = {{1}, {-1}},
        .propensities = {
            [](const markovkit::State &, const auto &rates, double) { return rates[0]; },
            [](const markovkit::State &state, const auto &rates, double) {
                return rates[1] * state[0];
            },
        }};
    const std::vector<double> rates{birth_rate, death_rate};

    cme::StateSpace states;
    for (num::idx state = 0; state < n; ++state) {
        states.add_state(markovkit::State{static_cast<int>(state)});
    }

    // Stationary Poisson weights for reversible scaling
    std::vector<double> h(n, 1.0);
    for (num::idx state = 1; state < n; ++state) {
        h[state] = h[state - 1] * std::sqrt(birth_rate / (death_rate * state));
    }

    auto restricted = cme::restrict_cme(states, model, rates);
    auto reversible = cme::reversible_generator(std::move(restricted), h);
    else_sim::Subnetwork subnetwork(std::move(reversible));

    // Initial entrance in state 0
    num::Vector entrance(n, 0.0);
    entrance[0] = 1.0;
    const auto u = subnetwork.occupation(entrance);

    // 2. Evaluate state-by-state loss under all three shedding methods
    else_sim::SheddingOptions opt_woodbury{
        .method = else_sim::SheddingMethod::CholeskyWoodbury,
        .error_threshold = 1e-6,
        .fallback_to_stable = true,
    };
    const auto losses_woodbury = else_sim::compute_shedding_losses(subnetwork, u, opt_woodbury);

    else_sim::SheddingOptions opt_symm{
        .method = else_sim::SheddingMethod::NormalizedSymmetrized,
        .sample_probes = 32,
    };
    const auto losses_symm = else_sim::compute_shedding_losses(subnetwork, u, opt_symm);

    else_sim::SheddingOptions opt_visits{
        .method = else_sim::SheddingMethod::ExpectedVisits,
    };
    const auto losses_visits = else_sim::compute_shedding_losses(subnetwork, u, opt_visits);

    // 3. Compare State Shedding Selection across Target Capacities
    std::cout << "--- State Selection Concordance across Target Capacities (Initial = 40) ---\n\n";
    std::cout << std::left << std::setw(12) << "Target Cap"
              << std::setw(12) << "Shed Count"
              << std::setw(20) << "Jaccard(M1, M3)"
              << std::setw(20) << "Jaccard(M1, M2)"
              << std::setw(20) << "Jaccard(M2, M3)" << "\n";
    std::cout << std::string(84, '-') << "\n";

    const std::vector<num::idx> test_capacities = {35, 30, 25, 20, 15, 10};
    const std::vector<num::idx> protected_states = {0};

    std::vector<double> cap_plot;
    std::vector<double> jaccard_m1_m3;
    std::vector<double> jaccard_m1_m2;
    std::vector<double> jaccard_m2_m3;

    for (num::idx cap : test_capacities) {
        auto res1 = else_sim::shed_states(subnetwork, u, protected_states, {.method = else_sim::SheddingMethod::CholeskyWoodbury, .target_capacity = cap});
        auto res2 = else_sim::shed_states(subnetwork, u, protected_states, {.method = else_sim::SheddingMethod::NormalizedSymmetrized, .target_capacity = cap});
        auto res3 = else_sim::shed_states(subnetwork, u, protected_states, {.method = else_sim::SheddingMethod::ExpectedVisits, .target_capacity = cap});

        const double j13 = compute_jaccard(res1.shed_indices, res3.shed_indices);
        const double j12 = compute_jaccard(res1.shed_indices, res2.shed_indices);
        const double j23 = compute_jaccard(res2.shed_indices, res3.shed_indices);

        cap_plot.push_back(static_cast<double>(cap));
        jaccard_m1_m3.push_back(j13);
        jaccard_m1_m2.push_back(j12);
        jaccard_m2_m3.push_back(j23);

        std::cout << std::left << std::setw(12) << cap
                  << std::setw(12) << (40 - cap)
                  << std::setw(20) << (std::to_string(j13) + " %")
                  << std::setw(20) << (std::to_string(j12) + " %")
                  << std::setw(20) << (std::to_string(j23) + " %") << "\n";
    }
    std::cout << "\n";

    // 4. Print the exact states chosen for target capacity = 25 (shedding 15 states)
    auto shed_m1 = else_sim::shed_states(subnetwork, u, protected_states, {.method = else_sim::SheddingMethod::CholeskyWoodbury, .target_capacity = 25});
    auto shed_m2 = else_sim::shed_states(subnetwork, u, protected_states, {.method = else_sim::SheddingMethod::NormalizedSymmetrized, .target_capacity = 25});
    auto shed_m3 = else_sim::shed_states(subnetwork, u, protected_states, {.method = else_sim::SheddingMethod::ExpectedVisits, .target_capacity = 25});

    std::cout << "States chosen to shed for Target Capacity = 25 (15 states shed):\n";
    std::cout << "  Method 1 (Woodbury):   ";
    for (num::idx s : shed_m1.shed_indices) std::cout << s << " ";
    std::cout << "\n";

    std::cout << "  Method 2 (Symmetrized):";
    for (num::idx s : shed_m2.shed_indices) std::cout << s << " ";
    std::cout << "\n";

    std::cout << "  Method 3 (ExpVisits):  ";
    for (num::idx s : shed_m3.shed_indices) std::cout << s << " ";
    std::cout << "\n\n";

    // 5. Generate comparative plots
    auto state_indices = num::linspace(0.0, static_cast<double>(n - 1), n);
    std::vector<double> woodbury_plot(losses_woodbury.begin(), losses_woodbury.end());
    std::vector<double> symm_plot(losses_symm.begin(), losses_symm.end());
    std::vector<double> visits_plot(losses_visits.begin(), losses_visits.end());

    // Normalize each method's losses by its maximum so the relative shapes can be compared directly
    const double max_w = *std::max_element(woodbury_plot.begin(), woodbury_plot.end());
    const double max_s = *std::max_element(symm_plot.begin(), symm_plot.end());
    const double max_v = *std::max_element(visits_plot.begin(), visits_plot.end());

    for (double &val : woodbury_plot) val /= max_w;
    for (double &val : symm_plot) val /= max_s;
    for (double &val : visits_plot) val /= max_v;

    num::plt::plot(state_indices, woodbury_plot, "Method 1: Cholesky Woodbury (Normalized Loss)", "lines lw 2 lc rgb '#2980b9'");
    num::plt::plot(state_indices, symm_plot, "Method 2: Symmetrized Diagonal (Normalized Loss)", "lines dt 2 lw 2 lc rgb '#e67e22'");
    num::plt::plot(state_indices, visits_plot, "Method 3: Expected Visits (Normalized Activity)", "points pt 7 lc rgb '#27ae60'");
    num::plt::title("ELSE State Shedding Relative Ranking Profiles (Normalized)");
    num::plt::xlabel("Molecule Count State Index");
    num::plt::ylabel("Normalized Metric Value");
    num::plt::legend();
    num::plt::savefig("else_shedding_comparison.png");

    std::cout << "[SUCCESS] Plot saved to else_shedding_comparison.png\n";
    return 0;
}
