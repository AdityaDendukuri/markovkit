#include "markovkit.hpp"
#include <vector>

int main() {
    // This model has constant birth and count-dependent death.
    markovkit::ReactionSystem model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const markovkit::State &, const std::vector<double> &r, double) { return r[0]; },
        [](const markovkit::State &x, const std::vector<double> &r, double) {
            return r[1] * x[0];
        }};

    const std::vector<double> rates = {5.0, 0.1};
    const markovkit::State initial{0};

    // Compare exact SSA with fixed-tolerance tau leaping.
    const auto exact = ssa::gillespie(model, rates, initial, 0.0, 20.0, 42);
    const auto approximate = ssa::tau_leaping(model, rates, initial, 0.0, 20.0, {}, 42);

    // Extract scalar counts from the exact path.
    const auto exact_counts = markovkit::trajectory_component(exact, 0);

    // Extract scalar counts from the approximate path.
    const auto approximate_counts = markovkit::trajectory_component(approximate, 0);

    // Overlay both piecewise-constant trajectories.
    num::plt::plot(exact.times, exact_counts, "Gillespie", "steps lw 2");
    num::plt::plot(approximate.times, approximate_counts, "tau-leaping", "steps lw 2 dt 2");
    num::plt::xlabel("time");
    num::plt::ylabel("count");
    num::plt::legend();
    num::plt::savefig("ssa_birth_death.png");
}
