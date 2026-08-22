#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "markovkit.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    // Load the graph Laplacian, stationary weights, and committors.
    const auto document = num::io::read_json("laplacians/cs-medium.json");
    const auto &data = document.as_object();
    const auto laplacian = num::io::sparse_matrix(data.at("L"));
    const auto h = num::io::json_vector<double>(data.at("h"));
    const auto committors = num::io::json_matrix<double>(data.at("C"));

    constexpr std::size_t count = 5;
    std::array<num::idx, count> starts{};
    // Start each panel at its dominant committor state.
    for (std::size_t observable = 0; observable < count; ++observable) {
        starts[observable] = num::argmax(
            laplacian.n_rows(), [&](num::idx state) { return committors[state][observable]; });
    }

    // Resolve the fast graph dynamics on a logarithmic time grid.
    const auto times = num::logspace(-12.0, -4.0, 41);

    std::array<std::array<std::vector<double>, count>, count> curves;
    for (std::size_t panel = 0; panel < count; ++panel) {
        // Propagate a local density over logarithmic time points.
        auto subnetworks = else_sim::laplacian_restrictions(laplacian, h, starts[panel], 500, 2);
        else_sim::TalbotDensitySolver density(std::move(subnetworks));

        for (double time : times) {
            const auto solution =
                density.solve(markovkit::State{static_cast<int>(starts[panel])}, time, 14);
            // Project the density onto each committor.
            for (std::size_t observable = 0; observable < count; ++observable) {
                const double mean = num::weighted_sum(
                    std::span<const double>(solution.probability), [&](num::idx index) {
                        const auto state = static_cast<num::idx>(solution.states[index][0]);
                        return committors[state][observable];
                    });
                curves[panel][observable].push_back(mean);
            }
        }
    }

    // Each panel compares all committors from one starting basin.
    num::plt::subplot(2, 3);
    for (std::size_t panel = 0; panel < count; ++panel) {
        for (std::size_t observable = 0; observable < count; ++observable) {
            num::plt::plot(times, curves[panel][observable], "C" + std::to_string(observable + 1),
                           "lines lw 2");
        }
        num::plt::title("start = argmax(C" + std::to_string(panel + 1) + ")");
        num::plt::xlabel("t");
        num::plt::ylabel("C^T p(t)");
        num::plt::semilogx();
        num::plt::ylim(-0.02, 1.02);
        if (panel == 0) {
            num::plt::legend();
        }
        num::plt::next();
    }
    num::plt::savefig("density_laplacian.png");
}
