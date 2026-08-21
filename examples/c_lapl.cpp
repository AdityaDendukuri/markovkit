#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "markovkit.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

template <class T> std::vector<T> json_vector(const boost::json::value& value) {
  std::vector<T> result;
  result.reserve(value.as_array().size());
  for (const auto& item : value.as_array()) {
    result.push_back(item.to_number<T>());
  }
  return result;
}

std::vector<std::vector<double>> json_matrix(const boost::json::value& value) {
  std::vector<std::vector<double>> result;
  result.reserve(value.as_array().size());
  for (const auto& row : value.as_array()) {
    result.push_back(json_vector<double>(row));
  }
  return result;
}

} // namespace

int main() {
  const auto document = num::io::read_json("laplacians/cs-medium.json");
  const auto& data = document.as_object();
  const auto laplacian = num::io::sparse_matrix(data.at("L"));
  const auto h = json_vector<double>(data.at("h"));
  const auto committors = json_matrix(data.at("C"));

  constexpr std::size_t count = 5;
  std::array<num::idx, count> starts{};
  for (std::size_t observable = 0; observable < count; ++observable) {
    for (num::idx state = 1; state < laplacian.n_rows(); ++state) {
      if (committors[state][observable] > committors[starts[observable]][observable]) {
        starts[observable] = state;
      }
    }
  }

  std::vector<double> times;
  for (double exponent : num::linspace(-12.0, -4.0, 41)) {
    times.push_back(std::pow(10.0, exponent));
  }

  std::array<std::array<std::vector<double>, count>, count> curves;
  for (std::size_t panel = 0; panel < count; ++panel) {
    auto subnetworks = else_sim::laplacian_density_subnetworks(laplacian, h, starts[panel], 500, 2);
    else_sim::TalbotDensitySolver density(std::move(subnetworks));

    for (double time : times) {
      const auto solution =
          density.solve(markovkit::State{static_cast<int>(starts[panel])}, time, 14);
      for (std::size_t observable = 0; observable < count; ++observable) {
        double mean = 0.0;
        for (num::idx index = 0; index < solution.probability.size(); ++index) {
          const auto state = static_cast<num::idx>(solution.states[index][0]);
          mean += committors[state][observable] * solution.probability[index];
        }
        curves[panel][observable].push_back(mean);
      }
    }
  }

  num::plt::subplot(2, 3);
  for (std::size_t panel = 0; panel < count; ++panel) {
    for (std::size_t observable = 0; observable < count; ++observable) {
      num::plt::plot(
          times, curves[panel][observable], "C" + std::to_string(observable + 1), "lines lw 2");
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
  num::plt::savefig("c_lapl_evolution.png");
}
