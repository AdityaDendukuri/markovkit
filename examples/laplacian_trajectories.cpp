#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "markovkit.hpp"
#include <array>
#include <cmath>
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
  const else_sim::ReversibleLaplacian generator(laplacian, h);

  constexpr std::size_t count = 5;
  constexpr std::size_t samples = 20;
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
    for (auto& curve : curves[panel]) {
      curve.assign(times.size(), 0.0);
    }

    for (std::size_t sample = 0; sample < samples; ++sample) {
      const auto path = else_sim::laplacian_else_trajectory(
          generator,
          starts[panel],
          0.0,
          times.back(),
          {.capacity = 60},
          static_cast<int>((1000 * panel) + sample));

      std::size_t state_index = 0;
      for (std::size_t time_index = 0; time_index < times.size(); ++time_index) {
        while (state_index + 1 < path.times.size() &&
               path.times[state_index + 1] <= times[time_index]) {
          ++state_index;
        }
        const auto state = static_cast<num::idx>(path.states[state_index][0]);
        for (std::size_t observable = 0; observable < count; ++observable) {
          curves[panel][observable][time_index] +=
              committors[state][observable] / static_cast<double>(samples);
        }
      }
    }
  }

  num::plt::subplot(2, 3);
  for (std::size_t panel = 0; panel < count; ++panel) {
    for (std::size_t observable = 0; observable < count; ++observable) {
      num::plt::plot(times,
                     curves[panel][observable],
                     "C" + std::to_string(observable + 1),
                     "linespoints lw 2 pt 7 ps 0.35");
    }
    num::plt::title("start = argmax(C" + std::to_string(panel + 1) + ")");
    num::plt::xlabel("t");
    num::plt::ylabel("mean C^T p(t)");
    num::plt::semilogx();
    num::plt::ylim(-0.02, 1.02);
    if (panel == 0) {
      num::plt::legend();
    }
    num::plt::next();
  }
  num::plt::savefig("laplacian_trajectories.png");
}
