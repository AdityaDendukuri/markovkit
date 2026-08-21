#include "markovkit.hpp"
#include <cmath>
#include <numeric>
#include <vector>

int main() {
  constexpr num::idx n = 32;
  constexpr double birth_rate = 8.0;
  constexpr double death_rate = 1.0;

  markovkit::ReactionSystem model{
      .changes = {{1}, {-1}},
      .propensities = {
          [](const markovkit::State&, const auto& rates, double) { return rates[0]; },
          [](const markovkit::State& state, const auto& rates, double) {
            return rates[1] * state[0];
          },
      }};
  const std::vector<double> rates{birth_rate, death_rate};

  fsp::StateSpace states;
  for (num::idx state = 0; state < n; ++state) {
    states.add_state(markovkit::State{static_cast<int>(state)});
  }

  // The full immigration-death CME has Poisson stationary weights.
  std::vector<double> h(n, 1.0);
  for (num::idx state = 1; state < n; ++state) {
    h[state] = h[state - 1] * std::sqrt(birth_rate / (death_rate * state));
  }

  else_sim::Subnetwork subnetwork(states, model, rates, else_sim::Reversible{h});
  num::Vector source(n, 0.0);
  source[0] = 1.0;
  const auto time_spent = subnetwork.occupation(source);

  std::vector<double> count(n);
  std::vector<double> occupation(n);
  std::vector<double> stationary(n);
  for (num::idx state = 0; state < n; ++state) {
    count[state] = static_cast<double>(state);
    occupation[state] = time_spent[state];
    stationary[state] = h[state] * h[state];
  }
  const double occupation_mass = std::accumulate(occupation.begin(), occupation.end(), 0.0);
  const double stationary_mass = std::accumulate(stationary.begin(), stationary.end(), 0.0);
  for (num::idx state = 0; state < n; ++state) {
    occupation[state] /= occupation_mass;
    stationary[state] /= stationary_mass;
  }

  num::plt::plot(count, stationary, "stationary", "lines lw 2");
  num::plt::plot(count, occupation, "occupation", "points pt 7 lc rgb '#c0392b'");
  num::plt::title("Reversible CME: dense Cholesky occupation solve");
  num::plt::xlabel("molecule count");
  num::plt::ylabel("probability");
  num::plt::legend();
  num::plt::savefig("dense_cholesky_cme.png");
}
