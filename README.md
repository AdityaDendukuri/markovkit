# markovkit

C++20 library for continuous-time Markov chains and Chemical Master Equation computations.

The library provides finite state projection, subnetwork elimination, and stochastic simulation methods.

## Components

### `fsp`

Finite State Projection methods.

* adaptive FSP
* Krylov matrix exponential propagation
* reaction-network state-space construction
* generator assembly

### `markovkit::else_solver`

Mean-time Ensemble ELSE and Fourier ELSE.

- subnetwork occupation and first-exit laws
- conditional mean exit times
- ensemble block solves
- mean-exit-time and expected-entry-count shedding
- resolvent/Talbot reconstruction

### `markovkit::ssa`

Stochastic simulation methods.

* Gillespie direct method
* tau-leaping

The numerical backend provides dense BLAS/LAPACK operations and Krylov methods.

## Requirements

* C++20
* CMake 3.20+
* BLAS/LAPACK
* OpenMP (optional)

Supported platforms are macOS and Linux.

## Build

```bash
git clone https://github.com/AdityaDendukuri/markovkit.git
cd markovkit

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Basic use

```cpp
#include <markovkit.hpp>

using namespace fsp;

ReactionSystem sys;
sys.stoich_vecs = {
    MultiIndex{1},
    MultiIndex{-1}
};

sys.propensities = {
    [](const MultiIndex&, const std::vector<double>& r, double) {
        return r[0];
    },
    [](const MultiIndex& x, const std::vector<double>& r, double) {
        return r[1] * x[0];
    }
};

FSPProblem problem;
problem.model = sys;
problem.rates = {5.0, 0.1};
problem.u0 = MultiIndex{0};
problem.t0 = 0.0;
problem.tf = 10.0;
problem.bc = rect_boundary({100});

AdaptiveFSPParams params;
params.eps_dt = 0.1;
params.krylov_m = 30;

auto [solution, diagnostics] =
    solve_adaptive_fsp(problem, params);
```

See the examples directory for additional FSP, ELSE, SSA, toggle-switch, and Oregonator examples.

## License

MIT License. See `LICENSE`.
