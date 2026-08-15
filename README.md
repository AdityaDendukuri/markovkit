# markovkit

Modular C++20 library for Markov jump processes, Stochastic Chemical Kinetics (Finite State Projection / FSP), and Markov Subnetwork Elimination (ELSE).

---

## Features

- **Finite State Projection (FSP)**:
  - Multi-index discrete state space enumeration and stoichiometric reachability.
  - CME generator matrix assembly as a high-performance CSR sparse matrix.
  - Adaptive state space expansion and Krylov matrix exponential integration (`num::expv`).
- **Exact Linear Subnetwork Elimination (ELSE)**:
  - Continuous-Time Markov Chain (CTMC) subnetwork elimination and fundamental matrix computation ($Z = -R^{-1}$).
  - First-passage time distributions and exponential parameter fitting.
- **High Performance Integration**:
  - Leverages **`numerics`** for matrix exponential action, direct factorizations, and Krylov routines.
  - Integrates **`approxchol`** for fast Graph Laplacian and Symmetric Diagonally Dominant (SDD) preconditioning.

---

## Build & Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## Code Example

```cpp
#include <markovkit.hpp>
#include <iostream>

int main() {
    using namespace fsp;

    // 1. Define Reaction System: DNA -> DNA + mRNA (k=5.0), mRNA -> 0 (gamma=0.1)
    ReactionSystem sys;
    sys.stoich_vecs = {MultiIndex{1}, MultiIndex{-1}};
    sys.propensities = {
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[0]; },
        [](const MultiIndex& x, const std::vector<double>& rates, double t) { return rates[1] * x[0]; }
    };
    std::vector<double> rates = {5.0, 0.1};

    // 2. Set Up State Space
    StateSpace states;
    states.add_state(MultiIndex{0}, 1.0);
    BoundaryCondition bc = rect_boundary(50);
    states.expand(sys, bc, 20);

    // 3. Assemble Generator Matrix Q
    auto [Q, erb, gd] = build_generator(states, sys, rates, 0.0, true);

    // 4. Time Evolution via Krylov Arnoldi (num::expv)
    num::Vector p0((num::idx)states.size(), 0.0);
    p0[0] = 1.0;

    num::operators::SparseOp Qop(Q);
    num::Vector p_t = num::expv(10.0, Qop, p0, 30, 1e-8);

    std::cout << "FSP solve completed. P(mRNA = 0) = " << p_t[0] << "\n";
    return 0;
}
```

---

## License

MIT License.
