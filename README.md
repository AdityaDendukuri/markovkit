# MarkovKit

MarkovKit is a C++20 library for continuous-time Markov chains and stochastic
reaction networks. It provides finite-state projection (FSP), stochastic
simulation, and exact linear subnetwork elimination (ELSE).

## Methods

- **FSP**
  - `AdaptiveFSP`: flux-controlled state-space growth
  - `KrylovFSP`: Krylov propagation with the Vo–Sidje mass-loss criterion
- **ELSE**
  - `else_trajectory` and `else_ensemble`: sampled subnetwork evolution
  - `density_subnetworks`: deterministic boundary-density propagation
  - `TalbotDensitySolver`: density recovery from shifted resolvent solves
- **SSA**
  - `gillespie`: exact direct-method simulation
  - `tau_leaping`: approximate Poisson tau-leaping

MarkovKit uses [Numerics](../numerics) for matrices, factorizations, resolvent
solves, Talbot inversion, JSON input, and plotting. The two repositories should
be adjacent when building from source.

Shared CME types live under `cme/`: reaction systems, finite state spaces,
generator restrictions, and reversible representations. `fsp/`, `ssa/`, and
`else/` contain method-specific policies and algorithms. The legacy
`fsp/{boundary,generator,state_space}.hpp`, `markovkit/reaction_system.hpp`, and
`else/else_solver.hpp` headers remain as forwarding compatibility headers.

`ReactionSystem` centralizes propensity evaluation. Shared distribution and
trajectory utilities provide species means, marginals, component extraction,
and observation-time lookup without duplicating loops in solvers or examples.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

## Examples

| Target | Demonstration | Output |
| --- | --- | --- |
| `fsp_oregonator` | Adaptive FSP for the Field–Körös–Noyes oscillator | `fsp_oregonator.png` |
| `fsp_toggle_switch` | Adaptive FSP for a bistable genetic switch | `fsp_toggle_switch.png` |
| `else_oregonator` | Fixed-step ELSE and SSA trajectory ensembles | `oregonator_else_ssa.png` |
| `else_birth_death` | ELSE trajectory for a birth–death process | `else_birth_death.png` |
| `ssa_birth_death` | Gillespie SSA and tau-leaping comparison | `ssa_birth_death.png` |
| `michaelis_menten` | FSP and dense Talbot-density comparison | `michaelis_menten.png` |
| `density_laplacian` | Talbot density evolution on a graph | `density_laplacian.png` |
| `laplacian_trajectories` | Reversible ELSE trajectories using dense Cholesky | `laplacian_trajectories.png` |
| `dense_cholesky_cme` | Dense Cholesky occupation solve for a reversible CME | `dense_cholesky_cme.png` |

Run an example from the repository root after building:

```bash
./build/examples/else_oregonator
./build/examples/michaelis_menten
./build/examples/density_laplacian
./build/examples/laplacian_trajectories
./build/examples/dense_cholesky_cme
```

`density_laplacian` reads `laplacians/cs-medium.json`. The JSON object must
contain the sparse symmetric matrix `L`, the stationary square root `h`, and
the five-column committor matrix `C`.

PNG rendering requires `gnuplot`.

## License

MIT License.
