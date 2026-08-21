# MarkovKit

C++20 library for stochastic reaction networks, finite-state projection (FSP),
and exact linear subnetwork elimination (ELSE).

## Structure

- **`fsp/`**: State space management, sparse generator construction, and FSP solvers:
  - `AdaptiveFSP`: Flux-preserving adaptive time-stepping ($\Delta t = \epsilon / \Phi$)
  - `KrylovFSP`: Vo & Sidje (2016) mass-loss criterion
- **`else/`**: Exact Linear Subnetwork Elimination (ELSE / FESS):
  - `Subnetwork`: restricted generator solves, occupation moments, and exit laws
  - `else_trajectory` and `else_ensemble`: sampled ELSE evolution
  - `density_subnetworks` and `TalbotDensitySolver`: resolvent density evolution
- **`ssa/`**: Exact and approximate Monte Carlo:
  - `gillespie`: Direct exact SSA
  - `tau_leaping`: Adaptive Poisson tau-leaping

## Build & Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

## Examples

- `fsp_oregonator`: Stiff Field-Körös-Noyes oscillator solved via Adaptive FSP
- `fsp_toggle_switch`: Gardner bistable mutual-inhibition switch
- `else_oregonator`: ELSE and SSA trajectory comparison for the Oregonator
- `else_birth_death`: Subnetwork jumping trajectory for 1D birth-death process
- `michaelis_menten`: Enzyme kinetics comparing FSP and ELSE Talbot density evolution
- `laplacian_committor`: Committor evolution on symmetric graph Laplacians
- `c_lapl`: Talbot/resolvent committor evolution for a Laplacian JSON file
- `ssa_birth_death`: Gillespie SSA vs Tau-Leaping

### Running examples

Build the examples once, then run the executable from the build directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

./build/examples/fsp_oregonator
./build/examples/fsp_toggle_switch
./build/examples/else_oregonator
./build/examples/else_birth_death
./build/examples/ssa_birth_death
./build/examples/michaelis_menten
./build/examples/laplacian_committor
```

The Laplacian resolvent example reads `laplacians/cs-medium.json` and writes
`c_lapl_evolution.png`:

```bash
./build/examples/c_lapl
```

PNG rendering requires `gnuplot`.

## License

MIT License.
