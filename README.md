# markovkit

C++20 library for Discrete Stochastic Simulation, Chemical Master Equations (CME), Finite State Projection (FSP), and Exact Linear Subnetwork Elimination (ELSE).

Built on top of [numerics](../numerics).

---

## Structure

- **`fsp/`**: State space management, sparse generator construction, and FSP solvers:
  - `AdaptiveFSP`: Flux-preserving adaptive time-stepping ($\Delta t = \epsilon / \Phi$)
  - `KrylovFSP`: Vo & Sidje (2016) mass-loss criterion
- **`else/`**: Exact Linear Subnetwork Elimination (ELSE / FESS):
  - `Subnetwork`: Restricted generator $R$, fundamental matrix $(-R)^{-1}$, moments $(-R)^{-2}$, exit laws, conditional mean exit time $\tau = u^{(2)}_b / u_b$
  - `else_step`, `else_trajectory`, `else_ensemble`, `else_density`
  - `Fourier` & `Talbot` resolvent time-inversion
- **`ssa/`**: Exact and approximate Monte Carlo:
  - `gillespie`: Direct exact SSA
  - `tau_leaping`: Adaptive Poisson tau-leaping

---

## Build & Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

---

## Examples

- `fsp_oregonator`: Stiff Field-Körös-Noyes oscillator solved via Adaptive FSP
- `fsp_toggle_switch`: Gardner bistable mutual-inhibition switch
- `else_oregonator`: Multi-trajectory ensemble sharing an adaptive subnetwork
- `else_birth_death`: Subnetwork jumping trajectory for 1D birth-death process
- `michaelis_menten`: Enzyme kinetics comparing FSP and ELSE Fourier density evolution
- `laplacian_committor`: Committor evolution on symmetric graph Laplacians
- `c_lapl`: Fourier/resolvent committor evolution for a Laplacian JSON file
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

The Laplacian resolvent example reads `laplacians/cs-medium.json` by default and
writes `c_lapl_curves.csv` and `c_lapl_evolution.png`. Pass a JSON path and an
optional top-left truncation size:

```bash
./build/examples/c_lapl
./build/examples/c_lapl laplacians/cs-medium.json 100
```

The second command evolves the top-left `100 × 100` block, which is useful for
quick experiments before running the full medium system. PNG rendering requires
`gnuplot`; the CSV output is written regardless.

---

## License

MIT License.
