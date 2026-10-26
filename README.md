# MarkovKit

MarkovKit is a C++20 library for continuous-time Markov chains and stochastic
reaction networks. It provides finite-state projection (FSP), stochastic
simulation, and reaction-diffusion master-equation (RDME) models.

## Methods

- **FSP**
  - `solve_adaptive_fsp`: flux-controlled state-space growth
  - `solve_krylov_fsp`: Krylov propagation with the Vo–Sidje mass-loss criterion
- **SSA**
  - `gillespie`: exact direct-method simulation
  - `tau_leaping`: approximate Poisson tau-leaping
- **RDME**
  - `voxel_domain`: structured or unstructured finite-volume diffusion geometry
  - `diffusion_state_space`: fixed-population sectors with dynamic molecule-count arrays
  - `diffusion_generator`: cached, allocation-free matrix-free generator application
  - `reaction_state_space`: contiguous unions of population sectors exposing block structure
  - `restricted_generator`: cached reaction-diffusion topology with allocation-free rate refresh
  - `population_transfer`: conservative voxel aggregation and multinomial population lifting
  - `population_multigrid`: recursive Petrov–Galerkin V-cycle for shifted diffusion sectors
  - `sector_diffusion_multigrid`: block preconditioner for full reaction-diffusion GMRES solves
  - `lift_to_cme`: exact event lift for small problems and solver verification

MarkovKit uses [Numerics](../numerics) for matrices, factorizations, resolvent
solves, Talbot inversion, JSON input, and plotting. The two repositories should
be adjacent when building from source.

Shared CME types live under `cme/`: reaction systems, finite state spaces,
and the generator on a finite state space. `fsp/`, `ssa/`, and
`rdme/` contain method-specific policies and algorithms. The legacy
`fsp/{boundary,generator,state_space}.hpp` and `markovkit/reaction_system.hpp`
headers remain as forwarding compatibility headers.

Every name is `snake_case`, matching Numerics; only template parameters
(`State`, `Graph`, `Solver`) keep CamelCase. Storage uses Numerics' container
vocabulary: `num::array<T>` where a sequence grows, `num::vec<double>`/`num::mat<double>` for
fixed-length numeric data, `num::view<T>` for non-owning windows, and
`num::unordered_map`/`num::map`/`num::set` for hashed and ordered lookup.

`reaction_system` centralizes propensity evaluation. Shared distribution and
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
| `fsp_oregonator` | Adaptive FSP for the field–Körös–Noyes oscillator | `fsp_oregonator.png` |
| `fsp_toggle_switch` | Adaptive FSP for a bistable genetic switch | `fsp_toggle_switch.png` |
| `ssa_birth_death` | Gillespie SSA and tau-leaping comparison | `ssa_birth_death.png` |
| `rdme_multigrid_benchmark` | Plain and multigrid-preconditioned RDME GMRES scaling | CSV on stdout |
| `rdme_two_particle_annihilation` | Resolvent statistics for diffusing `A + B -> empty` | CSV and optional PNG figures |
| `rdme_multigrid_analysis` | Smoothing and two-grid analysis on fixed-population configuration graphs | CSV and optional PNG figure |

Run an example from the repository root after building:

```bash
./build/examples/rdme_multigrid_benchmark > rdme_multigrid_results.csv
# Fast 1D/2D/3D validation subset:
./build/examples/rdme_multigrid_benchmark --quick
./build/examples/rdme_two_particle_annihilation --plot > annihilation_results.csv
# Fast 1D/2D/3D validation subset:
./build/examples/rdme_two_particle_annihilation --quick
./build/examples/rdme_multigrid_analysis --plot > rdme_multigrid_analysis.csv
```

The RDME benchmark emits tidy CSV with one row per solver and problem. Its
default matrix covers 1D, 2D, and 3D mesh/population scaling plus diffusion,
reaction, and resolvent-shift sweeps, comparing plain GMRES, Jacobi, ILU(0),
and sector multigrid. Use `--quick` for the five-case validation subset.

The two-particle annihilation example starts one molecule each of two diffusing
species either together or at opposite corners. For every resolvent shift `s`,
it reports the annihilation-time transform `E[exp(-s T)]`, the transformed
survival probability, and the numerical probability identity
`E[exp(-s T)] + s S_hat(s) = 1`. The state count is `1 + 2 V + V^2` for `V`
voxels, so the benchmark directly exercises the two-particle configuration
space while comparing plain and multigrid-preconditioned GMRES.
With `--plot`, it also writes `rdme_annihilation_transform.png`, showing the
mesh dependence of the reaction-time transform, and
`rdme_annihilation_multigrid.png`, comparing plain and preconditioned iteration
scaling. `rdme_annihilation_density.png` shows transient joint probability on
the two-particle configuration space: diffusion spreads probability in
`(x_A, x_B)`, while annihilation removes mass near the diagonal `x_A = x_B`.
Plot rendering requires gnuplot.

`rdme_multigrid_analysis` compares weighted Jacobi with aggregate-block
relaxation on the complement of the population-prolongation range, then measures
exact two-grid and recursive V- and W-cycle contraction.  Each block exactly
couples configurations that differ only by redistributing particles among the
children of one coarse voxel; trivial blocks are skipped and residuals are
updated sparsely.  Norms use the inverse-multinomial stationary weight, the
natural reversible norm for probability errors in a fixed diffusion sector.

Subnetwork sweeping (`subsweep`) is vendored under `include/subsweep/`; include
`<subsweep.hpp>`. Its development copy lives in the adjacent `else` repository.
The `laplacians/*.json` inputs its Laplacian examples read are kept here; each JSON
object contains the sparse symmetric matrix `L`, the stationary square root `h`,
and the five-column committor matrix `C`.

PNG rendering requires `gnuplot`.

## License

MIT License.

## Code conventions

This is a research prototype. Molecule counts use `num::array<int>` in both
CME and RDME code; RDME coordinates are ordered by voxel, then species.
There is no separate count container or generator workspace: allocate a
`num::vec<double> image(generator.rows())` and call `generator.apply(p, image)`.
Use Numerics for numeric storage, sparse matrices, factorizations, and Krylov
solvers. Keep model construction and numerical steps explicit.
