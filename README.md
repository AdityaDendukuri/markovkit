# MarkovKit

MarkovKit is a C++20 library for continuous-time Markov chains and stochastic
reaction networks. It provides finite-state projection (FSP), stochastic
simulation, reaction-diffusion master-equation (RDME) models, and exact linear
subnetwork elimination (ELSE).

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
- **RDME**
  - `VoxelDomain`: structured or unstructured finite-volume diffusion geometry
  - `DiffusionStateSpace`: dynamic fixed-population sectors beyond the CME state limit
  - `DiffusionGenerator`: cached, allocation-free matrix-free generator application
  - `ReactionStateSpace`: contiguous unions of population sectors exposing block structure
  - `RestrictedGenerator`: cached reaction-diffusion topology with allocation-free rate refresh
  - `PopulationTransfer`: conservative voxel aggregation and multinomial population lifting
  - `PopulationMultigrid`: recursive Petrov–Galerkin V-cycle for shifted diffusion sectors
  - `SectorDiffusionMultigrid`: block preconditioner for full reaction-diffusion GMRES solves
  - `lift_to_cme`: exact event lift for small problems and solver verification

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
| `fsp_oregonator` | Adaptive FSP for the field–Körös–Noyes oscillator | `fsp_oregonator.png` |
| `fsp_toggle_switch` | Adaptive FSP for a bistable genetic switch | `fsp_toggle_switch.png` |
| `else_oregonator` | Fixed-step ELSE and SSA trajectory ensembles | `oregonator_else_ssa.png` |
| `else_birth_death` | ELSE trajectory for a birth–death process | `else_birth_death.png` |
| `ssa_birth_death` | Gillespie SSA and tau-leaping comparison | `ssa_birth_death.png` |
| `michaelis_menten` | FSP and dense Talbot-density comparison | `michaelis_menten.png` |
| `density_laplacian` | Talbot density evolution on a graph | `density_laplacian.png` |
| `laplacian_trajectories` | Reversible ELSE trajectories using dense Cholesky | `laplacian_trajectories.png` |
| `dense_cholesky_cme` | Dense Cholesky occupation solve for a reversible CME | `dense_cholesky_cme.png` |
| `rdme_multigrid_benchmark` | Plain and multigrid-preconditioned RDME GMRES scaling | CSV on stdout |
| `rdme_two_particle_annihilation` | Resolvent statistics for diffusing `A + B -> empty` | CSV and optional PNG figures |
| `rdme_multigrid_analysis` | Smoothing and two-grid analysis on fixed-population configuration graphs | CSV and optional PNG figure |

Run an example from the repository root after building:

```bash
./build/examples/else_oregonator
./build/examples/michaelis_menten
./build/examples/density_laplacian
./build/examples/laplacian_trajectories
./build/examples/dense_cholesky_cme
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

`density_laplacian` reads `laplacians/cs-medium.json`. The JSON object must
contain the sparse symmetric matrix `L`, the stationary square root `h`, and
the five-column committor matrix `C`.

PNG rendering requires `gnuplot`.

## License

MIT License.
