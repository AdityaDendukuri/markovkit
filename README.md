# MarkovKit

Header-only C++20 library for continuous-time Markov chains: FSP, SSA, and the RDME.
Requires [Numerics](../numerics) in an adjacent directory.

```cpp
#include <markovkit.hpp>
using namespace markovkit;
```

## Concepts

```cpp
// a model lists the jumps out of a state
template <typename Model>
concept markov_model = requires(const Model &model, const state &x) {
    { model.transitions(x) } -> std::convertible_to<array<transition>>; // {to, rate}
};

// a finite state space numbers its states
template <typename Space>
concept state_space = requires(const Space &space, const state &x, idx i) {
    { space.size() } -> std::convertible_to<idx>;
    { space[i] } -> std::convertible_to<const state &>;
    { space.find(x) } -> std::convertible_to<std::optional<idx>>;
};
```

Models: `reaction_network`, `reaction_diffusion<Graph>`.
Spaces: `distribution`, `population_sector`, `population_box`.
Any model on any space gives a `generator`.

## Examples

A reaction network:

```cpp
const reaction_network model{{
    {{1}, [](const state &) { return 5.0; }},
    {{-1}, [](const state &x) { return 0.1 * x[0]; }},
}};
```

SSA:

```cpp
num::rng random(42);
trajectory path = gillespie(model, {0}, 20.0, random);
trajectory leap = tau_leaping(model, {0}, 20.0, random);
```

FSP:

```cpp
fsp_solution p = solve_adaptive_fsp(model, {0}, 2.0, {.inside = box(100)});
fsp_solution q = solve_krylov_fsp(model, {0}, 2.0, {.tolerance = 1e-6});
p.means();                       // means(k, s) at p.times[k]
p.snapshots.back().marginal(0);
```

An RDME, solved with a multigrid-preconditioned resolvent:

```cpp
const reaction_diffusion rdme{
    voxel_domain(grid<1>({16}, {1.0 / 16}), {1.0}),
    {{{1}, [](view<const int>, real volume) { return 0.05 * volume; }},
     {{-1}, [](view<const int> n, real) { return 0.02 * n[0]; }}}};

const population_box box(rdme.n_voxels(), {3});
const generator Q(box, rdme);
const shifted_generator A(Q, 0.5);                 // 0.5 I - Q
const sector_multigrid M(rdme.domain, box, 0.5);
num::gmres(A, M, b, x);
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

Programs in `examples/`: `fsp_oregonator`, `fsp_toggle_switch`, `ssa_birth_death`,
`rdme_multigrid_benchmark`, `rdme_two_particle_annihilation`, `rdme_multigrid_analysis`.

`include/subsweep/` is vendored from the `else` repository. `rdme/` holds the standalone
RDME multigrid paper code.

## License

MIT
