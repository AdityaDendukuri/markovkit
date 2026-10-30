# Structure and repeated work in the RDME FSP implementation

The largest measured saving came from representing reflected configurations
once. In the homogeneous benchmark models, reversing the voxel order leaves
both the rates and initial configuration unchanged. A configuration and its
reflection therefore have equal probability at every time. We store their
combined probability and reconstruct the original probabilities on export.
Self-reflected configurations are stored once. This is exact symmetry
reduction, with the same population truncation and time integration.

The general implementation changes also remove duplicated diffusion-factor
storage and unnecessary matrix applications. Their separate comparison is
in [structured_population_blocks/RESULTS.md](../structured_population_blocks/RESULTS.md).
Those changes alone reduced memory substantially but improved elapsed time
by only about 2%. They do not explain the roughly twofold gain below.

## Paired measurements

Two sequential repetitions, alternating execution order, on the recorded
Apple M1 Pro with 16 GB RAM. The baseline is the implementation at the start
of this investigation, after the previous block-reuse improvements. Both
versions include full-state CSV output, use W-cycles and population-block
ILU(0), and use identical reaction rates, voxel grids, populations, time
steps, and stopping tolerances. Entries below are arithmetic means. CPU
time is independently recorded by the Python parent using child resource
usage; process elapsed time uses Python's monotonic clock.

| Problem | Stored unknowns, before → after | Wall s, before → after | CPU s, before → after | Peak RSS GiB, before → after |
|---|---:|---:|---:|---:|
| Rare-probability FSP | 487,635 → 244,035 | 36.71 → 18.37 | 36.45 → 17.92 | 0.199 → 0.084 |
| Adaptive Schlögl | 2,123,555 → 1,062,229 | 60.03 → 30.50 | 59.55 → 29.94 | 0.903 → 0.486 |

The wall-time speedups are **2.00×** for the rare
probability problem and **1.97×** for the adaptive
problem. Mean peak RSS falls by 57.7% and
46.2%, respectively. Each individual
pair gives approximately the same runtime reduction; CPU time confirms it.

The physical FSP supports remain 487,635 and 2,123,555 states. The new counts
are stored probability unknowns, not smaller truncations. The rare test is
the established four-voxel endpoint calculation: initial population zero,
(c1,c2,c3,c4)=(0.0258,0.0024,0.00138,0.0829), D=1, cap 56, T=500, dt=10,
Crank–Nicolson with two initial backward-Euler half steps, and tolerance
1e-13. The adaptive test uses the default Schlögl rates, initial state
(2,2,2,2), D=1, T=20, backward Euler with dt=2, loss budget 1e-5, and
per-solve tolerance 1e-10. Both adaptive versions grow from cap 8 through
cap 82, with identical acceptance decisions and 666 cycles. The rare test
takes 1,209 cycles before and 1,215 after reflection reduction.

## Numerical agreement and the hierarchy control

The largest full-state l1 differences after reconstructing every original
configuration are **9.06e-15** for the rare
problem and **3.95e-14** for the adaptive
problem. The rare peak-bin probability at N=20 differs relatively by only
1.73e-13. Final probabilities are
nonnegative in every timing run, with no clipping or renormalization.
These compare the same discrete FSP calculation; they add no new claim
about time-discretization or truncation accuracy.

Reflection helps both direct and successive voxel aggregation. To check
that the hierarchy still matters, a separate control keeps reflection,
the fine problem, and solver settings fixed and omits the intermediate
two-voxel level. Direct restriction to the one-voxel CME takes
**32.18 s** and 2,179 cycles, versus
**18.37 s** and 1,215 cycles with the
intermediate level: a **1.75×** hierarchy benefit.
The control is one run, compared with the two recursive repetitions.
Its full-state l1 difference from the recursive calculation is
1.29e-15. Raw data are in `direct_control/`.

## Implementation and checks

`--reflection 1` enables the reduction; the default remains off. The driver
requires a reflection-invariant initial configuration and uniform Cartesian
model coefficients. It rejects asymmetric initial data. This does not apply
to the separately simulated spatial front with its asymmetric initial data.

The generator is assembled directly on reflected state pairs, with incoming
rates adjusted for pair sizes and duplicate transitions combined. Conditional
multinomial transfers include pair sizes, and population-block offsets use
the reduced ordering. Existing blocks and factors remain reusable during
adaptive expansion. The folded graph uses generic ILU because symmetry
identification need not preserve the triangle-free graph used by the general
specialized kernel. The bottom CME remains the original one-voxel system.

`states` reports stored unknowns; `full_states` reports the original FSP
support. State export reconstructs the original ordering and probabilities.
The resource guard still applies to the full state count. This is a constant
factor reduction for compatible models, not removal of combinatorial growth.

All **25 CTest tests pass**. The reflection tests compare complete lifted
vectors with full FSP solves for nonlinear reactions, zero diffusion, empty
population, 1D/2D/3D grids, multiple spatial levels, direct aggregation,
V/W/K cycles, Crank–Nicolson, and adaptive growth from zero and after
accepted steps. RP and RAP identities are checked independently. Invalid
asymmetric initial data and unsupported smoothers are rejected.

The general implementation separately compares its diagonal-only ILU
setup and fused block update against generic ILU on signed input vectors.
Convergence residuals still use the full selected fine operator.

## Timing provenance

Two internal step timers in optimized rare repetition 2 report 1,073.08 s
and 540.63 s, although the entire process takes 18.62 s and about 17.90 CPU s.
Their cause has not been established. Raw CSV/metadata are preserved, the
original derived summary is retained as `summary_raw.json`, and the current
summary marks component timing inconsistent and omits that solve-speedup
ratio. All speedups above use independent process elapsed and CPU times,
not the invalid component totals. Profiling data from that run should not
be used quantitatively.

The timed source snapshots and executable hashes are retained here. After
timing, one small cleanup avoids repeating an exact bottom solve on a W-cycle;
it does not change the correction. `current_source.cpp` and the final CTest
log record that cleanup. The benchmark source is `optimized_source.cpp`.

Reproduce into a fresh directory with:

```sh
python3 paper/scripts/benchmark_structured_population_blocks.py \
  --reflection --baseline-executable tmp/rdme_before_structured_blocks \
  --repeats 2 --output-dir /tmp/reflection_fsp_new
```

This work changes state representation and implementation. The voxel
hierarchy remains the main algorithmic contribution; Rob Saye's prospective
work on better smoothing and convergence remains separate. The manuscript now reports these implementation comparisons separately;
its original convergence and refinement timings are retained.
