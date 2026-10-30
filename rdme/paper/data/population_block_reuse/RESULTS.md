# Reusing population blocks in the RDME FSP implementation

The implementation now retains existing population-block factors and transfer
entries as the adaptive FSP grows. The old interior matrix rows are reused;
only the former boundary rows and new populations are assembled. Each block
is a fixed-population collection of spatial arrangements. Diffusion remains
inside a block, while the Schlögl birth and death channels connect adjacent
populations. The diagonal blocks already contain outward birth losses, so
moving the absorbing boundary does not change their numerical factors.

Block ILU now stores indices in 32 bits, releases its setup workspace, and
uses one persistent smoothing buffer per hierarchy level. The arithmetic
uses the same numerics ILU(0) and triangular-solve kernels. Matrix assembly
counts stored entries before reserving CSR arrays. Old fine-level work arrays
are released before new work arrays are allocated. The bottom one-voxel
Thomas factors are extended from their old boundary, and existing transfer
weights remain unchanged. Exact block relaxation retains its work vectors.

## Measured comparison

Two sequential repetitions per version and problem, alternating execution
order. Hardware: Apple M1 Pro, 16 GB. Both versions use W-cycles and one
population-block ILU sweep in each direction. The table lists individual
monotonic process elapsed times, including full-state CSV output, and reports
arithmetic means for setup times and per-process peak RSS. Raw individual
runs and executable hashes are retained next to this report.

| Problem | Before, wall s (runs 1, 2) | After, wall s (runs 1, 2) | Mean setup s, before → after | Mean peak RSS GiB, before → after |
|---|---:|---:|---:|---:|
| Adaptive Schlögl, 2,123,555 final states | 60.45, 60.59 | 268.31, 60.76 | 1.106 → 0.589 | 1.185 → 0.933 |
| Rare-probability FSP, 487,635 states | 36.96, 36.05 | 36.19, 36.22 | 0.135 → 0.119 | 0.301 → 0.195 |

The first optimized adaptive run took 268.3 s, with some solve intervals
taking about 44–47 s rather than about 6 s. The second optimized run took
60.8 s, close to the baseline repetitions. The cause of this timing variation
has not been established. The long run is retained in the raw data and the
table, and no end-to-end adaptive runtime speedup is claimed. Setup time
and memory were lower in both repetitions.

The adaptive case uses the default four-voxel Schlögl rates, initial state
(2,2,2,2), backward Euler with dt=2 through T=20, initial cap 8, and cumulative
FSP tolerance 1e-5. Both versions choose the same caps through 82 and take
666 cycles, including rejected trials. The rare-probability case uses
(c1,c2,c3,c4)=(0.0258,0.0024,0.00138,0.0829), four voxels, zero initial
population, cap 56, and the established Crank–Nicolson calculation through
T=500 with dt=10 and two initial backward-Euler half steps. Both benchmarks
use D=1 and the same per-solve tolerance within each problem.

All saved full-state probability vectors agree within output precision;
computed l1 differences and individual timing pairs are in summary.json.
Iteration counts agree in every timing pair. These are implementation
comparisons at identical truncations and time steps, not new accuracy claims.
Two repetitions support a local measurement, not a general scaling law.
Historical timings in the paper and other reports have not been replaced.

## Verification and scope

All 23 CTest tests pass. The adaptive validation now directly compares
incremental growth against --reuse-fsp 0 for point, exact-block, block-ILU,
one-voxel, 2D, direct-hierarchy, and K-cycle cases, including growth from cap
zero and expansion after accepted steps. It compares chosen caps, acceptance,
iteration counts, and complete saved probability vectors. Separate checks
exercise sparse-entry counts with immigration, autocatalytic birth, linear
death, and cubic death individually enabled.

Reuse is enabled by default for point, Jacobi, exact-block, and block-ILU
smoothers. Experimental patch and cascading smoothers retain full rebuilding.
Numerical factors are reused across cap expansion only with the same voxel
graph, rates, shift, and hierarchy. Blocks at different populations have
different sizes and rates and do not share a numerical factorization.
The implementation still stores the full sparse FSP operator and every
spatial arrangement at each retained population. This work does not remove
combinatorial state growth or establish optimal implementation efficiency.

The storage_bytes estimate now includes compact population ILU factors,
which were omitted from that estimate before this change. Use measured peak
RSS for old/new memory comparisons.

Reproduce the timing suite with:

```sh
python3 paper/scripts/benchmark_population_block_reuse.py \
  --baseline tmp/rdme_before_block_reuse --repeats 2 \
  --output-dir /tmp/population_block_reuse_new
```

Baseline and optimized source snapshots, source hashes, full commands,
executable hashes, raw CSV logs, and final probability vectors are retained
in this directory. Exploratory variants in ../population_block_reuse_initial
and ../population_block_reuse_preallocation were stopped to address memory
costs and are excluded from the final comparison. This remains the existing
population-block smoother; Rob Saye's prospective work on alternative
smoothing and convergence remains separate.
