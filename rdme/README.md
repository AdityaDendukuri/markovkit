# Multigrid finite state projection for the RDME

Research implementation and recorded results accompanying **A Multigrid Finite
State Projection Algorithm for the Solution of the Reaction-Diffusion Master
Equation**, by Aditya Dendukuri and Linda Petzold, Department of Computer Science,
University of California, Santa Barbara.

The paper driver is `cpp/rdme_full_multigrid.cpp`. It keeps the fine FSP
probability vector and solves shifted RDME equations using aggregation of voxel
counts, multinomial prolongation, and standard V-, W-, or K-cycles. Point and
population-block Gauss--Seidel relaxation, adaptive population selection, and
optional exact reflection reduction are included. The separate SSA front example
is `cpp/schlogl_spatial_front.cpp`. The code also contains exploratory smoothers
clearly separated from the reported method.

## Build and validate

The standalone project under `cpp` needs CMake >= 3.24, a C++20 compiler,
`pkg-config`, SuiteSparse KLU, OpenBLAS with LAPACKE, and the adjacent [Numerics](https://github.com/AdityaDendukuri/numerics)
source tree. The package is validated against Numerics commit
`dc6f1ff9121dea12ea7d75d948accf8af073f325`. Python 3 enables the validation
scripts; gnuplot with PDF/PNG support is needed only to rebuild figures.

The default layout is `numerics/` beside `markovkit/`, with Numerics built under
`numerics/build-clang`. From this `rdme` directory run

```sh
# Build the adjacent Numerics checkout at the revision above.
cmake -S ../../numerics -B ../../numerics/build-clang -DCMAKE_BUILD_TYPE=Release \
  -DNUMERICS_BUILD_EXAMPLES=OFF -DNUMERICS_BUILD_TESTS=OFF \
  -DNUMERICS_BUILD_IO=OFF -DNUMERICS_ENABLE_CUDA=OFF -DNUMERICS_ENABLE_MPI=OFF
cmake --build ../../numerics/build-clang --target numerics -j4
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j4
ctest --test-dir cpp/build --output-on-failure
cpp/build/rdme_full_multigrid --cells 4 --cap 12 --initial 8 --cycles 2 --check
```

For another Numerics layout, pass `-DNUMERICS_SOURCE_DIR=/path/to/numerics`
and `-DNUMERICS_BUILD_DIR=/path/to/numerics/build`. This project is built separately
from the older library examples at the MarkovKit root. The obsolete
sector-only drivers are not included in this standalone project.

## Experiments and figures

Run all commands from this `rdme` directory. Scripts retain their original
`cpp/` and `paper/` layout so that source paths and recorded filenames agree.

```sh
# Re-run convergence/scaling cases; large experiments are not part of CTest.
python3 paper/scripts/run_full_multigrid.py history
python3 paper/scripts/run_full_multigrid.py many_voxels
# Rebuild the main convergence/scaling and transient figures from saved records.
python3 paper/scripts/build_full_multigrid_figures.py
python3 paper/scripts/show_block_bistability.py
# Rebuild rare-switching and supplementary transfer/front figures.
python3 paper/scripts/summarize_rare_switching.py
python3 paper/scripts/plot_transfer_sparsity.py
python3 paper/scripts/plot_spatial_front_marginals.py --paper
```

All plotted figures can be rebuilt with one command (after building the solver):

```sh
python3 paper/scripts/reproduce_figures.py
```

The script copies the rare-switching figure into `paper/figures` and checks
that all seven data figures are present. The three schematic figures are TikZ
sources in `paper/figures`; they are included directly by the manuscript and
use the `arrows.meta`, `calc`, `positioning`, and `fit` TikZ libraries.
Shared gnuplot helpers are in `paper/scripts/plotting_helpers.py`.

| Paper figure | Reproduction script or source |
|---|---|
| Voxel hierarchy | `paper/figures/rdme_hierarchy_overview.tex` |
| Transfer operations | `paper/figures/rdme_transfer_operations.tex` |
| Graph energies and coarse classes | `paper/figures/rdme_state_graph_classes.tex` |
| Cycle convergence and spatial scaling | `build_full_multigrid_figures.py` |
| Transient distributions | `show_block_bistability.py` |
| Rare switching | `summarize_rare_switching.py` |
| Transfer sparsity and population blocks | `plot_transfer_sparsity.py` |
| Spatial front and voxel marginals | `plot_spatial_front_marginals.py --paper` |

Script paths in this table are relative to `paper/scripts`. These commands
replot the recorded experiments; they do not rerun the large simulations.
`run_full_multigrid.py` skips cases with existing metadata. To recompute cases,
move the corresponding records aside first. Large-run scripts and independent
checks are provided separately and may take substantial time and memory.

`paper/data/full_multigrid` contains convergence, refinement, timing, and
independent-reference records. `rare_switching` contains FSP refinements and
SSA endpoint summaries; `spatial_schlogl_front` contains the separate front
illustration. `structured_population_blocks`, `reflection_fsp`, and
`population_block_reuse` record paired implementation comparisons;
`adaptive_population_fsp` records adaptive selection. `transfer_sparsity`
contains exact exported transfers. Historical source snapshots in the paired comparisons preserve the measured
before/after implementations and are retained only as provenance.

Historical command lines and executable hashes are preserved unchanged and can
contain paths from the original workspace. The current scripts compute paths
from their checkout location. Timings describe the recorded hardware/builds;
rebuilding is not expected to reproduce wall-clock times exactly.

Full joint-probability dumps, the million-row trajectory dump, and obsolete
archived runs are omitted from Git; the manifest lists these exclusions.
Their generator commands and recorded comparison results are retained, and
the validation/benchmark scripts can regenerate the dumps. No generated
executables, dependency libraries, manuscript drafts, or collaboration notes
are included. `SOURCE_MANIFEST.json` records SHA-256 hashes of the source and
records shipped in this directory.
