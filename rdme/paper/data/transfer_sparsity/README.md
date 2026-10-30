# Transfer sparsity plots

These data are exported from `cpp/rdme_full_multigrid.cpp`, using the
four-voxel, cap-40 hierarchy of the recorded Schlogl relaxation comparison
in `paper/data/full_multigrid/block_schlogl_v4_k40_block-ilu_s1.csv`.
The transfers are independent of the relaxation choice and time step.
No time integration or timing experiment is rerun for these plots.

Run `python3 paper/scripts/plot_transfer_sparsity.py` after rebuilding
`cpp/build/rdme_full_multigrid`. The script invokes the solver's
`--transfer-output PREFIX --export-only --check` options. The exported
arrays are the actual `parent` and `weight` arrays used by multigrid,
not an independently reconstructed graph or matrix.

| Voxel transfer | R dimensions | P dimensions | Nonzeros in each |
|---|---|---|---|
| 4 to 2 | 861 x 135751 | 135751 x 861 | 135751 |
| 2 to 1 | 41 x 861 | 861 x 41 | 861 |

Each CSV row records `R[coarse_state,fine_state] = 1` and
`P[fine_state,coarse_state] = weight`. Indices are zero based, ordered
by total population followed by lexicographic weak-composition rank.
The script verifies contiguous indices, dimensions, positive finite
weights, complete coarse coverage, column normalization of P, and no
transfer between different total populations. The maximum normalization
errors were 1.11e-15 and 2.22e-16. The solver's independent coarse
operator checks also passed, with relative errors below 4e-15.

The full plots contain every nonzero with no sampling or reordering.
Row and column axes are scaled independently to make the rectangular
matrices readable. The smaller plot extracts all 35 nonzeros in the
population-four block of the first transfer; it is not a new cap-four
experiment. Only locations are plotted; the multinomial values are
preserved in the CSVs. Positive weights make the supports of R and P
transposes, although their nonzero values differ.

`provenance.json` records the experiment, source, executable, and data
hashes and the exact export command. The generated gnuplot scripts are
also retained. Four existing multigrid tests passed after the export
addition (smoke, 2D, 3D, and population-block relaxation). Additional
export checks passed for 2D and reflection reduction, and invalid export
option combinations were rejected.

The population-block comparison now has four panels. Its first row retains
the solver ordering. Its second row permutes fine states by coarse counts,
revealing five rectangular blocks with 5, 8, 9, 8, and 5 fine entries.
Restriction columns and prolongation rows receive the same permutation.
Every weight is unchanged. The grouped CSV records the original global
fine index, and `provenance.json` records the permutation explicitly.
This ordering is for illustration only; the solver still uses the original
lexicographic order.
