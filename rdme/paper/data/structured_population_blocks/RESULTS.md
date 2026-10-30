# Removing repeated work without symmetry reduction

The default Cartesian population-block ILU no longer stores a second sparse
copy of the diffusion graph. A diffusion jump changes the checkerboard parity
of a molecular arrangement, so the fixed-population graph is bipartite and
has no triangles. During ILU(0), an update to a retained off-diagonal entry
would require such a triangle. Only diagonal values therefore require new
numerical storage. The implementation stores diagonal reciprocals, reads
original diffusion edges, fuses residual formation with forward substitution,
and updates the solution during backward substitution. This is the same
ILU(0) correction in exact arithmetic, with changed floating-point ordering.

For the cap-56 four-voxel problem, fine-level factor array storage falls from
42,522,000 to 7,802,388 bytes. The full fine matrix is still retained.

Further repeated work removed: Crank–Nicolson reuses the matrix product from
its right-hand side for its initial residual; a converged residual is reused
instead of recalculated; residual formation and its norm use one traversal;
time-step vectors are reused; and the bottom solve copies into its existing
buffer. A final cleanup also avoids a repeated exact bottom solve during a
W visit. That last cleanup follows the timed source snapshot.

Two paired repetitions compare the implementation at the start of this
investigation with these changes, without reflection. Mean process elapsed
time is **36.69 → 35.84 s**
(1.024×), and mean peak RSS is
**0.195 → 0.127 GiB**
(35.0% lower). Full-state l1 difference is at most
4.76e-16; the rare N=20 bin differs relatively
by 4.92e-14. Counts are 1,209 versus 1,208 cycles.
The general optimization mainly saves memory; it is not a large runtime gain.

The model, grids, and time integration are identical to the rare test in
[the reflection report](../reflection_fsp/RESULTS.md). Timings include full-state
export, are sequential on Apple M1 Pro with 16 GB RAM, and alternate order.
No symmetry reduction or new truncation is used in this comparison.

`--block-kernel reference` retains generic sparse block ILU. `--check` validates
specialized pivots and a full fused update against that implementation using
signed inputs. All 25 tests pass, including independent comparisons of full
probability vectors in 1D/2D/3D, zero diffusion, direct/recursive aggregation,
V/W/K cycles, GMRES, Crank–Nicolson, and adaptive growth. Source snapshots,
hashes, CPU/elapsed times, CSV logs, and probability vectors are preserved.
The larger runtime gain comes from exact reflection reduction when the
model and initial data have that symmetry; it is reported separately.
