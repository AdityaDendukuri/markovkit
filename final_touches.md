# Final touches

## Implementation status

Completed across MarkovKit and Numerics:

- CME state spaces, reaction systems, generator assembly, restrictions, and
  reversible representations now have canonical ownership under `cme/`.
- ELSE trajectory evolution now lives in `else/trajectory.*`; the old header is
  a compatibility forwarder.
- Numerics owns sparse transpose/scaling/densification/diagonal operations,
  diagonal similarity transforms, automatic dense/sparse real and shifted
  solvers, transpose and in-place solves, blocked inverse diagonals, selected
  inverse entries, Cholesky update/downdate, and inverse-Laplace accumulation.
- Numerics also owns basis construction, scatter/gather, vector interoperability,
  categorical sampling, nonnegative normalization, and smallest-index selection.
- MarkovKit retains reaction rates, support expansion, boundary flux, pruning
  policy, exit laws, and trajectory semantics.

## Terminology

- Use `OccupationIntegrals` rather than `OccupationMoments`.
- Name the two resolvent quantities `occupation` and
  `time_weighted_occupation`:
  \[
  u = \int_0^\infty p(t)\,dt, \qquad
  v = \int_0^\infty t\,p(t)\,dt.
  \]
- Reserve "moment" for a moment of a normalized random variable. The second
  resolvent application computes a time-weighted occupation integral, not a
  second moment by itself.

## MarkovKit organization

- Move finite CME generator assembly and the shared state space out of `fsp/`.
  They are used by FSP and ELSE and belong in a small `cme/` module.
- Put `RestrictedGenerator` beside the common generator assembly rather than
  in the ELSE subnetwork implementation.
- Separate the reusable reversible representation from ELSE-specific
  Laplacian support construction. `ReversibleGenerator`, `ReversibleLaplacian`,
  and Laplacian-to-generator restriction belong in a precisely named shared
  module; neighborhoods and successive subnetworks remain under `else/`.
- Rename `else_solver` to `trajectory`. It implements ELSE trajectory
  evolution, while `density` implements deterministic density propagation.
- Avoid a broad `core` directory. Prefer domain names (`cme`, `fsp`, `ssa`,
  `else`) so ownership remains obvious.

A lean target layout is:

```text
cme/       state space, reaction system, restricted generator assembly
ssa/       Gillespie and tau-leaping
fsp/       FSP problems and algorithms
else/      subnetwork occupation, density, trajectories, support policies
```

## Numerics absorption

Move generic numerical mechanisms to Numerics while retaining Markov-chain
semantics in MarkovKit:

- reusable LU, Cholesky, KLU, and shifted-resolvent solves;
- automatic dense/sparse shifted-solver selection;
- vector and matrix right-hand-side solves, including in-place/workspace forms;
- Cholesky update and downdate;
- inverse-diagonal and selected-inverse calculations;
- Talbot nodes and generic inverse-Laplace accumulation.

Keep state expansion, boundary flux, reaction rates, pruning policy, and exit
sampling in MarkovKit.

### Small dense-building utilities

Useful, unambiguous primitives include:

```cpp
num::unit_vector(n, index);
num::identity(n);
num::identity_columns(n, first, count);
num::diagonal(matrix);
num::diagonal_matrix(values);
num::set_diagonal(matrix, values);
```

`identity_columns` describes the inverse-diagonal routine more accurately than
"delta matrix": its right-hand side is a contiguous block of columns from the
identity. These helpers primarily improve clarity and storage reuse. Constructing
the basis is negligible compared with factorization solves.

The higher-value API is factorization-aware:

```cpp
num::solve_in_place(factor, right_hand_sides);
num::inverse_diagonal(factor, diagonal, workspace);
```

It lets Numerics select the appropriate implementation for dense Cholesky,
dense LU, or sparse factors and reuse allocated workspaces. MarkovKit should
not contain the block size or explicitly build basis blocks.

### Complexity and update strategy

Solving all \(n\) identity columns in blocks does not materialize the full
inverse, but it still costs \(O(n^3)\) for a dense factor. Blocking reduces
working memory to \(O(nb)\) and makes the triangular solves more efficient; it
does not change their total arithmetic complexity.

For a reversible dense subnetwork, maintain:

1. the Cholesky factor, updated or downdated in \(O(n^2)\);
2. the inverse diagonal, updated with a block-inverse formula in \(O(n^2)\);
3. temporary inverse columns obtained from the Cholesky factor when an update
   requires them.

Use the Cholesky factor as the stable source of truth. Detect a small Schur
complement or cancellation in a diagonal update and refresh the affected entry,
or periodically refresh the complete diagonal. Sherman--Morrison is appropriate
when deliberately maintaining a full inverse, but is less stable under repeated
updates and stores information ELSE otherwise does not need.

For sparse symmetric factors, investigate selected inversion (for example,
Takahashi-style recurrences) rather than applying dense identity-block solves.
