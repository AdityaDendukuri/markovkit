# Longer four-voxel Schlogl transient

The model, initial state, time step (5), cap (96), and block-smoothed W-cycle match the existing t=100 experiment. Only the final time is extended to 500. Crank--Nicolson uses two initial backward-Euler half steps.

| Time | Low peak | High peak | High/low peak height | P(N >= 21) | Missing mass |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.0814704 | no local maximum | - | 0.1220515 | 7.44e-08 |
| 100 | 0.0441340 | 0.0138521 | 0.314 | 0.4556062 | 1.85e-05 |
| 250 | 0.0255713 | 0.0211978 | 0.829 | 0.6511329 | 0.000126 |
| 500 | 0.0220749 | 0.0226463 | 1.026 | 0.6884213 | 0.000349 |

The t=100 marginal agrees with the previous run to 0 in the 1-norm. All solves converged. Probabilities are plotted without clipping or renormalization.

The displayed missing mass is a truncation diagnostic for this numerical transient. The per-solve residual bound does not include time-discretization error. No stationary convergence or new time-refinement result is asserted.

Commands, executable hash, elapsed time, full marginal histories, and solver diagnostics are saved beside this report.
