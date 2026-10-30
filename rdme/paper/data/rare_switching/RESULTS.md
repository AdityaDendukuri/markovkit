# Rare endpoint probabilities in the four-voxel RDME

See BENCHMARK_DESIGN.md for the complete model and screening record. Final rate constants are (0.0258, 0.0024, 0.00138, 0.0829); D=1; initial (0,0,0,0); T=500. All results below solve or simulate the identical spatial model.

## Multigrid and numerical refinement

| Run | States | Wall time (s) | P(N >= 16) | P(N = 20) | Missing mass |
|---|---:|---:|---:|---:|---:|
| coarse | 487635 | 37.416 | 1.77190946689e-06 | 1.23081564734e-07 | 7.98e-09 |
| fine | 814385 | 134.389 | 1.77456056235e-06 | 1.23248925935e-07 | 4.98e-11 |
| time | 814385 | 231.308 | 1.77455723062e-06 | 1.23248699137e-07 | 4.99e-11 |
| cap | 1282975 | 209.272 | 1.77457729688e-06 | 1.23249981068e-07 | 9.54e-14 |
| direct | 487635 | 66.117 | 1.771909467e-06 | 1.23081564735e-07 | 7.98e-09 |

Reference for the comparisons is cap 72, dt=5. The time refinement uses cap 64, dt=2.5. Coarse and direct use cap 56, dt=10; fine uses cap 64, dt=5. Both recursive and direct use the same population-block relaxation, W coarse visits, shift, tolerance, and initial iterate. Timings are sequential single runs including output.

These are numerical cap/time refinement diagnostics. The per-solve residual bound and reported missing mass are not a rigorous bound on the complete continuous-time error. No state clipping or renormalization is used.

## Completed SSA run

One million independent trajectories completed in 12.189 s.

| Observable | FSP probability | Expected samples | Observed samples | Expected SSA relative standard error | Paths for 10% relative standard error |
|---|---:|---:|---:|---:|---:|
| high | 1.7745773e-06 | 1.775 | 3 | 0.751 | 56,351,348 |
| peak_bin | 1.23249981e-07 | 0.123 | 0 | 2.848 | 811,359,051 |

Sample requirements are computed from the Bernoulli variance using the refined FSP probabilities. They are predictions, not completed Monte Carlo counts or guaranteed confidence levels. Exact SSA samples paths without time-discretization bias; its endpoint probabilities have sampling error. The comparison concerns ordinary independent SSA, not specialized rare-event sampling.

The central numerical result is the resolved positive small secondary maximum at N=20. The low-population mode is N=0. The high-branch maximum is shallow, and the figure should not exaggerate its prominence. No stationary, first-passage, or general SSA-superiority claim follows from this experiment.

## Completed SSA run at the target basin accuracy

60,000,000 independent SSA trajectories completed in 673.488 s, retaining aggregate endpoint counts. This output mode was checked to reproduce the first million raw trajectory endpoints exactly. The larger ensemble contains the first million paths; the two ensembles are not independent additional samples.

High-region endpoints: 96; estimated probability 1.6e-06; expected relative standard error 0.0969.

Endpoints in the N=20 bin: 4; estimated probability 6.66666666667e-08; expected relative standard error 0.3677.

The sample size was set from the previously refined FSP high-region probability before viewing these endpoint counts. The expected high-region relative standard error is below 10%. The individual rare-bin probability remains much less precisely estimated.
