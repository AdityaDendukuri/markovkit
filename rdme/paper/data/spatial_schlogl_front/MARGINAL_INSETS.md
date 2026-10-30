# Four single-voxel marginals at the spatial front

The figure uses a new 256-trajectory ensemble of the unchanged spatial
front model. All four distributions are at time 300. A is voxel index 15
(position 15.5), B is index 42 (42.5), C is index 45 (45.5), and D is
index 59 (59.5). B and C were selected from the previous 64-trajectory
probability map before inspecting the count histograms.

Each histogram is the empirical marginal of one voxel at one time.
No voxels, times, or reaction states are pooled. Bins are 5 molecules
wide and their probabilities sum to one. The complete integer-count
marginals are retained in the raw CSV. All four marginals were checked
against the independently accumulated ensemble means and high-state
probabilities.

A is concentrated near 200 molecules and D near 40. B spans both
population ranges, while C is weighted toward low populations with a
high-count tail. These are broad front distributions, not evidence of
two sharply resolved local modes. This distinction should be preserved
in any paper caption or discussion. The model has two stable deterministic
concentrations, but that fact alone does not imply a bimodal marginal at
every voxel and time.

The data and figure remain an exact-SSA spatial demonstration, not a
multigrid/FSP benchmark. Run metadata and source/executable hashes are in
front_v64_w20_p256_marginals.json; marginal_insets_summary.json gives the
four numerical summaries. The independent SSA tests passed with optional
marginal recording, including checks of histogram normalization and moments.
