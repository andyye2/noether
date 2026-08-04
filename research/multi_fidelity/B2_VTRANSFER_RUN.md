# NCSA B2 vtransfer wall-distance two-arm run

This deployment repeats the B2 cell with the volume wall-distance input
feature, and runs only `S-matched-wd` and `P-FT-matched-wd`. Everything else --
subset, budget, coordinate frame, position scale, supernode radius, seeds -- is
the B2 cell unchanged, so the comparison inside this namespace measures the
initialization effect exactly as B2 does, with both arms reading one extra
input.

A feature arm may only be paired with another feature arm. Nothing here is
comparable to [`B2_MATCHED_RUN.md`](B2_MATCHED_RUN.md) arm by arm, because the
two deployments give the model different inputs.

The run is isolated from B2 and from every earlier experiment:

```text
repository  /scratch/andyye2/ABUPT/multi_fidelity_B2_vtransfer
slurm       /scratch/andyye2/ABUPT/slurm/multi_fidelity_B2_vtransfer
artifacts   /scratch/andyye2/ABUPT/multi_fidelity_B2_vtransfer_artifacts/<commit>
outputs     /scratch/andyye2/ABUPT/outputs/multi_fidelity_B2_vtransfer/<commit>
```

The batch scripts are `drivaerml_vtransfer_statistics_array.sbatch` and
`drivaerml_vtransfer_training_array.sbatch`. They are the B2 scripts with the
namespace substituted and three differences: the B2 output and artifact roots
join the protected list, the statistics command must request `volume_distance`,
and both training arms must carry `--wall-distance-feature`.

## Frozen cell

```text
task=common
N=100
replicate=0
subset_seed=1103
model_seed=7103
coordinate_frame=shapenet
position_scale=1000
supernode_radius=0.1
budget=compute_matched (40,000 updates)
evaluation_point_seed=4242
wall_distance_feature=true
```

## Statistics are refitted, not reused

The frozen B2 statistics artifact has no `volume_distance_logscale_*` entries,
so the preset refuses to build against it. This namespace fits its own artifact
with the extra field. Moments are accumulated per field, so the target
statistics come out bit-identical to B2's; that is what keeps the two
deployments on one normalization.

Every run directory must carry `volume_cell_surface_distance_cKDtree.pt`.
A missing file fails the first batch rather than silently skipping a design.

## Submission order

1. Materialize fresh schema-2 manifests under the commit-keyed vtransfer
   artifact root from a clean checkout.
2. Generate one statistics command and record its SHA256.
3. Run `sbatch --test-only --array=1-1` with
   `drivaerml_vtransfer_statistics_array.sbatch`, then submit it for real.
4. Require the statistics job to finish `COMPLETED 0:0`, then check the
   artifact carries `volume_distance_logscale_mean`/`_std` and that its
   `surface_pressure` and `volume_velocity` moments equal B2's.
5. Generate exactly two commands with
   `--arms S-matched-wd P-FT-matched-wd`; dry-run both and audit the radius
   graph.
6. Run `sbatch --test-only --array=1-2` with
   `drivaerml_vtransfer_training_array.sbatch`, then submit the production
   array without a dependency on an already-failed job.

The external copies of both batch files and every `.out`/`.err` file live in
the vtransfer Slurm directory above. A new attempt after any partial output
requires a fresh audit; it must not silently reuse or overwrite the existing
namespace.
