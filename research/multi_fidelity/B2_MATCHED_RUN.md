# NCSA B2 matched two-arm run

This deployment runs only `S-matched` and `P-FT-matched`. Both use the same
DrivAerML subset, train-only statistics, target optimizer budget, coordinate
frame, position scale, and supernode radius. The comparison therefore measures
the initialization effect under the matched rendering. It does **not** by
itself estimate the effect of changing the rendering from radius 9.0 to 0.1.

The run is isolated from every earlier experiment:

```text
repository  /scratch/andyye2/ABUPT/multi_fidelity_B2
slurm       /scratch/andyye2/ABUPT/slurm/multi_fidelity_B2
artifacts   /scratch/andyye2/ABUPT/multi_fidelity_B2_artifacts/<commit>
outputs     /scratch/andyye2/ABUPT/outputs/multi_fidelity_B2/<commit>
```

No command may name the historical `multi_fidelity`,
`multi_fidelity_exploratory`, or `multi_fidelity_paper` output roots. The batch
scripts bind a submission to the full clean Git commit and the SHA256 of its
command file. Statistics refuses to overwrite an existing artifact.

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
```

## Submission order

1. Materialize fresh schema-2 manifests under the commit-keyed B2 artifact
   root from a clean checkout.
2. Generate one statistics command and record its SHA256.
3. Run `sbatch --test-only --array=1-1` with
   `drivaerml_statistics_array.sbatch`, then submit it for real.
4. Require the statistics job to finish `COMPLETED 0:0` and validate the
   artifact before generating training commands.
5. Generate exactly two commands with
   `--arms S-matched P-FT-matched`; dry-run both and audit the radius graph.
6. Run `sbatch --test-only --array=1-2` with
   `drivaerml_training_array.sbatch`, then submit the production array without
   a dependency on an already-failed job.

The external copies of both batch files and every `.out`/`.err` file live in
the B2 Slurm directory above. A new attempt after any partial output requires a
fresh audit; it must not silently reuse or overwrite the existing namespace.
