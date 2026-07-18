# DrivAerML subsampled-10x data evidence audit

Audit date: 2026-07-16

## Remote inventory (bounded, read-only NCSA inspection)

Authoritative path:

```text
/scratch/andyye2/data/drivaerml_subsampled_10x
```

The directory occupies 357,054,090,936 bytes (`du -sh`: 333G, approximately
332.53 GiB).  It contains exactly 484 `run_*` directories.  No bulk data were
downloaded during this audit.

The 16 absent IDs are:

```text
167 211 218 221 248 282 291 295 316 325 329 364 370 376 403 473
```

They exactly equal Noether's official hidden-test list.  The visible 484 runs
therefore exactly realize the fixed official split: train 400, validation 34,
test 50.  The experiment must treat these lists as immutable and create nested
subsets only within the official 400-run training list.

Each remote run has 16 files: 15 common tensors plus `geo_ref_<id>.csv`.
The common tensor files are:

```text
surface_area_vtp.pt
surface_normal_stl.pt
surface_normal_stl_resampled100k.pt
surface_normal_vtp.pt
surface_position_stl.pt
surface_position_stl_resampled100k.pt
surface_position_vtp.pt
surface_pressure.pt
surface_wallshearstress.pt
volume_cell_position.pt
volume_cell_pressure.pt
volume_cell_surface_distance_cKDtree.pt
volume_cell_totalpcoeff.pt
volume_cell_velocity.pt
volume_cell_vorticity.pt
```

## Shape and alignment contract

- Surface VTP position, pressure, wall shear stress, normal, and area tensors are
  pointwise aligned.
- Volume cell position, static pressure, total-pressure coefficient, velocity,
  vorticity, and distance tensors are cellwise aligned.
- Raw STL position/normal point counts are not guaranteed to match.
- The `resampled100k` STL position and normal tensors are fixed-size and aligned.

The current CAEML file map exposes `volume_cell_totalpcoeff.pt` as
`volume_pressure`; it does **not** expose `volume_cell_pressure.pt` under that
trainer property.  Reports and plots must therefore call the existing full-task
quantity “volume total-pressure coefficient” unless the file map is deliberately
changed and separately audited.

## Local sample inspection

The local lightweight cache contains runs 1, 10, and 100-104.  Run 102 lacks
`volume_cell_totalpcoeff.pt`; run 104 lacks `volume_cell_velocity.pt`.  A partial
run 105 cache is not a usable sample.  Consequently the local cache is suitable
for loader/smoke checks, not for estimating population statistics or training.

Across 103 inspected local `.pt` tensors:

- every object is a `torch.Tensor` with dtype float32;
- no NaN or infinity was found;
- representative surface positions span approximately [-1.0879, 4.0023];
- volume positions span approximately [-39.673, 79.661];
- surface pressure spans approximately [-3712.92, 896.46];
- volume velocity spans approximately [-69.992, 72.306].

The local vorticity maximum is approximately 3.57e9 and is dominated by an
extreme value in run 102.  This makes per-run quality checks, log-scale handling,
robust summaries, and field-specific failure reporting mandatory.  Seven local
runs are not enough to label that observation a global corrupt-run rule.

## Metadata limitations

The inspected `geo_ref_<id>.csv` schema contains only:

```text
lRef, aRef, forcesCoR, lRefRef, aRefRef, forcesCoRRef
```

No drag/lift coefficient and no 16-dimensional morphing parameter vector were
found, despite a local README suggesting richer metadata.  Therefore this study
must not promise scalar drag regression, morphology-stratified acquisition, or
force-coefficient evaluation without locating and validating an additional
authoritative source.

## Strict use in the sample-efficiency experiment

1. Materialize the exact run IDs and base-dataset indices for every nested
   training subset; record a SHA256, not only an RNG seed.
2. Compute target-field mean/std using only the run IDs in that subset.  Reusing
   the repository's full-train `stats.yaml` at N<400 would consume labels from
   high-fidelity samples outside the stated budget.
3. Use the subset's stats for train, validation, and test transformation.  The
   validation/test labels must never contribute to those stats.
4. A fixed `raw_pos_min=-40`, `raw_pos_max=80` is acceptable only when declared
   as an a-priori CFD-domain coordinate bound; it is input metadata, not a label
   statistic.  Otherwise position extrema must come only from subset inputs.
5. Put the subset-manifest SHA in every stats filename, run ID, and cache key.
6. Keep test completely absent from training processes.  Create it only in a
   frozen, one-shot evaluation process after all decisions have been made.

## Evidence boundary

The remote audit established directory counts, file-name consistency, sizes,
and the identity of the missing hidden-test IDs through bounded metadata queries.
It did not stream all 357 GB or numerically inspect all 484 runs.  The first NCSA
stage must therefore run a manifest/file/finite-value preflight before expensive
training, and it must report failures rather than silently dropping runs.
