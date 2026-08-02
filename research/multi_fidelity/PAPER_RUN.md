> **Operational notice (NCSA B2):** the approved deployment runs only
> `S-matched` and `P-FT-matched` in isolated `*_B2` namespaces. Follow
> [`B2_MATCHED_RUN.md`](B2_MATCHED_RUN.md) and its separate CPU/GPU batch
> scripts. The generic `drivaerml_paper_array.sbatch` entry point is retired;
> the four-arm sequence below is retained as the design record, not a runnable
> submission recipe for this deployment.

# Reported run: regenerating every number from a clean slate

This is the operational contract for the results this implementation reports.
Everything below is produced by the code in this repository, from the frozen
preregistration and the audited ShapeNet-Car source checkpoint, and lands in a
namespace keyed by the exact commit that produced it.

## What is regenerated and what is not

| Artifact | Status |
| --- | --- |
| Subset manifests, train-only statistics | regenerated under this commit |
| All four training arms | regenerated under this commit |
| Frozen validation exports, merged table, comparison report | regenerated under this commit |
| ShapeNet-Car source checkpoint | **input, never retrained** (SHA256 `261a46b7…`) |
| Earlier P1 / exploratory results on NCSA | historical, read-only, used only for old-versus-new comparison |

Nothing in this repository reads a previous run's outputs. The commit-keyed
output root makes an accidental overwrite impossible: a new commit writes to a
new directory, and the batch script refuses any command line that does not
name its own commit.

## The reported design

One paired data cell, crossed with the two factors the audit isolated:

| arm | initialization | supernode radius | role |
| --- | --- | --- | --- |
| `S-frozen` | random | 9.0 | original scratch baseline |
| `P-FT-frozen` | source trunk, fresh readouts | 9.0 | naive transfer; reproduces the measured negative transfer |
| `S-matched` | random | 0.1 | separates the geometry repair from the transfer effect |
| `P-FT-matched` | source trunk, fresh readouts | 0.1 | the single audited improvement under test |

Fixed for every arm: `task=common`, `N=100`, `replicate=0`,
`coordinate_frame=shapenet`, `budget=compute_matched` (40,000 updates),
`position_scale=1000`, evaluation split `val`, one manifest, one statistics
artifact, one model seed (7103), one data-loader seed, one evaluator.

`position_scale` is fixed at the pretrained 1000.0 for every arm. The sincos
and RoPE frequency buffers are restored from the source checkpoint, so a
transfer arm rendered at another scale would evaluate frozen frequency bands at
shifted phases; the code refuses that combination outright. The message-passing
neighbourhood is therefore rescaled through the radius instead:

* radius 9.0 leaves every supernode at the degree cap of 32, so the geometry
  graph carries no shape information (a DrivAerML car spans ~39 of the 1000
  normalized units, and 9 units reaches roughly a quarter of the car);
* radius 0.1 restores the pretraining message-graph density: mean degree 5.31,
  maximum 30, zero-neighbour fraction 0.0, against a ShapeNet-Car source mean
  near 5.8. In raw units that is 12 mm inside the fixed 120 m envelope.

Both numbers are recorded in every training sidecar, in every metric row, and
in the run ID, so no result can be mistaken for the other rendering.

## Sequence

Let `COMMIT` be the commit being run and

```
ARTIFACTS=/scratch/andyye2/ABUPT/multi_fidelity_artifacts/$COMMIT
OUTPUTS=/scratch/andyye2/ABUPT/outputs/multi_fidelity_paper/$COMMIT
REPO=/scratch/andyye2/ABUPT/multi_fidelity
DATA=/scratch/andyye2/data/drivaerml_subsampled_10x
```

Every step runs through `uv run --no-sync` with
`PYTHONPATH=$REPO:$REPO/src:$REPO/recipes/aero_cfd/src`.

1. **Manifests.** Requires a clean worktree and writes outside it.

   ```bash
   uv run --no-sync python -m research.multi_fidelity.tools.materialize_manifests \
       --output-dir "$ARTIFACTS/manifests"
   ```

2. **Statistics.** One artifact for the shared cell, fitted on train only.

   ```bash
   uv run --no-sync python -m research.multi_fidelity.tools.generate_statistics_commands \
       --repo-root "$REPO" --dataset-root "$DATA" \
       --manifest-root "$ARTIFACTS/manifests" --stats-root "$ARTIFACTS/statistics" \
       --output "$ARTIFACTS/commands/statistics.txt"
   sbatch --test-only --array=1-1 research/multi_fidelity/slurm/drivaerml_paper_array.sbatch \
       "$COMMIT" "$ARTIFACTS/commands/statistics.txt"
   ```

3. **Training.** Four arms, one array index each.

   ```bash
   uv run --no-sync python -m research.multi_fidelity.tools.generate_training_commands \
       --repo-root "$REPO" --dataset-root "$DATA" \
       --manifest-root "$ARTIFACTS/manifests" --stats-root "$ARTIFACTS/statistics" \
       --output-path "$OUTPUTS" --source-output-path /scratch/andyye2/ABUPT/outputs \
       --output "$ARTIFACTS/commands/training.txt"
   sbatch --test-only --array=1-4 research/multi_fidelity/slurm/drivaerml_paper_array.sbatch \
       "$COMMIT" "$ARTIFACTS/commands/training.txt"
   ```

   Add `--arms P-FT-matched` to submit a single arm, and `--budget smoke` for a
   ten-update pipeline check that writes into its own run IDs.

4. **Frozen evaluation.** Cells are discovered from the training sidecars, so
   this step cannot mislabel a run or evaluate it under the wrong geometry.

   ```bash
   uv run --no-sync python -m research.multi_fidelity.tools.generate_evaluation_commands \
       --repo-root "$REPO" --dataset-root "$DATA" \
       --training-output-root "$OUTPUTS" --eval-output-root "$OUTPUTS-eval" \
       --metrics-root "$ARTIFACTS/metrics" --split val \
       --output "$ARTIFACTS/commands/evaluation.txt"
   ```

   `--checkpoint-tags latest best_model.loss.val.total` additionally scores the
   best-validation checkpoint through the identical evaluator, which is how the
   final-raw versus best-validation gap is reported.

5. **Merge and report.**

   ```bash
   uv run --no-sync python -m research.multi_fidelity.tools.merge_metric_csvs \
       "$ARTIFACTS"/metrics/val/common/*.csv \
       --output "$ARTIFACTS/reports/val_common.csv" \
       --provenance "$ARTIFACTS/reports/val_common_merge.json"
   uv run --no-sync python -m research.multi_fidelity.tools.compare_arms \
       "$ARTIFACTS/reports/val_common.csv" \
       --baseline-method S --baseline-rendering ps1000-sr9 \
       --output-json "$ARTIFACTS/reports/comparison.json" \
       --output-markdown "$ARTIFACTS/reports/comparison.md"
   ```

## Preflight before any submission

* `git status --porcelain` is empty and the commit matches what the batch is given.
* `--dry-run` on one training command prints the resolved config and the audit.
* `research/multi_fidelity/tools/audit_radius_graph.py` on that dry-run confirms
  the degree distribution of the arm's geometry graph.
* `sbatch --test-only` for every array before the real submission.

Steps 1, 3, and the dry runs need no dataset and no cluster, so the chain can be
rehearsed locally before anything is uploaded: materialize the manifests into a
scratch directory, generate the command list against it, and run each generated
command with `--dry-run` appended. The batch script's request guards are covered
by `tests/unit/multi_fidelity/test_batch_guards.py` and also run off-cluster.

## Where the frozen protocol and reality differ

`research/multi_fidelity/experiment_protocol.yaml` is a preregistration and is
deliberately left byte-identical, so its SHA256 still binds the manifests and
statistics. Read it with these corrections:

* `evidence_scope.not_demonstrated` and `phases[].real_status: pending` predate
  execution. P1 and the exploratory single comparison have since run; their
  results live on NCSA under the commits that produced them.
* `methods.linear_probe`, `methods.gradual_unfreeze`, and `methods.lora` are not
  implemented by this runner. Only `scratch` and `finetune` are, which is what
  the reported comparison uses; the removed strategies were never executed.
* `analysis` describes an acquisition-curve study across five sample sizes and
  eight replicates. This implementation reports one cell, so AULC, sample
  efficiency, replicate-level sign-flip tests, and the success gate are out of
  scope and are not implemented here.
* `preprocessing.graph_radius: 9.0` records the original default. It is the
  factor under test, not a constant.
