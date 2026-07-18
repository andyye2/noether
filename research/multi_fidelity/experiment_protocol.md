# Frozen experiment protocol

Status: `frozen_before_first_target_job`.

This document is the human-readable companion to `experiment_protocol.yaml`.
The YAML raw-file SHA256 is the study integrity anchor, not a complete runtime
source of truth. The strict runner parses and enforces only selected protocol
fields (study identity, replicate seeds, and the primary source tag/SHA).
Runtime facts are jointly defined by the frozen code, explicit CLI arguments,
the resolved config, manifests, subset-statistics artifacts, training
provenance sidecars, and evaluation audits. Any YAML edit changes the integrity
anchor and therefore requires manifests and dependent statistics to be
rematerialized before production.

## What would count as an answer?

The study asks a sample-efficiency question, not merely whether a pretrained
model converges faster. A transferred model saves high-fidelity data only if it
reaches a fixed held-out DrivAerML error with fewer labeled training runs than
an isomorphic scratch model under the same target-side optimizer-update budget.

The confirmatory contrast is P-FT versus scratch on the common-field learning
curve. The claim “ShapeNet pretraining reduces DrivAerML HF requirements” is
accepted only if all three gates hold:

1. common-field geometric AULC error ratio <=0.90 and its pointwise 95% CI upper
   bound is below 1;
2. at the scratch-N=200 error target, the estimated sample-efficiency ratio is
   at least 1.5 and its fully identifiable pointwise 95% CI lower bound exceeds
   1;
3. at N=400, the transfer/scratch pointwise 95% CI upper bound is below the 1.05
   negative-transfer margin.

A null or unidentifiable threshold crossing cannot pass condition 2. Faster
early optimization without condition 2 is a compute/optimization benefit, not
HF sample saving. This protocol is frozen before the first target job; no real P0/P1 result yet
supports or rejects these gates.

## Source checkpoint binding

The confirmatory source is exactly:

- tag: `latest`;
- filename: `ab_upt_cp=latest_model.th`;
- SHA256:
  `261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38`.

The strict runner reads the tag and SHA from the protocol, compares the SHA with
the audited code constant, resolves the exact file, and hashes its bytes. A CLI
argument cannot override this identity. ShapeNet has no validation split, so
the source test-selected best checkpoint and source EMA are planned
sensitivities only:

- `best_model.loss.test.total`:
  `221ce5a1a6803876b5d0fc6157a8f06aa5299abe104c4ee9fcbef3f0aaffc5f7`;
- `ema=0.9999_cp=latest`:
  `f1200056bee2a2be2cac0f390bf9aaa049d1b1f29b680bab0721969ad8ba4aeb`.

The current strict confirmatory runner deliberately rejects both source
sensitivities. They need a separate, explicitly nonconfirmatory execution path
before they can be run.

Evidence to date establishes checkpoint tensor-shape compatibility plus strict
initialization/config construction for source-compatible common/full models. It
does not establish a completed real DrivAerML train→checkpoint→frozen-eval
chain or semantic transfer.

## Paired nested ladders and integrity boundaries

For each of eight preregistered replicates, PCG64 permutes only the official 400
training IDs. N={25,50,100,200,400} are prefixes of that permutation. Scratch
and transfer use the same manifest, model/DataLoader seed, data order, and
validation designs. The train pipeline seed is `None` so each dataset visit
resamples geometry and anchors from the worker RNG; paired methods use the same
DataLoader seed and therefore the same reproducible sampling stream.

Each manifest records base-dataset indices, design IDs, official val/test IDs,
protocol raw SHA, implementation commit/dirty boolean, and canonical-payload
SHA. Consumers hash the complete manifest bytes to obtain the raw-file SHA and
record it downstream. The loader recomputes the canonical payload SHA and
verifies that positional indices resolve to the recorded official train IDs.

The manifest implementation binding is intentionally limited: it compares Git
commit and dirty boolean, but does not cryptographically identify the contents
of a dirty diff. Production therefore requires:

1. preserve this frozen protocol and the implementation in a clean commit;
2. rematerialize manifests with `implementation_git_dirty=false`;
3. write those manifests outside the Git worktree;
4. pass their directory explicitly through the statistics/training generators'
   `--manifest-root`;
5. recompute all dependent subset statistics.

Repository-resident dirty manifests are development evidence, not production
inputs.

## No label leakage through normalization

Each `(replicate,N,task,coordinate-frame)` cache cell uses target-field moments
from exactly its N selected training IDs. Full-train statistics at N<400 and
validation/test labels are forbidden. The scalar [-40,80] position range is
public CFD-domain input metadata, not a fitted label statistic.

The cache key remains
`[manifest_sha256, train_subset_size, task, coordinate_frame]`. The statistics
artifact records protocol/study identity, manifest raw SHA, selected IDs,
`train_subset_size`, frame, required field moments, and a train-only access
attestation. At configuration time the preset checks manifest raw SHA, N,
frame, leakage guard, and task-required statistic keys; the strict runner also
hashes the entire stats file into the training sidecar. A filename alone is not
an integrity guarantee.

## Tasks, architecture, and preprocessing

The confirmatory common task predicts surface pressure and volume velocity. The
gated full task adds surface wall shear stress, volume total-pressure
coefficient, and volume vorticity with fresh readout rows. The Noether property
named `volume_pressure` currently loads `volume_cell_totalpcoeff.pt`.

Scratch and transfer are identical 192-wide AB-UPTs with one geometry block, ten
physics blocks, two decoder blocks per domain, three heads, radius 9, coordinate
scale 1,000, and RoPE maximum wavelength 10,000. Only initialization differs.
Transfer strict-loads every compatible trunk tensor and resets the whole current
readout. Native DrivAer architecture/scale is a separate ablation.

The primary forward transform maps DrivAer `(x,y,z)` to ShapeNet `(y,z,x)`
before normalization for positions and all vector fields. Evaluation
inverse-normalizes predictions and targets to physical units but does not
inverse-rotate vectors to the DrivAer native frame. Reports must say “forward
coordinate transform plus inverse normalization,” not “inverse coordinate
transform.”

Before real P0, `audit_radius_graph.py` must consume a real strict-runner dry-run
resolved config, reconstruct the train dataset/pipeline, report capped
neighbor-degree quantiles and cap fraction, and fail above the preregistered
zero-neighbor fraction. A capped degree of 32 is not an uncapped physical
neighbor count.

## Methods and matched budgets

- **S:** random initialization, all layers trained.
- **P-FT:** protocol-locked source trunk plus fresh readouts, all layers trained
  from update 0.
- **P-LP:** fresh readouts only; a representation diagnostic.
- **P-GU:** readouts for 0-10%, readouts+domain decoders for 10-30%, then all
  parameters.
- **P-LoRA:** gated planned extension across all linear layers including the
  geometry encoder; not currently implemented in the strict runner.

The primary budget fixes 40,000 optimizer updates at effective batch size 1:

| N | Epochs | Updates |
|---:|---:|---:|
| 25 | 1,600 | 40,000 |
| 50 | 800 | 40,000 |
| 100 | 400 | 40,000 |
| 200 | 200 | 40,000 |
| 400 | 100 | 40,000 |

The practical-cost control fixes 100 epochs, yielding 2,500-40,000 updates.
There is no early stopping and no target EMA. The target `latest` checkpoint is
the final-raw primary endpoint; `best_model.loss.val.total` is a target
best-validation sensitivity. These target checkpoint identities are distinct
from the planned source checkpoint sensitivities above. Test data are absent
from training.

## Frozen validation/test evaluation

The evaluator instantiates exactly one official split per run. Its direct CLI
defaults to `val`; the evaluation-command generator requires an explicit
`--split val|test`. Test access has two code gates: the generator requires
`--confirm-test-release` before producing test commands, and every generated
test command carries the same flag for an independent runner check.

Evaluation cells may only be discovered from
`<training-output-root>/**/training_provenance.json`. Scientific labels are
inherited and validated from the sidecar: task, strategy→method, replicate, N,
frame, budget, manifest/stats paths, run/stage, and checkpoint hashes. Filters
include methods, replicates, N, tasks, budgets, and frames. The primary release
must explicitly use `compute_matched`, `shapenet`, and checkpoint tag `latest`;
`best_model.loss.val.total` is the target checkpoint sensitivity. Sidecar and
actual checkpoint SHA must both match.

The primary runtime policy uses 16,384 surface and 16,384 volume anchor/output
points per design with seed 4242. These are not full-grid predictions and no
materialized index artifact is claimed. Reproducibility requires identical
code, PyTorch version, and design order within a release.

Each evaluation audit records the official split IDs and count. The official 50
test IDs must be proved from those audits before merging/analyzing results.
`analyze_transfer_results.py` requires an explicit frozen-test confirmation and
enforces the eight replicate IDs, five N values, and exact official 50 test IDs.
Chunked dense full-grid sensitivity is `planned_not_implemented`.

Metrics are computed after inverse normalization in physical units, in the
selected coordinate frame. The per-design/per-field primary metric is relative
L2; MAE is also reported. Fields, vector components, anchors, and mesh points
are not independent statistical observations.

## Common-task statistical analysis

The common endpoint is the equal-field mean of log relative-L2 for pressure and
velocity. At each N, report the paired geometric P-FT/S error ratio. Integrate
log error over `log2(N)` for normalized AULC. Estimate the N needed to reach the
scratch-N=200 isotonic error target without extrapolation.

Confidence intervals use at least 10,000 **paired two-way crossed bootstrap**
draws that resample training replicates and test designs while preserving the
same indices across method, N, and field. This is not described as a nested
hierarchical bootstrap.

Both the primary AULC and every N receive a one-sided exact replicate-level
paired sign-flip enumeration with the alternative that transfer error is lower.
Holm adjustment applies only to the family of per-N common-task p-values. The
single AULC p-value is not in that family. Every bootstrap CI is pointwise and
unadjusted for multiplicity; no simultaneous CI is claimed.

If any sample-efficiency bootstrap draw cannot identify the threshold crossing
inside observed N, the implementation withholds the ratio CI. Such a null/
unidentifiable crossing cannot pass the success gate.

The preregistered full field×N Holm family is
`not_implemented`. P3 training is gated by the common result, and confirmatory
P3 inference remains blocked until that multiplicity procedure is implemented,
tested, and frozen. Descriptive full-field output is not a confirmatory pass.

## Staged release and futility boundary

P0 covers loader, source checksum/strict load, real dry-run, finite-loss smoke,
and graph preflight. P1 runs S versus P-FT for the first three replicates. P2
adds five replicates after P1; P2a is fixed-epoch control; P2b tests
probe/unfreezing; P2c tests raw axes. P3 is released only after the common gate,
and its inference remains subject to the unimplemented full family above.

The validation futility rule uses N=50/100 and the first three paired
replicates. Stop raw P-FT only if both per-N median ratios are >=1.10 and all six
ratios exceed 1. `assess_validation_gate.py` validates the rectangular grid,
required N/replicate counts, and numerical rule. Its long-table CSV cannot prove
split, final-raw/latest selection, or non-EMA provenance;
`--confirm-final-raw-val` is a caller assertion. A real gate decision must also
verify the training sidecars, each validation evaluation audit, and merge
provenance. Test data never drive a gate.

A method with algorithmic NaN/divergence in at least two of three pilot runs
stops; an infrastructure failure may be retried once with the exact same
manifest and seeds. Risk controls are preregistered and partly implemented, but
their real effectiveness remains pending P0/P1.

## Implemented chain and deployment prerequisites

Implemented components include manifest materialization, train-only subset
statistics, forward frame conversion, strict training/config construction,
training provenance, real-config radius-graph audit, statistics/training command
generators with external manifest roots, provenance-sourced evaluation command
generation, single-split frozen evaluation plus evaluation audit, deterministic
CSV merge plus SHA provenance, validation-gate arithmetic, and common-task
analysis.

This list does not imply a completed real target run. Before a real strict
dry-run, the final clean code, external clean manifests, required subset stats,
dataset path, and protocol-locked source checkpoint must all be staged. The
strict dry-run reads and validates these artifacts during config construction.
Final offline verification is reported in the execution conclusion rather than
as a hard-coded test count here.

Define `C0` by profiling the exact N=400, 40k-update source-compatible model on
the actual target GPU. The existing approximate 7 H100-hour showcase uses a
different architecture and is only an optimistic reference. Release allocation
phase by phase; do not interpret the budget as evidence that risk controls have
already passed.
