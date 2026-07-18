# ShapeNet-Car AB-UPT source/checkpoint evidence audit

Audit date: 2026-07-16

## Decision-relevant conclusion

The existing ShapeNet-Car checkpoint is technically suitable for a controlled
DrivAerML initialization experiment.  When the obsolete readout tensors are
discarded and the current readouts are initialized from the target model, every
non-readout state tensor has an exact name and shape match.  This is not yet
evidence of semantic transfer: the two datasets use different vehicle families,
flow axes, CFD regimes, mesh densities, and output contracts.

The confirmatory comparison must therefore use the source-checkpoint
architecture for *both* scratch and transfer.  Comparing a transferred 10/2
model with the current native DrivAerML 6/6 model would confound initialization
with architecture.

## Provenance

- Source repository: `/home/feng/Projects/ABUPT/noether`.
- Local HEAD when audited: `8bcc3fff...`.
- `origin/main`: `0e1400bd...` (2026-04-24); this commit's old AB-UPT readout
  API and 265-key state schema match the 2026-04-25 run.
- The run used `store_code_in_output: false`, so commit provenance is a strong
  schema/config match rather than a checkpoint-embedded cryptographic fact.
- Primary source run: `/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train`.

Available source checkpoints:

| Source checkpoint | Selection | Epoch/update | SHA256 | Intended role |
|---|---|---:|---|---|
| `ab_upt_cp=latest_model.th` | fixed final epoch | 500 / 394,500 | `261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38` | confirmatory primary |
| `ab_upt_cp=best_model.loss.test.total_model.th` | ShapeNet test total | 144 / 113,616 | `221ce5a1a6803876b5d0fc6157a8f06aa5299abe104c4ee9fcbef3f0aaffc5f7` | sensitivity only |
| `ab_upt_ema=0.9999_cp=latest_model.th` | fixed final EMA | epoch 500 run | `f1200056bee2a2be2cac0f390bf9aaa049d1b1f29b680bab0721969ad8ba4aeb` | sensitivity only |

ShapeNet-Car has 789 train (`param1`-`param8`) and 100 test
(`param0`) samples, with no validation split.  The source run selected its best
model on `loss/test/total`; this does not leak DrivAerML information, but it does
make that source checkpoint a test-selected model.  The fixed epoch-500
checkpoint is consequently the cleaner confirmatory source.

## Source task and architecture

The source model consumes coordinates only (`use_physics_features=false`).  It
predicts surface pressure and volume velocity.  The resolved architecture is:

- hidden dimension 192, 3 attention heads, MLP dimension 768;
- geometry depth 1;
- physics blocks: `perceiver,self,cross,self,cross,self,cross,self,cross,self`;
- two decoder blocks in each of the surface and volume domains;
- supernode radius 9;
- geometry points/supernodes 3,586, surface anchors 3,586, volume anchors 4,096;
- position scale 1,000 and default RoPE maximum wavelength 10,000;
- Lion, learning rate `5e-5`, batch size 1, float16, 500 epochs.

The training summary reports 7,007,236 trainable parameters.  Its recorded run
minima are total loss 0.0081323, surface-pressure relative L2 0.046669, and
volume-velocity relative L2 0.011781.  These are source-run diagnostics, not
expected DrivAerML performance.

## Exact load compatibility

The old checkpoint has 265 state tensors.  A current, source-architecture target
has 269 because the old direct linear projection was replaced by final LayerNorm
plus Linear in each domain.

After resetting `backbone.domain_decoder_projections`:

- 261 state keys match exactly;
- there are no same-name shape mismatches;
- all non-readout trainable parameters match (7,006,464 parameters);
- including 74 non-parameter buffer elements, the audit JSON counts 7,006,538
  compatible state elements;
- common target: 7,008,004 trainable parameters and 7,008,078 state elements;
- full target: 7,009,355 trainable parameters and 7,009,429 state elements.

The distinction between trainable parameters and state elements matters: RoPE
frequency tensors and similar buffers are present in `state_dict` but are not
optimizer parameters. Confirmatory machine-readable evidence is in
`evidence/transfer_compatibility_latest.json` and
`evidence/strict_transfer_load_latest.json`; best-total is sensitivity-only.

The current native DrivAerML 6-physics/6-decoder architecture would transfer
only 5,227,008 of 8,788,811 parameters (59.4734%).  It is a useful secondary
architecture-transfer experiment, not the primary initialization experiment.

## Semantic alignment required before loading

ShapeNet's mean velocity is approximately `[0.003, -0.023, 17.546]`, so its flow
axis is +z.  DrivAerML's is approximately `[16.791, -0.038, 0.408]`, so its flow
axis is +x.  The canonical target-to-source permutation is

```text
(x_source, y_source, z_source) = (y_target, z_target, x_target)
tensor_aligned = tensor[..., [1, 2, 0]]
```

It must be applied before normalization to positions and all vector fields:
velocity, friction, vorticity, and (if later enabled) normals.  Component-wise
mean/std values must be computed in or permuted into the same frame.  The model
uses absolute sinusoidal positions and RoPE and is not rotation equivariant, so
axis alignment is a causal preprocessing choice rather than cosmetic metadata.

The Python presets use position scale 1,000, whereas the newer DrivAer YAML uses
scale 100,000 and RoPE maximum wavelength 40,000.  The source-compatible primary
experiment retains scale 1,000 and 10,000 wavelength.  A native-scale ablation
must re-initialize all RoPE omega buffers; otherwise strict loading silently
overwrites the target frequency configuration with source values.  Every scale
choice also needs a measured radius-graph neighbor-count audit.

## Loader contract

The existing `PreviousRunInitializer` is sufficient.  It loads with
`weights_only=True`, supports remove/instantiate patterns, and ends with strict
`load_state_dict`.  The source-compatible contract is:

```yaml
initializers:
  - kind: noether.core.initializers.PreviousRunInitializer
    output_path: /home/feng/Projects/ABUPT/outputs
    run_id: 2026-04-25_7d0mv
    stage_name: train
    model_name: ab_upt
    checkpoint_tag: latest
    patterns_to_remove:
      - backbone.domain_decoder_projections
    patterns_to_instantiate:
      - backbone.domain_decoder_projections
```

Copying the old shared output rows into the new readout is not functionally
equivalent because the current readout includes a LayerNorm.  The confirmatory
experiment therefore resets the entire readout; any row-copy experiment must be
an explicitly old-API exploratory ablation.

## Source-code evidence locations

- ShapeNet split and fields: `src/noether/data/datasets/cfd/shapenet_car/dataset.py`.
- Source stats: `src/noether/data/datasets/cfd/shapenet_car/stats.yaml`.
- Resolved source config: `/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train/hp_resolved.yaml`.
- Source training summary: `/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train/tracker/summary.yaml`.
- Current readout API: `src/noether/modeling/models/ab_upt.py`.
- Strict prior-run initializer: `src/noether/core/initializers/previous_run.py`.
- Compatibility audit tools: `research/multi_fidelity/tools/audit_checkpoint.py` and
  `research/multi_fidelity/tools/check_transfer_compatibility.py`.

## Div-free branch boundary

`ABUPT/shapenet-divu0/5-14` contains a smoke-test implementation that predicts a
vector potential and takes its curl; it contains no real pretrained checkpoint.
The standard ShapeNet velocity head is not a vector-potential head, so only the
trunk could be reused and the volume head would need fresh initialization.  This
branch is a later physics-constraint experiment, not evidence for the present
pretraining claim.
