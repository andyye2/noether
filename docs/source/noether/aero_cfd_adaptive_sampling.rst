Adaptive Volume Anchor Sampling
===============================

The ShapeNet-Car AB-UPT preset can optionally use score-aware sampling for
volume anchor points. The default behavior is unchanged: volume anchors are
sampled uniformly unless adaptive sampling is explicitly enabled.

Overview
--------

Adaptive volume anchor sampling uses a per-point score sidecar to bias only the
training volume anchor distribution. Validation and test pipelines keep the
uniform sampler so evaluation metrics are not sampled from a biased point set.

For each selected anchor point, the sampler also emits:

- ``volume_anchor_sampling_prob``: the true selection probability ``p_i``.
- ``volume_anchor_sampling_weight``: the inverse-probability correction
  ``1 / (N * p_i)``.

The recipe-local ``SamplingWeightedLossTrainer`` consumes the weight key when
enabled. Without ``sample_weight_keys``, it behaves like the base weighted-loss
trainer.

Configuration
-------------

The relevant ShapeNet-Car preset arguments are:

.. code-block:: python

   use_volume_score_sampling=True
   volume_score_dir="/path/to/score_sidecars"
   volume_score_refresh_every_n_epochs=5
   volume_score_uniform_fraction=0.3
   volume_score_gamma=1.0
   volume_score_eps=1e-8

The score sidecar loader verifies that each sidecar matches the current sample
index and volume-position fingerprint before injecting ``volume_sampling_score``.
Missing sidecars, invalid scores, or fingerprint mismatches fall back to uniform
anchor sampling.

Score Refresh
-------------

When ``volume_score_refresh_every_n_epochs`` is set, the preset adds a
``score_refresh_train`` dataset. The refresh callback evaluates full volume
points as queries in chunks while keeping the anchor context fixed for the
sample. It writes one sidecar per sample with the residual score:

.. code-block:: text

   mean((pred_volume_velocity - target_volume_velocity) ** 2, dim=-1)

This keeps the score field aligned with the full ``volume_position`` ordering
used by the training pipeline.
