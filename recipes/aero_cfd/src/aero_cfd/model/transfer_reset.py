# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Named scopes of AB-UPT parameters to re-initialize when transferring a trunk.

Transfer keeps the source trunk and re-initializes a chosen part of the model.
The readout is always re-initialized because the target predicts different
fields, so it is architecturally incompatible with the source; every other
choice is an experimental variable.

The named scopes below follow the domain structure of
:class:`~noether.modeling.models.ab_upt.AnchorBranchedUPT`. ``domain_biases``,
``domain_decoder_blocks`` and ``domain_decoder_projections`` are
:class:`torch.nn.ModuleDict` instances keyed by domain name, so each domain owns
a disjoint set of parameters, while ``physics_blocks``, ``encoder`` and
``geometry_blocks`` are shared by every domain. Resetting a single domain
therefore cannot change another domain's prediction except through the gradients
that reach the shared trunk.

For the frozen ShapeNet-Car source (7,008,004 trainable parameters) the scopes
resolve to:

============== =========== ==============================================
scope          parameters  share of the model
============== =========== ==============================================
readout              1,540  0.022%
volume_decoder     891,268  12.718%
volume_path        965,380  13.775%
decoder          1,780,996  25.414%
============== =========== ==============================================

Verify these against a concrete checkpoint with
``research/multi_fidelity/tools/verify_strict_transfer_load.py``, which performs
the same remove/instantiate/strict-load operation the initializer performs.
"""

from __future__ import annotations

#: Per-domain output projections. Always reset: the target fields differ.
READOUT_PATTERN = "backbone.domain_decoder_projections"

#: Per-domain decoder blocks, both domains.
DECODER_BLOCKS_PATTERN = "backbone.domain_decoder_blocks"

#: Per-domain decoder blocks of the volume domain only.
VOLUME_DECODER_BLOCKS_PATTERN = "backbone.domain_decoder_blocks.volume"

#: MLP that turns a volume position embedding into the volume domain bias.
VOLUME_BIAS_PATTERN = "backbone.domain_biases.volume"

#: Substring patterns re-initialized by each named scope.
RESET_SCOPES: dict[str, tuple[str, ...]] = {
    "readout": (READOUT_PATTERN,),
    "volume_decoder": (READOUT_PATTERN, VOLUME_DECODER_BLOCKS_PATTERN),
    "volume_path": (READOUT_PATTERN, VOLUME_DECODER_BLOCKS_PATTERN, VOLUME_BIAS_PATTERN),
    "decoder": (READOUT_PATTERN, DECODER_BLOCKS_PATTERN),
}

#: Scope of the frozen confirmatory study; keeps published run identities stable.
DEFAULT_RESET_SCOPE = "readout"


def reset_patterns(scope: str) -> list[str]:
    """Return the substring patterns re-initialized by one named scope.

    The patterns are matched with ``pattern in key`` by
    :class:`~noether.core.initializers.previous_run.PreviousRunInitializer`, both
    against the source checkpoint keys it drops and against the freshly built
    model keys it reinstates.

    Args:
        scope: A key of :data:`RESET_SCOPES`.

    Returns:
        The patterns of that scope, in declaration order.

    Raises:
        ValueError: If ``scope`` is not a known scope name.

    Example:

        .. testcode::

            from aero_cfd.model.transfer_reset import reset_patterns

            print(reset_patterns("volume_decoder"))

        .. testoutput::

            ['backbone.domain_decoder_projections', 'backbone.domain_decoder_blocks.volume']
    """
    if scope not in RESET_SCOPES:
        raise ValueError(f"unknown reset scope {scope!r}; known scopes are {sorted(RESET_SCOPES)}")
    return list(RESET_SCOPES[scope])
