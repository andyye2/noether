# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""The fixed cell and the four arms of the reported comparison.

The preregistration describes a much larger acquisition study.  What is
reported is one paired cell crossed with the two factors the audit isolated:
how the model is initialized, and how the target geometry is rendered into the
normalized coordinates the pretrained positional basis expects.

Rendering evidence (dry-run radius-graph audits at N=100, replicate 0, three
point seeds, ``max_degree=32``):

* ``frozen`` (radius 9.0) reproduces the original protocol default. A
  DrivAerML car spans only ~39 of the 1000 normalized units, so a radius of 9
  reaches roughly a quarter of the car: every supernode saturates the degree
  cap and the geometry graph stops encoding shape.
* ``source_matched`` (radius 0.1) restores the message-graph density the
  source checkpoint was pretrained with: mean degree 5.31, maximum 30,
  zero-neighbour fraction 0.0, against a ShapeNet-Car source mean near 5.8.

Both renderings keep ``position_scale`` at the pretrained 1000.0, because the
sincos and RoPE frequency buffers are restored from the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from aero_cfd.multi_fidelity.experiment import (
    CHECKPOINT_SUPERNODE_RADIUS,
    Budget,
    CoordinateFrame,
    GeometryRendering,
    Strategy,
    Task,
)
from aero_cfd.multi_fidelity.protocol import ProtocolBinding

#: Supernode radius that reproduces the source message-graph density.
SOURCE_MATCHED_SUPERNODE_RADIUS = 0.1

FROZEN_RENDERING = GeometryRendering(supernode_radius=CHECKPOINT_SUPERNODE_RADIUS)
SOURCE_MATCHED_RENDERING = GeometryRendering(supernode_radius=SOURCE_MATCHED_SUPERNODE_RADIUS)


@dataclass(frozen=True)
class PaperCell:
    """The single paired data cell every reported arm shares.

    The seeds are read from the preregistration rather than restated here, so
    the generated commands cannot drift from the protocol the runner validates
    them against.

    Attributes:
        subset_seed: Subset seed of the replicate.
        model_seed: Model seed of the replicate.
        task: Preregistered field set.
        sample_size: Training-subset size.
        replicate: Preregistered replicate label.
        coordinate_frame: Frame the dataset is expressed in.
        budget: Preregistered stopping rule.
    """

    subset_seed: int
    model_seed: int
    task: Task = "common"
    sample_size: int = 100
    replicate: int = 0
    coordinate_frame: CoordinateFrame = "shapenet"
    budget: Budget = "compute_matched"

    @classmethod
    def from_protocol(cls, protocol: ProtocolBinding, *, budget: Budget = "compute_matched") -> PaperCell:
        """Build the reported cell from the frozen preregistration.

        Args:
            protocol: Validated protocol binding.
            budget: Preregistered stopping rule to report under.

        Returns:
            The reported :class:`PaperCell`.
        """
        replicate = protocol.replicate(0)
        return cls(subset_seed=replicate.subset_seed, model_seed=replicate.model_seed, budget=budget)


@dataclass(frozen=True)
class Arm:
    """One reported arm of the comparison.

    Attributes:
        name: Short arm identifier used in file names and reports.
        strategy: ``scratch`` or ``finetune``.
        geometry: Position scale and supernode radius.
        wall_distance_feature: Give volume tokens their distance to the
            vehicle surface as a token-level input feature.
        role: Why the arm is in the comparison.
    """

    name: str
    strategy: Strategy
    geometry: GeometryRendering
    role: str
    wall_distance_feature: bool = False


#: The 2x2 design: initialization crossed with geometry rendering.  The
#: scratch arms are what make the transfer effect separable from the effect of
#: repairing the geometry graph.
#:
#: The ``-wd`` arms repeat the matched pair with the wall-distance input
#: feature.  They are a second 2x2 rather than two extra points, because the
#: feature changes what the model reads and a feature arm may only be compared
#: with another feature arm.
ARMS: tuple[Arm, ...] = (
    Arm(
        name="S-frozen",
        strategy="scratch",
        geometry=FROZEN_RENDERING,
        role="original scratch baseline; pairs with P-FT-frozen",
    ),
    Arm(
        name="P-FT-frozen",
        strategy="finetune",
        geometry=FROZEN_RENDERING,
        role="naive transfer; reproduces the measured negative transfer",
    ),
    Arm(
        name="S-matched",
        strategy="scratch",
        geometry=SOURCE_MATCHED_RENDERING,
        role="isolates the geometry repair from the transfer effect",
    ),
    Arm(
        name="P-FT-matched",
        strategy="finetune",
        geometry=SOURCE_MATCHED_RENDERING,
        role="the single audited improvement under test",
    ),
    Arm(
        name="S-matched-wd",
        strategy="scratch",
        geometry=SOURCE_MATCHED_RENDERING,
        wall_distance_feature=True,
        role="scratch reference for the feature; pairs with P-FT-matched-wd",
    ),
    Arm(
        name="P-FT-matched-wd",
        strategy="finetune",
        geometry=SOURCE_MATCHED_RENDERING,
        wall_distance_feature=True,
        role="transfer with the wall-distance feature; the arm under test",
    ),
)

ARM_NAMES = tuple(arm.name for arm in ARMS)


def arm_by_name(name: str) -> Arm:
    """Return one reported arm.

    Args:
        name: Arm identifier from :data:`ARM_NAMES`.

    Returns:
        The matching :class:`Arm`.

    Raises:
        KeyError: If the name is not a reported arm.
    """
    for arm in ARMS:
        if arm.name == name:
            return arm
    raise KeyError(f"unknown arm {name!r}; expected one of {list(ARM_NAMES)}")
