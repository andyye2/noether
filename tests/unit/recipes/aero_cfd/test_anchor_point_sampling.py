import torch

from aero_cfd.pipeline.sample_processors.anchor_point_sampling import AnchorPointSamplingSampleProcessor


def _split_prefix_and_postfix(item: str) -> tuple[str, str]:
    prefix, postfix = item.split("_", maxsplit=1)
    return prefix, postfix


def _unused_three_way_split(item: str) -> tuple[str, str, str]:
    return item, "", ""


def test_anchor_point_sampling_accepts_scalar_tensor_index() -> None:
    sample = {
        "surface_position": torch.rand(10, 3),
        "index": torch.tensor(0),
    }
    processor = AnchorPointSamplingSampleProcessor(
        items={"surface_position"},
        num_points=4,
        to_prefix_and_postfix=_split_prefix_and_postfix,
        to_prefix_midfix_postfix=_unused_three_way_split,
        seed=0,
    )

    processed1 = processor(sample)
    processed2 = processor(sample)

    assert torch.equal(processed1["surface_anchor_position"], processed2["surface_anchor_position"])
