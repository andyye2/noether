#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.sample_processors import SupernodeSamplingSampleProcessor


@pytest.fixture
def sample_data():
    return [
        {
            "pointcloud": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "pressure": torch.tensor([10.0, 20.0, 30.0]),
        },
        {
            "pointcloud": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]),
            "pressure": torch.tensor([40.0, 50.0, 60.0, 70.0]),
        },
    ]


def test_supernode_sampling_precollator_basic(sample_data):
    torch.random.manual_seed(42)
    items_at_supernodes = {"pointcloud", "pressure"}
    precollator = SupernodeSamplingSampleProcessor(
        item="pointcloud", num_supernodes=2, items_at_supernodes=items_at_supernodes
    )
    for sample in sample_data:
        processed_sample = precollator(sample)
        assert "supernode_idx" in processed_sample
        assert len(processed_sample["supernode_idx"]) == 2
        assert torch.all(
            sample["pointcloud"][processed_sample["supernode_idx"]] == processed_sample["supernode_pointcloud"]
        )
        assert torch.all(
            sample["pressure"][processed_sample["supernode_idx"]] == processed_sample["supernode_pressure"]
        )


def test_supernode_sampling_precollator_single_supernode(sample_data):
    torch.random.manual_seed(42)
    items_at_supernodes = {"pointcloud", "pressure"}
    precollator = SupernodeSamplingSampleProcessor(
        item="pointcloud", num_supernodes=1, items_at_supernodes=items_at_supernodes
    )
    for sample in sample_data:
        processed_sample = precollator(sample)
        assert "supernode_idx" in processed_sample
        assert len(processed_sample["supernode_idx"]) == 1


def test_supernode_sampling_precollator_no_supernodes(sample_data):
    precollator = SupernodeSamplingSampleProcessor(item="pointcloud", num_supernodes=0)

    for sample in sample_data:
        processed_sample = precollator(sample)
        assert "supernode_idx" in processed_sample
        assert len(processed_sample["supernode_idx"]) == 0


def test_preprocess_itemsatsupernodes():
    old_sample = {
        "input_position": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        "input_pressure": torch.tensor([[2.0], [3.0], [4.0]]),
    }
    precollator = SupernodeSamplingSampleProcessor(
        item="input_position",
        num_supernodes=2,
        items_at_supernodes={"input_pressure"},
    )
    new_sample = precollator(old_sample)
    assert len(new_sample) == 4
    assert "input_position" in new_sample
    assert "input_pressure" in new_sample
    assert "supernode_idx" in new_sample
    assert "supernode_input_pressure" in new_sample

    assert new_sample["input_position"].shape == (3, 2)
    assert new_sample["input_pressure"].shape == (3, 1)
    assert new_sample["supernode_idx"].shape == (2,)
    assert new_sample["supernode_input_pressure"].shape == (2, 1)


def test_preprocess_nondeterministic():
    torch.manual_seed(0)
    sample = {"pos": torch.rand(10, 3)}
    precollator = SupernodeSamplingSampleProcessor(item="pos", num_supernodes=4)

    processed1 = precollator(sample)
    processed2 = precollator(sample)

    assert len(processed1["supernode_idx"]) == len(processed2["supernode_idx"])
    assert not torch.equal(processed1["supernode_idx"], processed2["supernode_idx"])


def test_preprocess_deterministic():
    sample = {"pos": torch.rand(10, 3), "index": 0}
    precollator = SupernodeSamplingSampleProcessor(item="pos", num_supernodes=4, seed=0)

    processed1 = precollator(sample)
    processed2 = precollator(sample)

    assert len(processed1["supernode_idx"]) == len(processed2["supernode_idx"])
    assert torch.equal(processed1["supernode_idx"], processed2["supernode_idx"])


def test_denormalize():
    precollator = SupernodeSamplingSampleProcessor(item="pointcloud", num_supernodes=2)
    value = torch.tensor([1, 2, 3])
    with pytest.raises(NotImplementedError):
        new_key, same_value = precollator.inverse(key="pointcloud", value=value)
        # This should raise an error since denormalization is not implemented for this collator.
