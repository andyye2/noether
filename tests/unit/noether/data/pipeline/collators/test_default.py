#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.data.pipeline.collators.default import DefaultCollator


def test_default_collator_init_with_valid_items():
    items = ["item1", "item2", "item3"]
    collator = DefaultCollator(items)
    assert collator.items == items


def test_default_collator_init_with_empty_items():
    items = []
    collator = DefaultCollator(items)
    assert collator.items == items


def test_default_collator_call():
    items = [
        "item1",
        "item2",
    ]

    collator = DefaultCollator(items)

    samples = [
        {"item1": torch.rand(10, 10, 3), "item2": torch.rand(1)},
        {"item1": torch.rand(10, 10, 3), "item2": torch.rand(1)},
        {"item1": torch.rand(10, 10, 3), "item2": torch.rand(1)},
    ]

    batch = collator(samples)

    assert "item1" in batch
    assert "item2" in batch
    assert torch.equal(batch["item1"], torch.stack([sample["item1"] for sample in samples]))
    assert batch["item1"].shape == torch.Size([3, 10, 10, 3])
    assert torch.equal(batch["item2"], torch.stack([sample["item2"] for sample in samples]))
    assert batch["item2"].shape == torch.Size([3, 1])


def test_default_collator_call_noitems():
    collator = DefaultCollator(optional_items=["item2"])
    sample1 = {"item1": torch.rand(10, 10, 3), "item2": torch.rand(1)}
    sample2 = {"item1": torch.rand(10, 10, 3)}

    batch1 = collator([sample1])
    batch2 = collator([sample2])

    assert "item1" not in batch1
    assert "item2" in batch1
    assert "item1" not in batch2
    assert "item2" not in batch2
    assert torch.equal(batch1["item2"], sample1["item2"].unsqueeze(0))
    assert batch1["item2"].shape == torch.Size([1, 1])


def test_default_collator_call_noitems_and_nooptionalitems():
    collator = DefaultCollator()
    sample = {"item1": torch.rand(10, 10, 3), "item2": torch.rand(1)}
    batch = collator([sample])
    assert len(batch) == 0


def test_default_collator_call_optionalitems():
    collator = DefaultCollator(items=["item1"], optional_items=["item2"])
    sample1 = {"item1": torch.rand(10, 10, 3), "item2": torch.rand(1)}
    sample2 = {"item1": torch.rand(10, 10, 3)}

    batch1 = collator([sample1])
    batch2 = collator([sample2])

    assert "item1" in batch1
    assert "item2" in batch1
    assert "item1" in batch2
    assert "item2" not in batch2
    assert torch.equal(batch1["item1"], sample1["item1"].unsqueeze(0))
    assert batch1["item1"].shape == torch.Size([1, 10, 10, 3])
    assert torch.equal(batch2["item1"], sample2["item1"].unsqueeze(0))
    assert batch2["item1"].shape == torch.Size([1, 10, 10, 3])
    assert torch.equal(batch1["item2"], sample1["item2"].unsqueeze(0))
    assert batch1["item2"].shape == torch.Size([1, 1])
