#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline import BatchProcessor, Collator, MultiStagePipeline, SampleProcessor


class DummySampleProcessor(SampleProcessor):
    def __init__(self, item=None):
        self.item = item

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        x = sample["data"]

        if isinstance(x, torch.Tensor):
            x = x.clone().detach()
        else:
            x = torch.as_tensor(x)

        sample["data"] = x * 2
        return sample


class DummyCollator(Collator):
    def __call__(self, samples: list[dict[str, any]]) -> dict[str, torch.Tensor]:
        batch = {"data": torch.concat([s["data"] for s in samples], dim=0)}
        return batch


class DummyBatchProcessor(BatchProcessor):
    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        sample["data"] = sample["data"].sum(dim=0)
        return sample


def test_multistage_pipeline_call():
    """Tests the __call__ method with one of each collator type."""
    # Mock collators
    multistage_pipeline = MultiStagePipeline(
        sample_processors=[DummySampleProcessor()],
        collators=[DummyCollator()],
        batch_processors=[DummyBatchProcessor()],
    )
    num_samples = 3
    # Create a dummy input
    samples = [{"data": torch.tensor([i])} for i in range(num_samples)]

    # Call the multistage collator
    batch = multistage_pipeline(samples)

    assert "data" in batch
    assert batch["data"] == torch.tensor([6])  # (0*2 + 1*2 + 2*2) = 6
    for i, sample in enumerate(samples):
        assert sample["data"] == torch.tensor([i])


def test_multistage_pipeline_no_collators():
    """Tests that an error is raised if no collators are provided."""
    pipeline = MultiStagePipeline(
        sample_processors=[DummySampleProcessor()], collators=[], batch_processors=[DummyBatchProcessor()]
    )
    assert len(pipeline.collators) == 1  # Default collator is used


def test_get_precollator():
    """Tests the get_precollator method."""
    precollator1 = DummySampleProcessor(item="a")
    precollator2 = DummySampleProcessor(item="b")
    multistage_pipeline = MultiStagePipeline(
        sample_processors=[precollator1, precollator2],
        collators=[DummyCollator()],
        batch_processors=[DummyBatchProcessor()],
    )

    # Test retrieving by type
    found = multistage_pipeline.get_sample_processor(lambda p: isinstance(p, DummySampleProcessor) and p.item == "a")
    assert found == precollator1

    found = multistage_pipeline.get_sample_processor(lambda p: isinstance(p, DummySampleProcessor) and p.item == "b")
    assert found == precollator2

    # Test error when no match
    with pytest.raises(ValueError):
        multistage_pipeline.get_sample_processor(lambda p: False)

    # Test error when multiple matches
    with pytest.raises(ValueError):
        multistage_pipeline.get_sample_processor(lambda p: isinstance(p, DummySampleProcessor))
