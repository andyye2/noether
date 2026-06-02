#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.data.pipeline.collator import Collator


def test_default_collator_call():
    default_collator = Collator()
    num_samples = 3
    dim = 100
    samples = [{"data": torch.tensor([i] * dim)} for i in range(num_samples)]
    batch = default_collator(samples)

    assert "data" in batch
    assert batch["data"].shape == (num_samples, dim)
    for i, sample in enumerate(samples):
        assert torch.all(sample["data"] == torch.tensor([i] * dim))
