#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.sample_processors import MomentNormalizationSampleProcessor
from noether.modeling.functional.logscale import to_logscale


@pytest.fixture
def precollator():
    return MomentNormalizationSampleProcessor(item="pressure", mean=[2.0, 3.0], std=[1.0, 2.0])


def test_call_normalizes_batch(precollator):
    sample = {
        "pressure": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        "unchanged": torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
    }
    new_sample = precollator(sample)

    assert torch.allclose(
        new_sample["pressure"],
        (sample["pressure"] - torch.tensor([2.0, 3.0])) / torch.tensor([1.0, 2.0]),
    )
    assert torch.all(new_sample["unchanged"] == sample["unchanged"])


def test_call_raises_key_error_for_missing_item(precollator):
    sample = {"unchanged": torch.tensor([[3.0, 4.0], [5.0, 6.0]])}
    with pytest.raises(KeyError):
        precollator(sample)


def test_denormalize(precollator):
    sample = {
        "pressure": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        "unchanged": torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
    }
    normalized_sample = precollator(sample)
    denormalized_sample = {}
    for key, value in normalized_sample.items():
        new_key, new_value = precollator.inverse(key=key, value=value)
        assert new_key == key
        denormalized_sample[key] = new_value
    assert torch.equal(denormalized_sample["pressure"], sample["pressure"])
    assert torch.equal(denormalized_sample["unchanged"], sample["unchanged"])


def test_logscale():
    precollator = MomentNormalizationSampleProcessor(
        item="pressure",
        logmean=[0.0, 0.0],
        logstd=[1.0, 1.0],
        logscale=True,
    )
    sample = {"pressure": torch.tensor([[3.0, 4.0], [5.0, 6.0]])}
    normalized_sample = precollator(sample)
    assert torch.equal(normalized_sample["pressure"], to_logscale(sample["pressure"]))
    denormalized_sample = {}
    for key, value in normalized_sample.items():
        new_key, new_value = precollator.inverse(key=key, value=value)
        assert new_key == key
        denormalized_sample[key] = new_value
    assert torch.allclose(denormalized_sample["pressure"], sample["pressure"])


def test_denormalize_noop(precollator):
    og_sample = {"unchanged": torch.tensor([[3.0, 4.0], [5.0, 6.0]])}
    denormalized_sample = {}
    for key, value in og_sample.items():
        new_key, new_value = precollator.inverse(key=key, value=value)
        assert new_key == key
        denormalized_sample[key] = new_value
    assert torch.equal(denormalized_sample["unchanged"], og_sample["unchanged"])
