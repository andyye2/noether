#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

import torch

ScalarOrSequence = Sequence[float] | torch.Tensor | float | int
