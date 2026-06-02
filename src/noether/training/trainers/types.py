#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from dataclasses import dataclass

import torch


@dataclass
class TrainerResult:
    total_loss: torch.Tensor
    losses_to_log: dict[str, torch.Tensor] | None = None
    additional_outputs: dict[str, torch.Tensor] | None = None


LossResult = dict[str, torch.Tensor] | torch.Tensor | list[torch.Tensor]
