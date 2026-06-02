#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel


class BasePipeline(BaseModel):
    kind: str
    default_collate_modes: list[str]
