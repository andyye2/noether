#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from enum import Enum


class Provider(str, Enum):
    HUGGINGFACE = "huggingface"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
