#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.distributed.config import is_managed
from noether.core.distributed.utils import (
    accelerator_to_device,
    check_single_device_visible,
    log_device_info,
    parse_devices,
)

from .managed import run_managed
from .unmanaged import run_unmanaged

__all__ = [
    "run",
    # --- from managed:
    "run_managed",
    # --- from unmanaged:
    "run_unmanaged",
    # --- from utils:
    "accelerator_to_device",
    "check_single_device_visible",
    "log_device_info",
    "parse_devices",
]


def run(main, devices=None, accelerator="gpu", master_port=None):
    if is_managed():
        run_managed(
            main=main,
            accelerator=accelerator,
            devices=devices,
        )
    else:
        run_unmanaged(
            main=main,
            accelerator=accelerator,
            devices=devices,
            master_port=master_port,
        )
