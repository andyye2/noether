#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
import platform

from noether.core.distributed.utils import (
    accelerator_to_device,
    check_single_device_visible,
    get_backend,
    log_device_info,
    parse_devices,
)

logger = logging.getLogger(__name__)


def run_unmanaged(main, devices: str | None, accelerator: str = "gpu", master_port: int | None = None):
    # single node run
    world_size, device_ids = parse_devices(
        accelerator=accelerator,
        devices=devices,
    )
    if world_size == 1:
        # single process
        if accelerator == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
        check_single_device_visible(accelerator=accelerator)
        log_device_info(accelerator=accelerator, device_ids=device_ids)
        device = accelerator_to_device(accelerator=accelerator)
        main(device=device)
    else:
        if devices is None:
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", f"[{','.join(str(i) for i in range(world_size))}]")

        if master_port is None:
            raise RuntimeError("master_port must be specified for multi-process unmanaged runs")
        # spawn multi process training
        logger.info(
            f"running multi process training on {world_size} processes (devices={devices} host={platform.uname().node}), master_port={master_port}"
        )
        # dont log device info as this would load torch on device0 and block the VRAM required for this
        # log_device_info(accelerator, device_ids)
        args = (accelerator, device_ids, master_port, world_size, main)
        from torch.multiprocessing import spawn

        spawn(_run_multiprocess, nprocs=world_size, args=args)


def _run_multiprocess(rank, accelerator, device_ids, master_port, world_size, main):
    # unmanaged is limited to single-node -> use "localhost"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    if accelerator == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[rank]
    check_single_device_visible(accelerator=accelerator)

    from torch.distributed import destroy_process_group, init_process_group

    init_process_group(
        backend=get_backend(accelerator, device_ids),
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    device = accelerator_to_device(accelerator)
    main(device=device)
    destroy_process_group()
