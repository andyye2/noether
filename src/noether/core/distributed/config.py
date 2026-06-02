#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os

import torch.distributed as dist

__all__ = [
    "DistributedConfig",
    "set_config",
    "is_managed",
    "get_local_rank",
    "get_world_size",
    "get_rank",
    "get_num_nodes",
    "get_managed_world_size",
    "get_managed_rank",
    "is_distributed",
    "is_rank0",
    "is_data_rank0",
    "is_local_rank0",
    "barrier",
]


class DistributedConfig:
    @staticmethod
    def is_managed() -> bool:
        return "SLURM_PROCID" in os.environ

    def get_local_rank(self) -> int:
        if "SLURM_LOCALID" in os.environ:
            return int(os.environ["SLURM_LOCALID"])
        return self.get_rank()

    def get_num_nodes(self) -> int:
        if "SLURM_JOB_NUM_NODES" in os.environ:
            return int(os.environ["SLURM_JOB_NUM_NODES"])
        return 1

    def get_managed_world_size(self) -> int:
        return self.get_num_nodes() * int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))

    def get_managed_rank(self) -> int:
        if "SLURM_PROCID" in os.environ:
            return int(os.environ["SLURM_PROCID"])
        raise RuntimeError("SLURM_PROCID not found in environment, cannot determine managed rank")

    @staticmethod
    def is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def get_rank(self) -> int:
        if self.is_distributed():
            return dist.get_rank()
        return 0

    def get_world_size(self) -> int:
        if self.is_distributed():
            return dist.get_world_size()
        return 1

    def is_data_rank0(self) -> bool:
        # data has to be copied in 2 cases
        # - is_local_rank0: single-gpu, multi-gpu, multi-gpu SLURM
        #   - process with is_local_rank0 copies the data
        #   - other processes have to wait for the copying to finish via barrier
        # - get_world_size == 1: SLURM runs that are not using multi-gpu require every process to copy data
        #   - no guarantee that the processes use the same dataset
        #   - avoid race conditions
        return self.is_local_rank0() or self.get_world_size() == 1

    def is_rank0(self) -> bool:
        return self.get_rank() == 0

    def is_local_rank0(self) -> bool:
        return self.get_local_rank() == 0

    def barrier(self) -> None:
        if self.is_distributed():
            dist.barrier()


_config: DistributedConfig = DistributedConfig()


def set_config(new_config: DistributedConfig):
    global _config
    _config = new_config


def is_managed():
    return _config.is_managed()


def get_local_rank():
    return _config.get_local_rank()


def get_num_nodes():
    return _config.get_num_nodes()


def get_managed_world_size():
    return _config.get_managed_world_size()


def get_managed_rank():
    return _config.get_managed_rank()


def is_distributed():
    return _config.is_distributed()


def get_rank():
    return _config.get_rank()


def get_world_size():
    return _config.get_world_size()


def is_data_rank0():
    return _config.is_data_rank0()


def is_rank0():
    return _config.is_rank0()


def is_local_rank0():
    return _config.is_local_rank0()


def barrier():
    return _config.barrier()
