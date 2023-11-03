from typing import Optional

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch


def wrap_in_fsdp(
    enabled: bool,
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool = False,
    offload_params: bool = False,
    print_model: bool = False,
):
    if not enabled:
        return module
    else:
        wrapped = FSDP(
            module,
            device_id=rank,
            mixed_precision=MixedPrecision(
                param_dtype=param_precision,
                reduce_dtype=torch.float32,
                cast_forward_inputs=cast_inputs,
            ),
            cpu_offload=CPUOffload(offload_params=offload_params),
        )
        if print_model:
            print("------- MODEL AFTER WRAPPING IN FSDP -------")
            print(wrapped)
            print("--------------------------------------------")
        return wrapped


def wrap_in_ddp(
    module: nn.Module,
    rank: int,
):
    return DDP(module=module, device_ids=[rank])
