from typing import Optional

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch

from lizrd.core.misc import Noop


def wrap_in_fsdp(
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool = False,
    offload_params: bool = False,
    print_model: bool = False,
    output_cast_dtype: Optional[torch.dtype] = None,
):
    def _create_single_fsdp_module(module_to_wrap, precision):
        return FSDP(
            module_to_wrap,
            device_id=rank,
            mixed_precision=MixedPrecision(
                param_dtype=precision,
                reduce_dtype=torch.float32,
                cast_forward_inputs=cast_inputs,
            ),
            cpu_offload=CPUOffload(offload_params=offload_params),
        )

    if output_cast_dtype is not None:
        main_module = _create_single_fsdp_module(module, param_precision)
        casting_module = _create_single_fsdp_module(Noop(), output_cast_dtype)
        wrapped = nn.Sequential([main_module, casting_module])
    else:
        wrapped = _create_single_fsdp_module(module, param_precision)

    if print_model:
        print("------- MODEL AFTER WRAPPING IN FSDP -------")
        print(wrapped)
        print("--------------------------------------------")

    return wrapped


def wrap_in_ddp(
    module: nn.Module,
    rank: int,
):
    return DDP(module=module.to(f"cuda:{rank}"), device_ids=[rank])
