from typing import Optional

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch

from lizrd.core.misc import Noop


def wrap_module_in_fsdp(
    enabled: bool,
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool = False,
    offload_params: bool = False,
    print_model: bool = False,
):
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


def wrap_in_fsdp(
    enabled: bool,
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool = False,
    offload_params: bool = False,
    print_model: bool = False,
    cast_outputs_to: Optional[torch.dtype] = None,
):
    if not enabled:
        return module
    elif cast_outputs_to is not None:
        cast_module = wrap_module_in_fsdp(
            enabled=enabled,
            module=Noop(),
            rank=rank,
            param_precision=cast_outputs_to,
            cast_inputs=True,
        )
        main_module = wrap_module_in_fsdp(
            enabled=enabled,
            module=module,
            rank=rank,
            param_precision=param_precision,
            cast_inputs=cast_inputs,
            offload_params=offload_params,
            print_model=print_model,
        )
        return CastWrapper(module=main_module, cast_module=cast_module)
    else:
        return wrap_module_in_fsdp(
            enabled=enabled,
            module=module,
            rank=rank,
            param_precision=param_precision,
            cast_inputs=cast_inputs,
            offload_params=offload_params,
            print_model=print_model,
        )


class CastWrapper(nn.Module):
    def __init__(self, module: nn.Module, cast_module: nn.Module):
        super().__init__()
        self.module = module
        self.cast_module = cast_module

    def forward(self, *args, **kwargs):
        return self.cast_module(self.module(*args, **kwargs))


def wrap_in_ddp(
    module: nn.Module,
    rank: int,
):
    return DDP(module=module.to(f"cuda:{rank}"), device_ids=[rank])
