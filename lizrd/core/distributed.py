from typing import Optional
from functools import partial

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def wrap_in_fsdp(
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool,
    mixed_precision_ignore_classes: list,
    offload_params: bool,
    print_model: bool,
    min_num_params: int,
    modules_to_wrap: tuple[nn.Module],
):
    if modules_to_wrap is not None and min_num_params is not None:
        raise Exception(
            "The FSDP arguments `modules_to_wrap` and `min_num_params` are mutually exclusive. Either supply one, or the other."
        )
    if modules_to_wrap is None and min_num_params is None:
        raise Exception(
            "FSDP wrapping policy unspecified. Either `modules_to_wrap` or `min_num_params` must be supplied."
        )

    if modules_to_wrap is not None:

        def explicit_wrap_policy(
            module: nn.Module,
            recurse: bool,
            nonwrapped_numel: int,
            _modules_to_wrap: tuple[nn.Module],
        ) -> bool:
            return isinstance(module, _modules_to_wrap)

        wrap_policy = partial(explicit_wrap_policy, _modules_to_wrap=modules_to_wrap)

    else:
        wrap_policy = (
            partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
            if min_num_params is not None
            else size_based_auto_wrap_policy
        )

    wrapped = FSDP(
        module,
        device_id=rank,
        mixed_precision=MixedPrecision(
            param_dtype=param_precision,
            reduce_dtype=torch.float32,
            cast_forward_inputs=cast_inputs,
            _module_classes_to_ignore=mixed_precision_ignore_classes,
        ),
        cpu_offload=CPUOffload(offload_params=offload_params),
        auto_wrap_policy=wrap_policy,
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
    return DDP(module=module.to(f"cuda:{rank}"), device_ids=[rank])
