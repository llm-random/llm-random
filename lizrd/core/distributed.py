from typing import Optional, Type, Sequence
from functools import partial

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, CPUOffload, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def wrap_in_ddp(
    module: nn.Module,
    rank: int,
):
    return DDP(module=module.to(f"cuda:{rank}"), device_ids=[rank])


def wrap_in_fsdp(
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool,
    mixed_precision_ignored_classes: Sequence[Type[nn.Module]],
    offload_params: bool,
    print_model: bool,
    min_num_params: int,
    modules_to_wrap: tuple[Type[nn.Module]],
    is_logging_process: bool,
):
    assert (modules_to_wrap is None and min_num_params is not None) or (
        modules_to_wrap is not None and min_num_params is None
    ), "The FSDP arguments `modules_to_wrap` and `min_num_params` are mutually exclusive. Either supply one, or the other."

    if modules_to_wrap is not None:
        wrap_policy = ModuleWrapPolicy(modules_to_wrap)
    else:
        wrap_policy = (
            partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
            if min_num_params is not None
            else size_based_auto_wrap_policy
        )

    wrapped = FSDP(
        module,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        device_id=rank,
        mixed_precision=MixedPrecision(
            param_dtype=param_precision,
            reduce_dtype=param_precision,
            cast_forward_inputs=cast_inputs,
            _module_classes_to_ignore=mixed_precision_ignored_classes,
        ),
        cpu_offload=CPUOffload(offload_params=offload_params),
        auto_wrap_policy=wrap_policy,
    )

    if print_model and is_logging_process:
        print("------- MODEL AFTER WRAPPING IN FSDP -------")
        print(wrapped)
        print("--------------------------------------------")

    return wrapped
