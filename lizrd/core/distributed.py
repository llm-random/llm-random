from typing import Optional
from functools import partial

from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch

from research.conditional.moe_layers.expert_choice import ExpertGating
from lizrd.core.llm import AttentionMechanism
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def custom_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
) -> bool:
    return nonwrapped_numel >= min_num_params


def wrap_in_fsdp(
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool = False,
    offload_params: bool = False,
    print_model: bool = False,
):
    def _create_single_fsdp_module(module_to_wrap, precision):
        return FSDP(
            module_to_wrap,
            device_id=rank,
            mixed_precision=MixedPrecision(
                param_dtype=precision,
                reduce_dtype=torch.float32,
                cast_forward_inputs=cast_inputs,
                _module_classes_to_ignore=(
                    ExpertGating,
                    AttentionMechanism,
                    nn.LayerNorm,
                ),
            ),
            cpu_offload=CPUOffload(offload_params=offload_params),
            auto_wrap_policy=partial(custom_auto_wrap_policy, min_num_params=int(1e04)),
        )

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
