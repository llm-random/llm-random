from typing import Optional, cast
from functools import partial

from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch
from torch.distributed.fsdp import wrap

from lizrd.core.misc import Noop
from research.conditional.moe_layers.expert_choice import ExpertGating
from lizrd.core.llm import AttentionMechanism


def custom_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
) -> bool:
    return nonwrapped_numel >= min_num_params


def _monkey_wrap(
    module: nn.Module,
    auto_wrap_policy,
    wrapper_cls,
    ignored_modules,
    ignored_params,
    only_wrap_children=False,
    **kwargs,
):
    """
    Wraps submodules of ``module`` for which ``auto_wrap_policy`` returns
    ``True`` with ``wrapper_cls``.

    Args:
        module (nn.Module): Module to recursively wrap.
        auto_wrap_policy (Callable): A callable representing a policy that
            determines which modules to recursively wrap with ``wrapper_cls``.
        ignored_modules (Set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
    Returns:
        (nn.Module, int):
            ``module`` after wrapping and the numel recursively wrapped.
    """
    assert auto_wrap_policy is not None, "Must specify auto_wrap_policy."
    assert wrapper_cls is not None, "Must specify wrapper_cls"
    # Make sure no child is already wrapped.
    printed = False
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            if isinstance(child, cast(type, wrapper_cls)):
                if not printed:
                    print(f"CURRENT CONFLICTING CLASS: {module}")
                    printed = True
                print(f"!!!MONKEY!!! found conflicting class: {child}")
            # assert not isinstance(child, cast(type, wrapper_cls))
        except TypeError:
            # wrapper_cls is a function as opposed to a class type, just bypass above check.
            pass

    # We count all params, assuming none of them are already wrapped.
    nonwrapped_numel = sum(
        p.numel() for p in module.parameters() if p not in ignored_params
    )

    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, nonwrapped_numel=nonwrapped_numel):
        total_wrapped_numel = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _monkey_wrap(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                **kwargs,
            )
            setattr(module, name, wrapped_child)
            # Keep track of how many parameters have been wrapped
            total_wrapped_numel += num_wrapped_params
        # decide if we need to wrap the current module,
        # since the left over parameters exceed the number of params to wrap
        remainder = nonwrapped_numel - total_wrapped_numel
        if not only_wrap_children and auto_wrap_policy(
            module=module, recurse=False, nonwrapped_numel=remainder
        ):
            # Leaf node or final wrapping of the remainder both happen here.
            return wrap._wrap(module, wrapper_cls, **kwargs), nonwrapped_numel
        else:
            return module, total_wrapped_numel
    return module, 0

wrap._recursive_wrap = _monkey_wrap

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def wrap_in_fsdp(
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool = False,
    offload_params: bool = False,
    print_model: bool = False,
    output_cast_dtype: Optional[torch.dtype] = None,
):
    # print(f"wrapping module: {module} on rank: {rank}")

    def _create_single_fsdp_module(module_to_wrap, precision):
        return FSDP(
            module_to_wrap,
            device_id=rank,
            mixed_precision=MixedPrecision(
                param_dtype=precision,
                reduce_dtype=torch.float32,
                cast_forward_inputs=cast_inputs,
                _module_classes_to_ignore=(ExpertGating, AttentionMechanism)
            ),
            # cpu_offload=CPUOffload(offload_params=offload_params),
            auto_wrap_policy=partial(custom_auto_wrap_policy, min_num_params=int(1e04)),
        )

    counter = 0
    for _, child in module.named_modules():
        if isinstance(child, cast(type, FSDP)):
            print(f"found conflicting class: {child}")
            counter += 1
    print(f"FOUND {counter} conflicting classes")

    # if output_cast_dtype is not None:
    #     main_module = _create_single_fsdp_module(module, param_precision)
    #     casting_module = _create_single_fsdp_module(Noop(), output_cast_dtype)
    #     wrapped = nn.Sequential(main_module, casting_module)
    # else:
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
