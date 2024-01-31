from typing import Callable, Optional, Union, Type

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from lizrd.core import llm
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function


def get_model(
    max_length: int,
    vocab_size: int,
    ff_layer_fun: Union[
        Callable[[], torch.nn.Module], list[Callable[[], torch.nn.Module]]
    ],
    attention_layer_fun: Callable[[], torch.nn.Module],
    dm: int,
    n_blocks: int,
    device: torch.device,
    init_type,
    init_scale,
    ddp_enabled: bool,
    fsdp_enabled: bool,
    fsdp_param_precision: torch.dtype,
    fsdp_mixed_precision_ignore_classes: list[Type[torch.nn.Module]],
    fsdp_offload_params: bool,
    fsdp_min_num_params: int,
    fsdp_modules_to_wrap: Union[tuple[Type[torch.nn.Module]], None],
    activation_checkpointing_modules: Union[tuple[Type[torch.nn.Module]], None],
    is_logging_process: bool,
    rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_layer = llm.EmbeddingLayer(
        llm.PositionalEmbedding(
            max_length, dm, init_type=init_type, init_scale=init_scale
        ).to(first_gpu),
        llm.TokenEmbedding(
            vocab_size, dm, init_type=init_type, init_scale=init_scale
        ).to(first_gpu),
    )

    layer_dict = {
        "attention": attention_layer_fun,
        "feedforward": ff_layer_fun,
    }
    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = llm.TransformerTower(
        n_blocks,
        dm,
        layer_dict,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn,
    )

    head = llm.PredictionHead(
        dm, vocab_size, init_type=init_type, init_scale=init_scale
    ).to(last_gpu)

    model = llm.LLM(embedding_layer, encoder_tower, head)
    breakpoint()
    if ddp_enabled:
        model = wrap_in_ddp(module=model, rank=rank)
    elif fsdp_enabled:
        model = wrap_in_fsdp(
            module=model,
            rank=rank,
            param_precision=fsdp_param_precision,
            cast_inputs=True,
            mixed_precision_ignored_classes=fsdp_mixed_precision_ignore_classes,
            offload_params=fsdp_offload_params,
            print_model=True,
            min_num_params=fsdp_min_num_params,
            modules_to_wrap=fsdp_modules_to_wrap,
            is_logging_process=is_logging_process,
        )

    if activation_checkpointing_modules is not None:
        check_fn = lambda x: isinstance(x, activation_checkpointing_modules)
        apply_activation_checkpointing(
            model,
            check_fn=check_fn,
            checkpoint_wrapper_fn=make_checkpoint_wrapper_function(),
        )

    return model
