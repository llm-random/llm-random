import logging
from functools import partial
from typing import Callable, Optional, Type, Union

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.nn import LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.profiler import ProfilerAction

from lizrd.core import llm
from lizrd.core.distributed import wrap_in_ddp, wrap_in_fsdp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights
from research.grad_norm.modules import (
    BlockGradModifPlacement,
    GradCaptureLayer,
    GradientSTDNormLayerV1,
    GradientSTDNormLayerV2,
    GradientSTDNormLayerV3,
    GradModiedTransformerTower,
)

logger = logging.getLogger(__name__)


def get_grad_modif_placement(args) -> BlockGradModifPlacement:
    return BlockGradModifPlacement.from_list(args.grad_modif_placement)


def get_grad_modif_fn(args) -> Optional[Callable[[], torch.nn.Module]]:
    grad_modif_type = getattr(args, "grad_modif_type", None)
    grad_modif_params = getattr(args, "grad_modif_params", [])

    if grad_modif_type is None:
        return None

    param_dict = dict()
    for param in grad_modif_params:
        t = tuple(map(str.strip, param.split("=")))
        if len(t) != 2:
            raise ValueError(f"Invalid grad_modif_params value {param}")
        param_dict[t[0]] = t[1]

    if grad_modif_type == "std_norm":
        if "c" not in param_dict:
            raise ValueError("Parameter 'c' is required for grad_modif_type 'std_norm' (add it to 'grad_modif_params')")
        if "eps" not in param_dict:
            raise ValueError(
                "Parameter 'eps' is required for grad_modif_type 'std_norm' (add it to 'grad_modif_params')"
            )
        if "layer_type" not in param_dict:
            raise ValueError(
                "Parameter 'layer_type' is required for grad_modif_type 'std_norm' (add it to 'grad_modif_params')"
            )

        layer_type = param_dict["layer_type"]
        if layer_type == "v1":
            layer = GradientSTDNormLayerV1
        elif layer_type == "v2":
            layer = GradientSTDNormLayerV2
        elif layer_type == "v3":
            layer = GradientSTDNormLayerV3
        else:
            raise ValueError(f"Unknown value of 'layer_type' {layer_type}")

        return partial(layer, c=float(param_dict["c"]), eps=float(param_dict["eps"]))
    else:
        raise ValueError(f"Unknown grad_modif_type {grad_modif_type}")


def get_attention_layer(args):
    causal = args.model_type == "gpt"
    if args.attention_mode == "vanilla":
        attention_layer_fun = lambda: llm.Attention(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            causal=causal,
            dhead=args.dhead,
            flash=args.flash_attention,
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
    elif args.attention_mode == "rope":
        attention_layer_fun = lambda: llm.AttentionRoPE(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            length=args.cutoff,
            causal=causal,
            dhead=args.dhead,
            flash=args.flash_attention,
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
    else:
        raise NotImplementedError(f"Attention type {args.attention_mode} not implemented")

    return attention_layer_fun


def get_norm_class(norm_class):
    if norm_class == "layer_norm":
        return LayerNorm
    elif norm_class == "rms_norm":
        return llm.RMSNorm
    else:
        raise NotImplementedError(f"Norm type {norm_class} not implemented")


def get_residual_layer(args):
    norm_class = get_norm_class(args.norm_class)
    if args.residual_mode == "pre_norm":
        return partial(llm.PreNormBlock, dmodel=args.dmodel, norm_class=norm_class)
    elif args.residual_mode == "parallel_pre_norm":
        return partial(llm.ParallelPreNormBlock, dmodel=args.dmodel, norm_class=norm_class)
    elif args.residual_mode == "post_norm":
        return partial(llm.PostNormBlock, dmodel=args.dmodel, norm_class=norm_class)
    elif args.residual_mode == "rezero":
        return partial(llm.RezeroBlock, dmodel=args.dmodel, norm_class=norm_class)
    else:
        raise NotImplementedError(f"Residual type {args.residual_mode} not implemented")


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale)
    elif args.ff_mode == "swi_glu":
        return_fn = lambda: llm.SwiGLUFeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")

    if args.every_other_layer:
        if args.standard_ff_first:
            return_fn = llm.EveryOtherLayer(lambda: llm.FeedForward(args.dmodel, args.dff), return_fn)
        else:
            return_fn = llm.EveryOtherLayer(
                return_fn,
                lambda: llm.FeedForward(
                    args.dmodel,
                    args.dff,
                    init_type=args.init_type,
                    init_scale=args.init_scale,
                ),
            )

    return return_fn


def get_classes_from_module_names(
    packed_names,
) -> Union[tuple[Type[torch.nn.Module]], None]:
    """
    Unpacks a comma-separated list of module names into a tuple of modules.
    """
    classes = []
    if packed_names is None:
        return None
    for name in packed_names.split(","):
        if name == "Attention":
            classes.append(llm.Attention)
        elif name == "AttentionRoPE":
            classes.append(llm.AttentionRoPE)
        elif name == "AttentionMechanism":
            classes.append(llm.AttentionMechanism)
        elif name == "RoPE":
            classes.append(llm.RoPE)
        elif name == "FeedForward":
            classes.append(llm.FeedForward)
        elif name == "Residual":
            classes.append(llm.Residual)
        elif name == "TransformerBlock":
            classes.append(llm.TransformerBlock)
        elif name == "TransformerTower":
            classes.append(llm.TransformerTower)
        elif name == "LLM":
            classes.append(llm.LLM)
        elif name == "EmbeddingLayer":
            classes.append(llm.EmbeddingLayer)
        elif name == "PredictionHead":
            classes.append(llm.PredictionHead)
        elif name == "Softmax":
            classes.append(torch.nn.Softmax)
        else:
            raise ValueError(f"Unknown name {name}")
    return tuple(classes)


def get_mixed_precision_ignored_classes(args) -> list[Type[torch.nn.Module]]:
    ignored_classes = [
        LayerNorm,
        _BatchNorm,
    ]

    selective_precision_modules = get_classes_from_module_names(args.fsdp_selective_precision_modules)
    if selective_precision_modules is not None:
        ignored_classes += list(selective_precision_modules)

    return ignored_classes


def disable_profile_schedule_fn(_: int) -> ProfilerAction:
    """
    Passing this function to the profiler as a scheduler disables profiling
    """
    return ProfilerAction.NONE


def get_model(
    max_length: int,
    vocab_size: int,
    block_modules: dict[str, Callable[[], torch.nn.Module]],
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
    grad_modif_placement: BlockGradModifPlacement,
    grad_modif_fn: Callable[[], torch.nn.Module],
    rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    include_positional_embedding: bool = True,
    checkpoint: dict[str, torch.Tensor] = None,
):

    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_components = [llm.TokenEmbedding(vocab_size, dm, init_type=init_type, init_scale=init_scale)]

    if include_positional_embedding:
        embedding_components.append(llm.PositionalEmbedding(max_length, dm, init_type=init_type, init_scale=init_scale))

    embedding_layer = llm.EmbeddingLayer(*embedding_components).to(first_gpu)

    grad_log_fn = lambda: GradCaptureLayer()

    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = GradModiedTransformerTower(
        n_blocks,
        dm,
        block_modules,
        device,
        model_fragmentation=model_fragmentation,
        gn_placement=grad_modif_placement,
        grad_modif_fn=grad_modif_fn,
        grad_log_fn=grad_log_fn,
    )

    head = llm.PredictionHead(dm, vocab_size, init_type=init_type, init_scale=init_scale).to(last_gpu)

    model = llm.LLM(embedding_layer, encoder_tower, head)

    if checkpoint is not None:
        load_model_weights(model, checkpoint)

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
