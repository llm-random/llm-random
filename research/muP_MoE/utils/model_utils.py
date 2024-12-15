from functools import partial

# import json
# from diskcache import Cache
from typing import Optional, Type, Union, Callable

import torch
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.profiler import ProfilerAction
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from lizrd.core import llm
from lizrd.text.data import LLMBatch
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights
from lizrd.core import llm
from research.muP_MoE import mup_modules
from research.muP_MoE.moe_layers.expert_types import (
    ExpertFF,
    ExpertGated,
    ExpertLinear,
)
from research.muP_MoE.moe_layers.token_choice import TokenChoiceFF
from research.muP_MoE.moe_layers.expert_choice import ExpertChoiceFF
from research.muP_MoE.moe_layers.moe_gating import (
    MoeGating,
    ExpertGating,
    TokenGating,
)


def make_loss_and_gradient_function(
    loss_checkpoint_chungs: int,
) -> Callable:
    if loss_checkpoint_chungs == 0:
        return calculate_llm_loss_and_gradient
    else:
        return partial(chungized_llm_loss_and_gradient, n_chungs=loss_checkpoint_chungs)


def calculate_single_chung_loss(
    model: torch.nn.Module,
    mixed_precision_dtype: torch.dtype,
    encoder_output: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
):
    output = model(encoder_output)
    with torch.autocast(device_type="cuda", enabled=False, dtype=mixed_precision_dtype):
        gt = gt.to(output.device)
        loss = F.cross_entropy(
            output.flatten(0, -2),
            gt.reshape(-1).long(),
            reduction="none",
        )

        correct_tokens = gt.long() == output.argmax(dim=-1)
        correct_tokens = correct_tokens.long().reshape(-1) * mask.reshape(-1)
        correct_tokens = correct_tokens.sum()

        total_tokens = mask.sum()

    return loss[mask.reshape(-1) == 1], correct_tokens, total_tokens


def run_backward(
    loss: torch.Tensor,
    mixed_precision_dtype: torch.dtype,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    with torch.autocast(device_type="cuda", enabled=False, dtype=mixed_precision_dtype):
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()


def chungized_llm_loss_and_gradient(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    n_chungs: int,
    mixed_precision_dtype: torch.dtype,
    num_checkpoint_accumulation_steps: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[float, dict]:
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=mixed_precision_dtype
    ):
        embeddings = model.embedding_layer(input_tokens)
        encoder_output = model.encoder(embeddings)
        encoder_output_detach = encoder_output.detach()
        encoder_output_detach.requires_grad = True
        chunged_encoder_outputs = torch.chunk(encoder_output_detach, n_chungs, dim=0)
        chunged_non_masked_inputs = torch.chunk(gt_tokens, n_chungs, dim=0)
        chunged_non_masked_masks = torch.chunk(mask, n_chungs, dim=0)

        total_loss = 0
        total_correct_tokens = 0
        total_masked_tokens = 0
        for chunged_encoder_output, chunged_gt, chunged_mask in zip(
            chunged_encoder_outputs, chunged_non_masked_inputs, chunged_non_masked_masks
        ):
            (
                single_chung_loss,
                single_chung_correct_tokens,
                single_chung_masked_tokens,
            ) = calculate_single_chung_loss(
                model.head,
                mixed_precision_dtype,
                chunged_encoder_output,
                chunged_gt,
                chunged_mask,
            )
            partial_loss = (
                single_chung_loss.mean() / n_chungs / num_checkpoint_accumulation_steps
            )
            if model.training:
                run_backward(partial_loss, mixed_precision_dtype, scaler)
            total_loss += partial_loss.item()
            total_correct_tokens += single_chung_correct_tokens
            total_masked_tokens += single_chung_masked_tokens

        aux_info = {
            "correct_tokens": total_correct_tokens,
            "total_masked_tokens": total_masked_tokens,
            "losses": retrieve_additional_losses(model),
        }

    for key, value in aux_info["losses"].items():
        aux_info["losses"][key] = value / num_checkpoint_accumulation_steps
    if model.training:
        # ok, we need to backward one loss (because of torch autograd)
        # the "loss" that has the same gradient as the original cross entropy loss is the sum below
        assert encoder_output_detach.grad.shape == encoder_output.shape
        loss_to_optimize = (encoder_output * encoder_output_detach.grad).sum()
        for value in aux_info["losses"].values():
            loss_to_optimize += value if scaler is None else scaler.scale(value)
        loss_to_optimize.backward()
    clear_additional_losses(model)
    return total_loss, aux_info


def calculate_llm_loss_and_gradient(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    mixed_precision_dtype: torch.dtype,
    num_checkpoint_accumulation_steps: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[float, dict]:
    def hack_for_python_garbage_collection():
        """we want to have no reference to model output while backpropagating to allow torch to free memory,
        so we wrap loss calculation in a function"""
        input_tokens = batch.input_ids
        gt_tokens = batch.target_ids
        mask = batch.should_calculate_loss

        with torch.autocast(
            device_type="cuda", enabled=mixed_precision, dtype=mixed_precision_dtype
        ):
            model_output = model(input_tokens)

        # move the gt tokens and mask to the same device as the model output - they should be on the same device for loss calculation
        gt_tokens = gt_tokens.to(model_output.device)
        mask = mask.to(model_output.device)

        mask_loss = F.cross_entropy(
            model_output.flatten(0, -2),
            gt_tokens.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[mask.reshape(-1) == 1]
        loss = mask_loss.mean() / num_checkpoint_accumulation_steps

        correct_tokens = gt_tokens.long() == model_output.argmax(dim=-1)
        correct_tokens = correct_tokens.long().reshape(-1) * mask.reshape(-1)
        correct_tokens = correct_tokens.sum()
        total_masked_tokens = mask.sum()

        aux_info = {
            "correct_tokens": correct_tokens,
            "total_masked_tokens": total_masked_tokens,
            "losses": retrieve_additional_losses(model),
        }
        return loss, aux_info

    loss, aux_info = hack_for_python_garbage_collection()
    for key, value in aux_info["losses"].items():
        aux_info["losses"][key] = value / num_checkpoint_accumulation_steps
    if model.training:
        loss_to_optimize = loss.clone()
        for value in aux_info["losses"].values():
            loss_to_optimize += value
        run_backward(loss_to_optimize, mixed_precision_dtype, scaler)

    clear_additional_losses(model)
    return loss.item(), aux_info


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
            attn_scale=args.attention_normalization,
        )
    elif args.attention_mode == "rope":
        attention_layer_fun = lambda: llm.Attention(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            rope=True,
            seq_len=args.cutoff,
            causal=causal,
            dhead=args.dhead,
            flash=args.flash_attention,
            init_type=args.init_type,
            init_scale=args.init_scale,
            attn_scale=args.attention_normalization,
        )
    else:
        raise NotImplementedError(
            f"Attention type {args.attention_mode} not implemented"
        )

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
        return partial(
            llm.ParallelPreNormBlock, dmodel=args.dmodel, norm_class=norm_class
        )
    elif args.residual_mode == "post_norm":
        return partial(llm.PostNormBlock, dmodel=args.dmodel, norm_class=norm_class)
    elif args.residual_mode == "rezero":
        return partial(llm.RezeroBlock, dmodel=args.dmodel, norm_class=norm_class)
    else:
        raise NotImplementedError(f"Residual type {args.residual_mode} not implemented")


def retrieve_additional_losses(model: torch.nn.Module):
    losses = {}
    if not hasattr(model, "forward_pass_cache"):
        return losses

    if "load_balancing_losses" in model.forward_pass_cache:
        load_balancing_losses = model.forward_pass_cache.get(
            "load_balancing_losses", []
        )
        load_balancing_losses = torch.stack(load_balancing_losses)
        load_balancing_loss = torch.mean(load_balancing_losses)
        losses["load_balancing_loss"] = load_balancing_loss

    return losses


def clear_additional_losses(model: torch.nn.Module):
    if not hasattr(model, "forward_pass_cache"):
        return

    if "load_balancing_losses" in model.forward_pass_cache:
        model.forward_pass_cache.pop("load_balancing_losses", None)


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(
            args.dmodel,
            args.dff,
            init_type=args.init_type,
            init_scale=args.init_scale,
            bias="none",
        )
    elif args.ff_mode == "swi_glu":
        return_fn = lambda: llm.SwiGLUFeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    elif args.ff_mode == "expert_choice":
        args = determine_moe_args(args)
        ff_args, make_expert_inner_function = get_expert_choice_args(args)
        return_fn = lambda: ExpertChoiceFF(
            **ff_args,
            expert_inner_function=make_expert_inner_function(),
            zloss_weight=args.zloss_weight,
        )
    elif args.ff_mode == "token_choice":
        args = determine_moe_args(args)
        make_expert_inner_function = get_inner_expert(args)
        use_topk_initialization = get_expert_init(
            args.expert_use_topk_initialization, default=False
        )
        make_expert_inner_function = partial(
            make_expert_inner_function, use_topk_initialization=use_topk_initialization
        )
        return_fn = lambda: TokenChoiceFF(
            dmodel=args.dmodel,
            n_experts=args.n_experts,
            capacity_factor=args.capacity_factor,
            expert_inner_function=make_expert_inner_function(),
            load_balancing_loss_weight=args.load_balancing_loss_weight,
            zloss_weight=args.zloss_weight,
            routing_top_k=args.routing_top_k,
            init_scale=args.init_scale,
            init_type=args.init_type,
            **get_weightless_args(args),
        )
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")
    return return_fn


def determine_moe_args(args):
    set_arguments_option1 = all(
        [args.total_experts_width, args.effective_dff, args.n_experts]
    ) and not any([args.expert_size, args.topk_fraction])
    set_arguments_option2 = all(
        [args.expert_size, args.topk_fraction, args.n_experts]
    ) and not any([args.effective_dff, args.total_experts_width])
    set_arguments_option3 = all(
        [args.granularity, args.expansion_rate, args.effective_dff_x]
    )
    set_arguments_option4 = all([args.n_experts, args.expert_size, args.routing_top_k])

    if 1 != sum(  # exactly one of the options must be set
        [
            set_arguments_option1,
            set_arguments_option2,
            set_arguments_option3,
            set_arguments_option4,
        ]
    ):
        raise AssertionError(
            "You must specify either total_experts_width, effective_dff, and n_experts "
            "or expert_size, topk_fraction, and n_experts "
            "or granularity, expansion_rate, and effective_dff_x "
            "or n_experts, expert_size, and routing_top_k."
        )
    # 4 is the standard dff_x, we assume it's defined relative to that
    dff = args.dmodel * 4

    if set_arguments_option4:
        args.total_experts_width = args.n_experts * args.expert_size
        args.expansion_rate = args.total_experts_width / dff
        args.effective_dff = args.expert_size * args.routing_top_k
    if set_arguments_option3:
        args.total_experts_width = (
            args.dmodel * args.effective_dff_x * args.expansion_rate
        )
        args.n_experts = args.expansion_rate * args.granularity
        args.effective_dff = args.effective_dff_x * args.dmodel

    if set_arguments_option2:
        args.routing_top_k = args.topk_fraction * args.n_experts
        args.effective_dff = args.routing_top_k * args.expert_size
        args.total_experts_width = args.expert_size * args.n_experts
    else:
        expert_size = args.total_experts_width / args.n_experts
        assert expert_size == int(expert_size)
        args.expert_size = int(expert_size)

        args.routing_top_k = args.effective_dff / expert_size
        args.topk_fraction = args.routing_top_k / args.n_experts
        assert 0.0 <= args.topk_fraction <= 1.0

    assert args.routing_top_k == int(args.routing_top_k)
    args.routing_top_k = int(args.routing_top_k)

    # in the end, these arguments should be set
    assert all(
        [
            args.routing_top_k,
            args.total_experts_width,
            args.n_experts,
            args.expert_size,
            args.topk_fraction,
        ]
    )
    return args


def get_expert_choice_args(args):
    use_topk_initialization = get_expert_init(
        args.expert_use_topk_initialization, default=True
    )
    expert_inner_function = partial(
        get_inner_expert(args), use_topk_initialization=use_topk_initialization
    )
    args = dict(
        **get_expert_choice_args_old(args),
        **get_weightless_args(args),
    )
    del args["use_full_einsum"]  # this is no longer compatible
    del args["expert_size"]
    return args, expert_inner_function


def get_expert_choice_args_old(args):
    return {
        "dmodel": args.dmodel,
        "n_experts": args.n_experts,
        "expert_size": args.expert_size,
        "topk_fraction": args.topk_fraction,
        "random_perm": args.expert_random_perm,
        "group_by_batch": args.group_granular_moe_by_batch,
        "softmax_ungrouped": args.softmax_ungrouped,
        "one_hot_impl": args.granular_moe_one_hot_impl,
        "softmax_over": args.softmax_over,
        "use_full_einsum": args.use_full_einsum,
        "group_size": args.simulate_group_size,
        "init_type": args.init_type,
        "init_scale": args.init_scale,
        "use_torch_bmm": args.use_torch_bmm,
        "use_layer_norm": args.layer_norm_in_expert_choice,
    }


def get_weightless_args(args):
    try:
        moe_values_exp = float(args.moe_values_exp)
    except ValueError:
        moe_values_exp = None

    return dict(
        get_router_values_from=args.get_router_values_from,
        moe_values_exp=moe_values_exp,
        detach_gate=args.moe_detach_gate,
    )


def get_expert_init(parameter, default=False):
    if parameter == "Always":
        return True
    elif parameter == "Never":
        return False
    elif parameter == "Default":
        return default
    else:
        raise ValueError(f"Unknown expert init type {parameter}")


def get_inner_expert(args):
    if args.moe_inner_expert == "ff":
        expert_inner_class = partial(ExpertFF, activation_name=args.activation_type)
    elif args.moe_inner_expert == "ff_gated":
        expert_inner_class = partial(ExpertGated, activation_name=args.activation_type)
    elif args.moe_inner_expert == "linear":
        expert_inner_class = ExpertLinear
    # these experts names are left for backward compatibility
    elif args.moe_inner_expert == "relu":
        expert_inner_class = partial(ExpertFF, activation_name="relu")
    elif args.moe_inner_expert == "swi_glu":
        expert_inner_class = partial(ExpertGated, activation_name="silu")
    elif args.moe_inner_expert == "geglu":
        expert_inner_class = partial(ExpertGated, activation_name="gelu")
    else:
        raise NotImplementedError(
            f'Inner expert type "{args.moe_inner_expert}" not implemented'
        )
    return partial(
        expert_inner_class,
        dmodel=args.dmodel,
        n_experts=args.n_experts,
        expert_size=args.expert_size,
        init_scale=args.init_scale,
        init_type=args.init_type,
        topk=args.routing_top_k,
    )


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
        elif name == "ExpertChoiceFF":
            classes.append(ExpertChoiceFF)
        elif name == "ExpertGating":
            classes.append(ExpertGating)
        elif name == "TokenGating":
            classes.append(TokenGating)
        elif name == "MoeGating":
            classes.append(MoeGating)
        else:
            raise ValueError(f"Unknown name {name}")
    return tuple(classes)


def get_mixed_precision_ignored_classes(args) -> list[Type[torch.nn.Module]]:
    ignored_classes = [
        LayerNorm,
        _BatchNorm,
    ]

    selective_precision_modules = get_classes_from_module_names(
        args.fsdp_selective_precision_modules
    )
    if selective_precision_modules is not None:
        ignored_classes += list(selective_precision_modules)

    return ignored_classes


def update_model_fit_gpu_info(database: str, params: dict, value: str):
    """
    This function is used to records whether a model with given params fits in gpu.
    """
    # if database is not None and params is not None:
    #     with Cache(database) as cache:
    #         serialized_params = json.dumps(params, sort_keys=True)
    #         cache[serialized_params] = value
    print(database, params)


def get_model_fit_gpu_info(database: str, params: dict):
    """
    This function is used to records whether a model with given params fits in gpu.
    """
    # if database is not None and params is not None:
    #     with Cache(database) as cache:
    #         serialized_params = json.dumps(params, sort_keys=True)
    #         return cache[serialized_params]
    print(database, params)


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
    rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    include_positional_embedding: bool = True,
    checkpoint: dict[str, torch.Tensor] = None,
    mup_config: dict = None,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_components = [
        llm.TokenEmbedding(vocab_size, dm, init_type=init_type, init_scale=init_scale)
    ]

    if include_positional_embedding:
        embedding_components.append(
            llm.PositionalEmbedding(
                max_length, dm, init_type=init_type, init_scale=init_scale
            )
        )

    embedding_layer = llm.EmbeddingLayer(*embedding_components).to(first_gpu)

    # Python officially preserves dict order since 3.7, so we pass the layer dict
    transformer_tower = llm.TransformerTower(
        n_blocks,
        dm,
        block_modules,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn,
    )

    head = llm.PredictionHead(
        dm, vocab_size, init_type=init_type, init_scale=init_scale
    ).to(last_gpu)

    model = mup_modules.muP_LLM(embedding_layer, transformer_tower, head, mup_config)

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
