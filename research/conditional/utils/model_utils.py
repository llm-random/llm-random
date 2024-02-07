from functools import partial

# import json
# from diskcache import Cache
from typing import Type, Union
import torch
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.profiler import ProfilerAction

from lizrd.core import llm, nn
from lizrd.text.data import LLMBatch
from lizrd.core.llm import Parallel
from research.conditional.moe_layers.cont_moe_designs.common_weighted_parameter_matrices import (
    ContinuousMoECommonWeightedParameters,
)
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature_positive import (
    ContinuousMoEAdaTempPositive,
)
from research.conditional.moe_layers.cont_moe_designs.random_grouping import (
    ContinuousMoERandomGroups,
)
from research.conditional.moe_layers.cont_moe_designs.learn_temp_and_common_base import (
    ContinuousMoEFinal,
)
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature import (
    ContinuousMoEAdaTemp,
)
from research.conditional.moe_layers.cont_moe_designs.add_layernorms import (
    ContinuousMoELayernorm,
)
from research.conditional.moe_layers.cont_moe_designs.no_softmax_on_weights import (
    ContinuousMoENosoftmax,
)
from research.conditional.moe_layers.cont_moe_designs.send_result_only_to_top1_token import (
    ContinuousMoETopmerge,
)
from research.conditional.moe_layers.cont_moe_designs.merge_without_weights import (
    ContinuousMoERawmerge,
)
from research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights_common_base import (
    ContinuousMoEMergeDifferentlyCommonBase,
)
from research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights import (
    ContinuousMoEMergeDifferentlySimple,
)
from research.conditional.moe_layers.cont_moe_designs.separate_weighted_parameter_matrices import (
    ContinuousMoESeparateWeightedParameters,
)
from research.conditional.moe_layers.continuous_moe import (
    ContinuousMoE,
    LegacyContinuousMoE,
)
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF, ExpertGating
from research.conditional.moe_layers.token_choice import (
    TokenChoiceFF,
    TokenChoiceRouter,
    ExpertRelu,
    ExpertSwiGLU,
)
from research.conditional.moe_layers._token_choice_deprecated import (
    TokenChoiceFF as TokenChoiceFFDeprecated,
)
from research.mamba.moe_in_mamba import MambaInProj
from research.conditional.moe_layers.ff_timed import FeedForwardTimed


def make_loss_function(loss_checkpoint_chungs: int):
    if loss_checkpoint_chungs == 0:
        return calculate_llm_loss
    else:
        return partial(chungized_llm_loss, n_chungs=loss_checkpoint_chungs)


def chungized_llm_loss(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    vocab_size: int,
    n_chungs: int,
    mixed_precision_dtype: torch.dtype,
):
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

    def make_custom_forward():
        def custom_forward(*inputs):
            x, gt, mask = inputs
            output = model.head(x)
            with torch.autocast(
                device_type="cuda", enabled=False, dtype=mixed_precision_dtype
            ):
                gt = inputs[1]
                mask = inputs[2]
                gt = gt.to(output.device)
                loss = F.cross_entropy(
                    output.reshape(-1, vocab_size),
                    gt.reshape(-1).long(),
                    reduction="none",
                )

                correct_tokens = gt.long() == output.argmax(dim=-1)
                correct_tokens = correct_tokens.long().reshape(-1) * mask.reshape(-1)
                correct_tokens = correct_tokens.sum()

                total_tokens = mask.sum()

            return loss[mask.reshape(-1) == 1], correct_tokens, total_tokens

        return custom_forward

    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=mixed_precision_dtype
    ):
        embeddings = model.embedding_layer(input_tokens)
        encoder_output = model.encoder(embeddings)
        chunged_inputs = torch.chunk(encoder_output, n_chungs, dim=0)
        chunged_non_masked_inputs = torch.chunk(gt_tokens, n_chungs, dim=0)
        chunged_non_masked_masks = torch.chunk(mask, n_chungs, dim=0)

        num_tokens = 0
        total_loss = 0
        total_correct_tokens = 0
        total_masked_tokens = 0
        for chunged_input, chunged_gt, chunged_mask in zip(
            chunged_inputs, chunged_non_masked_inputs, chunged_non_masked_masks
        ):
            (
                partial_loss_output,
                partial_correct_tokens,
                partial_masked_tokens,
            ) = checkpoint(
                make_custom_forward(), chunged_input, chunged_gt, chunged_mask
            )
            num_tokens += partial_loss_output.shape[0]
            total_loss += partial_loss_output.sum()
            total_correct_tokens += partial_correct_tokens
            total_masked_tokens += partial_masked_tokens

        aux_info = {
            "correct_tokens": total_correct_tokens,
            "total_masked_tokens": total_masked_tokens,
            "losses": retrieve_additional_losses(model),
        }

        return total_loss / num_tokens, aux_info


def calculate_llm_loss(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    vocab_size: int,
    mixed_precision_dtype: torch.dtype,
):
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
        model_output.reshape(-1, vocab_size),
        gt_tokens.reshape(-1).long(),
        reduction="none",
    )
    mask_loss = mask_loss[mask.reshape(-1) == 1]
    loss = mask_loss.mean()

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
        raise NotImplementedError(
            f"Attention type {args.attention_mode} not implemented"
        )

    return attention_layer_fun


def get_residual_layer(args):
    if args.norm_class == "layer_norm":
        norm_class = LayerNorm
    elif args.norm_class == "rms_norm":
        norm_class = llm.RMSNorm
    else:
        raise NotImplementedError(f"Norm type {args.norm_class} not implemented")
    if args.residual_mode == "pre_norm":
        return partial(llm.PreNormBlock, dmodel=args.dmodel, norm_class=norm_class)
    elif args.residual_mode == "post_norm":
        return partial(llm.PostNormBlock, dmodel=args.dmodel, norm_class=norm_class)
    elif args.residual_mode == "rezero":
        return partial(llm.RezeroBlock, dmodel=args.dmodel, norm_class=norm_class)
    else:
        raise NotImplementedError(f"Residual type {args.residual_mode} not implemented")


def get_expert_choice_args(args):
    set_arguments_option1 = (
        args.total_experts_width is not None
        and args.effective_dff is not None
        and args.n_experts is not None
    ) and (args.expert_size is None and args.topk_fraction is None)
    set_arguments_option2 = (
        args.expert_size is not None
        and args.topk_fraction is not None
        and args.n_experts is not None
    ) and (args.effective_dff is None and args.total_experts_width is None)
    set_arguments_option3 = (  # this should be the default
        args.granularity is not None
        and args.expansion_rate is not None
        and args.effective_dff_x is not None
    )

    if (
        not set_arguments_option1
        and not set_arguments_option2
        and not set_arguments_option3
    ):
        raise AssertionError(
            "You must specify either total_experts_width, effective_dff, and n_experts "
            "or expert_size, topk_fraction, and n_experts "
            "or granularity, expansion_rate, and effective_dff_x "
        )

    if set_arguments_option3:
        # 4 is the standard dff_x, we assume it's defined relative to that
        dff_x = 4
        args.total_experts_width = args.dmodel * dff_x * args.expansion_rate
        args.n_experts = args.expansion_rate * args.granularity
        args.effective_dff = args.effective_dff_x * args.dmodel

    if args.total_experts_width is not None:
        expert_size = args.total_experts_width / args.n_experts
        assert expert_size == int(expert_size)
        args.expert_size = int(expert_size)

        experts_per_token = args.effective_dff / expert_size

        topk_fraction = experts_per_token / args.n_experts
        assert 0.0 <= topk_fraction <= 1.0
        args.topk_fraction = topk_fraction
    else:
        experts_per_token = args.topk_fraction * args.n_experts
        args.effective_dff = experts_per_token * args.expert_size
        args.total_experts_width = args.expert_size * args.n_experts

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


def get_expert_choice_with_parallel_ff_args(args):
    expert_choice_params = get_expert_choice_args(args)
    n_experts = expert_choice_params["n_experts"]
    expert_size = expert_choice_params["expert_size"]
    top_k_fraction = expert_choice_params["topk_fraction"]

    def calculate_effective_expert_dff(_expert_size, _n_experts, _topk_fraction):
        return _topk_fraction * _n_experts * _expert_size

    if args.ff_parallel_mode == "modify_expert_size":
        expert_size = int(
            expert_choice_params["expert_size"]
            * (1 - args.ff_parallel_compute_fraction)
        )
        expert_choice_params["expert_size"] = expert_size

    elif args.ff_parallel_mode == "modify_topk_fraction":
        top_k_fraction = expert_choice_params["topk_fraction"] * (
            1 - args.ff_parallel_compute_fraction
        )

        expert_choice_params["topk_fraction"] = top_k_fraction

    elif args.ff_parallel_mode == "modify_n_experts":
        n_experts = int(
            expert_choice_params["n_experts"] * (1 - args.ff_parallel_compute_fraction)
        )
        expert_choice_params["n_experts"] = n_experts
    else:
        raise ValueError(
            f"Invalid ff_parallel_mode {args.ff_parallel_mode}. Possible values are modify_expert_size, modify_topk_fraction, modify_n_experts"
        )

    dff_expert = int(
        calculate_effective_expert_dff(expert_size, n_experts, top_k_fraction)
    )
    dff_parallel = args.effective_dff - dff_expert
    return {
        "expert_choice_kwargs": expert_choice_params,
        "parallel_ff_args": (args.dmodel, dff_parallel),
    }


def retrieve_additional_losses(model: torch.nn.Module):
    losses = {}
    if not hasattr(model, "forward_pass_cache"):
        return losses

    if "load_balancing_losses" in model.forward_pass_cache:
        load_balancing_losses = model.forward_pass_cache["load_balancing_losses"]
        load_balancing_losses = torch.stack(load_balancing_losses)
        load_balancing_loss = torch.mean(load_balancing_losses)
        losses["load_balancing_loss"] = load_balancing_loss

    return losses


def get_common_mot_kwargs(args):
    return {
        "dm": args.dmodel,
        "dff": args.dff,
        "n_experts": args.n_experts,
        "group_size": args.group_size,
        "sparsity_dim": args.sparsity_dim,
        "temperature": args.temperature,
        "expert_size": args.expert_size,
        "use_opt_einsum": args.use_opt_einsum,
        "flop_matched": args.flop_matched,
        "init_type": args.init_type,
        "init_scale": args.init_scale,
        "emit_softmax_over_experts": args.emit_softmax_over_experts,
    }


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    elif args.ff_mode == "swi_glu":
        return_fn = lambda: llm.SwiGLUFeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    elif args.ff_mode == "vanilla_timed":
        return_fn = lambda: FeedForwardTimed(
            args.dmodel, args.dff, args.activation_type, args.no_ff
        )
    elif args.ff_mode == "cont_moe" or args.ff_mode == "cont_moe_quick":
        return_fn = lambda: ContinuousMoE(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_merge_diff_simple":
        return_fn = lambda: ContinuousMoEMergeDifferentlySimple(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_merge_diff_comm_base":
        return_fn = lambda: ContinuousMoEMergeDifferentlyCommonBase(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_rawmerge":
        return_fn = lambda: ContinuousMoERawmerge(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_topmerge":
        return_fn = lambda: ContinuousMoETopmerge(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_nosoft":
        return_fn = lambda: ContinuousMoENosoftmax(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_adatemp":
        return_fn = lambda: ContinuousMoEAdaTemp(
            **get_common_mot_kwargs(args),
            share_by_experts=args.share_by_experts,
            share_by_emit_merge=args.share_by_emit_merge,
        )
    elif args.ff_mode == "cont_moe_adatemp_positive":
        return_fn = lambda: ContinuousMoEAdaTempPositive(
            **get_common_mot_kwargs(args),
            share_by_experts=args.share_by_experts,
            share_by_emit_merge=args.share_by_emit_merge,
        )
    elif args.ff_mode == "cont_moe_ln":
        return_fn = lambda: ContinuousMoELayernorm(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_final":
        return_fn = lambda: ContinuousMoEFinal(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_random_groups":
        return_fn = lambda: ContinuousMoERandomGroups(
            **get_common_mot_kwargs(args),
            batch_size=args.batch_size,
            seqlen=args.cutoff,
            mix_whole_batch=args.mix_whole_batch,
        )
    elif args.ff_mode == "cont_moe_common_weighted_parameters":
        return_fn = lambda: ContinuousMoECommonWeightedParameters(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_separate_weighted_parameters":
        return_fn = lambda: ContinuousMoESeparateWeightedParameters(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_legacy":
        return_fn = lambda: LegacyContinuousMoE(**get_common_mot_kwargs(args))
    elif args.ff_mode == "expert_choice":
        ff_args = get_expert_choice_args(args)
        return_fn = partial(ExpertChoiceFF, **ff_args)
    elif args.ff_mode == "expert_choice_with_parallel_ff":
        expert_choice_kwargs = get_expert_choice_with_parallel_ff_args(args)[
            "expert_choice_kwargs"
        ]
        parallel_ff_args = get_expert_choice_with_parallel_ff_args(args)[
            "parallel_ff_args"
        ]
        return_fn = lambda: Parallel(
            ExpertChoiceFF(**expert_choice_kwargs),
            llm.FeedForward(*parallel_ff_args),
        )
    elif args.ff_mode == "token_choice":
        if args.token_choice_inner == "relu":
            expert_inner_class = ExpertRelu
        elif args.token_choice_inner == "swi_glu":
            expert_inner_class = ExpertSwiGLU

        else:
            raise NotImplementedError(
                f"Token choice logic {args.token_choice_inner} not implemented"
            )
        make_expert_inner_function = partial(
            expert_inner_class,
            dmodel=args.dmodel,
            n_experts=args.n_experts,
            expert_size=args.expert_size,
            init_scale=args.init_scale,
            init_type=args.init_type,
        )
        return_fn = lambda: TokenChoiceFF(
            dmodel=args.dmodel,
            n_experts=args.n_experts,
            capacity_factor=args.capacity_factor,
            expert_inner_function=make_expert_inner_function(),
            load_balancing_loss_weight=args.load_balancing_loss_weight,
            routing_top_k=args.routing_top_k,
            init_scale=args.init_scale,
            init_type=args.init_type,
            vectorize=(not args.dont_vectorize_switch),
        )
    elif args.ff_mode == "token_choice_deprecated":
        return_fn = lambda: TokenChoiceFFDeprecated(
            dmodel=args.dmodel,
            n_experts=args.n_experts,
            expert_size=args.expert_size,
            capacity_factor=args.capacity_factor,
            load_balancing_loss_weight=args.load_balancing_loss_weight,
            init_scale=args.init_scale,
            init_type=args.init_type,
        )
    elif args.ff_mode == "kernelized_fc":
        from research.conditional.moe_layers.kernelized import FCKernelized

        return_fn = lambda: FCKernelized(
            dmodel=args.dmodel,
            dff=args.dff,
            kernel_r=args.kernel_r,
            kernel_type=args.kernel_type,
            redraw_projections_interval=args.redraw_projections_interval,
            no_kernel_norm=args.no_kernel_norm,
            no_average_attn=args.no_average_attn,
            nystrom=args.nystrom,
            xfavor=args.xfavor,
        )
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")

    if args.every_other_layer:
        if args.standard_ff_first:
            return_fn = llm.EveryOtherLayer(
                lambda: llm.FeedForward(args.dmodel, args.dff), return_fn
            )
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


def get_vanilla_mamba_layer(args):
    import mamba_ssm

    return lambda: mamba_ssm.Mamba(
        d_model=args.dmodel, expand=args.mamba_expansion, use_fast_path=False
    )


def get_mamba_layer(args):
    import mamba_ssm

    if args.mamba_mode == "vanilla":
        return_fn = lambda: mamba_ssm.Mamba(
            d_model=args.dmodel, expand=args.mamba_expansion, use_fast_path=False
        )
    elif args.mamba_mode in [
        "out_proj_moe",
        "conv_proj_moe",
        "gate_proj_moe",
        "conv_gate_proj_moe",
        "conv_out_proj_moe",
        "gate_out_proj_moe",
        "conv_gate_out_proj_moe",
    ]:

        def get_token_choice_ff(d_input, d_output):
            return TokenChoiceFF(
                dmodel=d_input,
                doutput=d_output,
                n_experts=args.n_experts,
                expert_size=args.expert_size,
                capacity_factor=args.capacity_factor,
                load_balancing_loss_weight=args.load_balancing_loss_weight,
                routing_top_k=args.routing_top_k,
                init_scale=args.init_scale,
                init_type=args.init_type,
            )

        def modified_out_mamba():
            mamba = mamba_ssm.Mamba(
                d_model=args.dmodel, expand=args.mamba_expansion, use_fast_path=False
            )
            if "out" in args.mamba_mode:
                mamba.out_proj = get_token_choice_ff(
                    d_input=mamba.d_inner, d_output=mamba.d_model
                )
            conv_proj = (
                get_token_choice_ff(d_input=mamba.d_model, d_output=mamba.d_inner)
                if "conv" in args.mamba_mode
                else nn.Linear(mamba.d_model, mamba.d_inner, bias=False)
            )
            gate_proj = (
                get_token_choice_ff(d_input=mamba.d_model, d_output=mamba.d_inner)
                if "gate" in args.mamba_mode
                else nn.Linear(mamba.d_model, mamba.d_inner, bias=False)
            )
            mamba.in_proj = MambaInProj(
                batch_size=args.batch_size,
                conv_proj=conv_proj,
                gate_proj=gate_proj,
                dtype=mamba.in_proj.weight.dtype,
            )

            return mamba

        return_fn = modified_out_mamba
    else:
        raise NotImplementedError(f"Mamba mode {args.mamba_mode} not implemented")

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
        elif name == "ExpertChoiceFF":
            classes.append(ExpertChoiceFF)
        elif name == "ExpertGating":
            classes.append(ExpertGating)
        elif name == "Softmax":
            classes.append(torch.nn.Softmax)
        elif name == "TokenChoiceRouter":
            classes.append(TokenChoiceRouter)
        elif name == "Mamba":
            import mamba_ssm

            classes.append(mamba_ssm.Mamba)
        else:
            raise ValueError(f"Unknown name {name}")
    return tuple(classes)


def get_mixed_precision_ignored_classes(args) -> list[Type[torch.nn.Module]]:
    ignored_classes = [
        ExpertGating,
        LayerNorm,
        _BatchNorm,
        TokenChoiceRouter,
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
