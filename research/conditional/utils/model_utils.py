from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lizrd.core import llm
from lizrd.text.data import LLMBatch
from lizrd.core.llm import Parallel
from research.conditional.moe_layers.cont_moe_designs.common_weighted_parameter_matrices import (
    ContinuousMoECommonWeightedParameters,
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
)
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.moe_layers.token_choice import TokenChoiceFF
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
):
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

    def make_custom_forward():
        def custom_forward(*inputs):
            output = model.head(inputs[0])
            with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float16):
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
        device_type="cuda", enabled=mixed_precision, dtype=torch.float16
    ):
        encoder_output = model.encoder(input_tokens)
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
):
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=torch.float16
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
    if args.model_type == "gpt":
        attention_layer_fun = lambda: llm.CausalAttention(
            args.dmodel, args.n_att_heads, args.dhead
        )
    elif args.model_type == "bert":
        attention_layer_fun = lambda: llm.Attention(args.dmodel, args.n_att_heads)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    return attention_layer_fun


def get_residual_layer(args):
    if args.residual_mode == "pre_norm":
        return partial(llm.PreNormBlock, dmodel=args.dmodel)
    elif args.residual_mode == "post_norm":
        return partial(llm.PostNormBlock, dmodel=args.dmodel)
    elif args.residual_mode == "rezero":
        return partial(llm.RezeroBlock, dmodel=args.dmodel)
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

    if not set_arguments_option1 and not set_arguments_option2:
        raise AssertionError(
            "You must specify either total_experts_width, effective_dff, and n_experts or expert_size, topk_fraction, and n_experts"
        )

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


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(args.dmodel, args.dff)
    elif args.ff_mode == "vanilla_timed":
        return_fn = lambda: FeedForwardTimed(
            args.dmodel, args.dff, args.activation_type, args.no_ff
        )
    elif args.ff_mode == "cont_moe" or args.ff_mode == "cont_moe_quick":
        return_fn = lambda: ContinuousMoE(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_merge_diff_simple":
        return_fn = lambda: ContinuousMoEMergeDifferentlySimple(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_merge_diff_comm_base":
        return_fn = lambda: ContinuousMoEMergeDifferentlyCommonBase(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_rawmerge":
        return_fn = lambda: ContinuousMoERawmerge(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_topmerge":
        return_fn = lambda: ContinuousMoETopmerge(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_nosoft":
        return_fn = lambda: ContinuousMoENosoftmax(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_adatemp":
        return_fn = lambda: ContinuousMoEAdaTemp(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
            separate_temp_for_experts=args.separate_temp_for_experts,
            separate_temp_for_emit_merge=args.separate_temp_for_emit_merge,
        )
    elif args.ff_mode == "cont_moe_ln":
        return_fn = lambda: ContinuousMoELayernorm(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_final":
        return_fn = lambda: ContinuousMoEFinal(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_random_groups":
        return_fn = lambda: ContinuousMoERandomGroups(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
            batch_size=args.batch_size,
            seqlen=args.cutoff,
            mix_whole_batch=args.mix_whole_batch,
        )
    elif args.ff_mode == "cont_moe_common_weighted_parameters":
        return_fn = lambda: ContinuousMoECommonWeightedParameters(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_separate_weighted_parameters":
        return_fn = lambda: ContinuousMoESeparateWeightedParameters(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
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
        return_fn = lambda: TokenChoiceFF(
            dmodel=args.dmodel,
            n_experts=args.n_experts,
            expert_size=args.effective_dff,
            capacity_factor=args.capacity_factor,
            load_balancing_loss_weight=args.load_balancing_loss_weight,
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
                return_fn, lambda: llm.FeedForward(args.dmodel, args.dff)
            )

    return return_fn
