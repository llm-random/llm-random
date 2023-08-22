from functools import partial
from typing import Literal
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lizrd.core import llm
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
from research.conditional.moe_layers.ff_timed import FeedForwardTimed


def make_loss_function(model: Literal["bert", "gpt"], loss_checkpoint_chungs: int):
    if model == "bert":
        if loss_checkpoint_chungs == 0:
            return partial(calculate_bert_loss)
        else:
            return partial(
                chungized_bert_loss,
                n_chungs=loss_checkpoint_chungs,
            )
    elif model == "gpt":
        if loss_checkpoint_chungs == 0:
            return calculate_gpt_loss
        else:
            return partial(chungized_gpt_loss, n_chungs=loss_checkpoint_chungs)
    else:
        raise ValueError(f"Model type {model} not implemented")


def chungized_llm_loss(
    input_tokens,
    gt_tokens,
    mask,
    model,
    mixed_precision,
    vocab_size,
    n_chungs,
):
    def make_custom_forward():
        def custom_forward(*inputs):
            output = model.head(inputs[0])
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

    return total_loss / num_tokens, {
        "correct_tokens": total_correct_tokens,
        "total_masked_tokens": total_masked_tokens,
    }


def chungized_bert_loss(batch, model, mixed_precision, vocab_size, n_chungs):
    return chungized_llm_loss(
        input_tokens=batch.masked_tokens,
        gt_tokens=batch.tokens,
        mask=batch.mask_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
        n_chungs=n_chungs,
    )


def chungized_gpt_loss(batch, model, mixed_precision, vocab_size, n_chungs):
    return chungized_llm_loss(
        input_tokens=batch.tokens,
        gt_tokens=batch.target_tokens,
        mask=batch.non_padded_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
        n_chungs=n_chungs,
    )


def calculate_llm_loss(
    input_tokens, gt_tokens, mask, model, mixed_precision, vocab_size
):
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

    return loss, {
        "correct_tokens": correct_tokens,
        "total_masked_tokens": total_masked_tokens,
    }


def calculate_gpt_loss(batch, model, mixed_precision, vocab_size):
    return calculate_llm_loss(
        input_tokens=batch.tokens,
        gt_tokens=batch.target_tokens,
        mask=batch.non_padded_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
    )


def calculate_bert_loss(batch, model, mixed_precision, vocab_size):
    return calculate_llm_loss(
        input_tokens=batch.masked_tokens,
        gt_tokens=batch.tokens,
        mask=batch.mask_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
    )


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
    if args.granularity_expert_config:
        if (args.expert_size is not None) or (args.topk_fraction is not None):
            raise ValueError(
                "Cannot specify expert_size or topk_fraction when using granularity config"
            )

        expert_size = args.total_experts_width / args.n_experts
        assert expert_size == int(expert_size)
        expert_size = int(expert_size)

        experts_per_token = args.effective_dff / expert_size

        topk_fraction = experts_per_token / args.n_experts
        assert 0.0 <= topk_fraction <= 1.0
    else:
        expert_size = args.expert_size
        topk_fraction = args.topk_fraction

    return {
        "dmodel": args.dmodel,
        "n_experts": args.n_experts,
        "expert_size": expert_size,
        "topk_fraction": topk_fraction,
        "random_perm": args.expert_random_perm,
        "softmax_over": args.softmax_over,
        "group_granular_moe_by_batch": args.group_granular_moe_by_batch,
    }


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
            share_by_experts=args.share_by_experts,
            share_by_emit_merge=args.share_by_emit_merge,
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
        return_fn = lambda: ExpertChoiceFF(
            **get_expert_choice_args(args),
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
