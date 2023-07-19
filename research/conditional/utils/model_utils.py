import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lizrd.core import llm
from research.conditional.archive.continuous_moe_alternatives import (
    ContinuousMoEQuickMergeDifferentlySimple,
    ContinuousMoEQuickMergeDifferentlyCommonBase,
    ContinuousMoEQuickRawmerge,
    ContinuousMoEQuickTopmerge,
    ContinuousMoEQuickNosoftmax,
    ContinuousMoEQuickAdaTemp,
)
from research.conditional.moe_layers.continuous_moe import (
    ContinuousMoE,
    ContinuousMoEQuick,
    FeedForwardTimed,
)
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF


def chungized_bert_loss(
    batch, model, mixed_precision, vocab_size, mask_percent, n_chungs
):
    def make_custom_forward(vocab_size):
        def custom_forward(*inputs):
            with torch.autocast(
                device_type="cuda", enabled=mixed_precision, dtype=torch.float16
            ):
                output = model.head(inputs[0])
                gt = inputs[1]
                mask = inputs[2]
                loss = F.cross_entropy(
                    output.reshape(-1, vocab_size),
                    gt.reshape(-1).long(),
                    reduction="none",
                )
            return loss * mask.reshape(-1)

        return custom_forward

    input = batch.masked_tokens
    non_masked_input = batch.tokens
    non_masked_mask = batch.mask_mask

    encoder_output = model.encoder(input)

    chunged_inputs = torch.chunk(encoder_output, n_chungs, dim=0)
    chunged_non_masked_inputs = torch.chunk(non_masked_input, n_chungs, dim=0)
    chunged_non_masked_masks = torch.chunk(non_masked_mask, n_chungs, dim=0)

    num_tokens = 0
    total_loss = 0
    for chunged_input, chunged_gt, chunged_mask in zip(
        chunged_inputs, chunged_non_masked_inputs, chunged_non_masked_masks
    ):
        partial_loss_output = checkpoint(
            make_custom_forward(vocab_size), chunged_input, chunged_gt, chunged_mask
        )

        num_tokens += partial_loss_output.shape[0]
        total_loss += partial_loss_output.sum()
    return total_loss / num_tokens / mask_percent


def calculate_gpt_loss(batch, model, mixed_precision, vocab_size):
    input = batch.tokens
    target = batch.target_tokens
    non_padded_mask = batch.non_padded_mask

    if mixed_precision:
        with torch.autocast(
            device_type="cuda", enabled=mixed_precision, dtype=torch.float16
        ):
            model_output = model(input)
    else:
        model_output = model(input)

    lm_loss = F.cross_entropy(
        model_output.reshape(-1, vocab_size),
        target.reshape(-1).long(),
        reduction="none",
    )
    lm_loss *= non_padded_mask.reshape(-1)
    loss = lm_loss.mean()
    return loss


def calculate_bert_loss(batch, model, mixed_precision, vocab_size, mask_percent):
    input = batch.masked_tokens
    non_masked_input = batch.tokens
    non_masked_mask = batch.mask_mask

    if mixed_precision:
        with torch.autocast(
            device_type="cuda", enabled=mixed_precision, dtype=torch.float16
        ):
            model_output = model(input)
    else:
        model_output = model(input)

    mask_loss = F.cross_entropy(
        model_output.reshape(-1, vocab_size),
        non_masked_input.reshape(-1).long(),
        reduction="none",
    )
    mask_loss *= non_masked_mask.reshape(-1)
    loss = mask_loss.mean() / mask_percent
    return loss


def get_attention_layer(args):
    if args.model_type == "gpt":
        attention_layer_fun = lambda: llm.CausalAttention(args.dmodel, args.n_att_heads)
    elif args.model_type == "bert":
        attention_layer_fun = lambda: llm.Attention(args.dmodel, args.n_att_heads)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    return attention_layer_fun


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
    }


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(args.dmodel, args.dff)
    elif args.ff_mode == "vanilla_timed":
        return_fn = lambda: FeedForwardTimed(args.dmodel, args.dff)
    elif args.ff_mode == "cont_moe":
        return_fn = lambda: ContinuousMoE(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
        )
    elif args.ff_mode == "cont_moe_quick":
        return_fn = lambda: ContinuousMoEQuick(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
            args.use_opt_einsum,
        )
    elif args.ff_mode == "cont_moe_quick_merge_diff_simple":
        return_fn = lambda: ContinuousMoEQuickMergeDifferentlySimple(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
            args.use_opt_einsum,
        )
    elif args.ff_mode == "cont_moe_quick_merge_diff_comm_base":
        return_fn = lambda: ContinuousMoEQuickMergeDifferentlyCommonBase(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
            args.use_opt_einsum,
        )
    elif args.ff_mode == "cont_moe_quick_rawmerge":
        return_fn = lambda: ContinuousMoEQuickRawmerge(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
            args.use_opt_einsum,
        )
    elif args.ff_mode == "cont_moe_quick_topmerge":
        return_fn = lambda: ContinuousMoEQuickTopmerge(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
            args.use_opt_einsum,
        )
    elif args.ff_mode == "cont_moe_quick_nosoft":
        return_fn = lambda: ContinuousMoEQuickNosoftmax(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
            args.use_opt_einsum,
        )
    elif args.ff_mode == "cont_moe_quick_adatemp":
        return_fn = lambda: ContinuousMoEQuickAdaTemp(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
            args.expert_size,
            args.use_opt_einsum,
            args.share_by_experts,
            args.share_by_emit_merge,
        )
    elif args.ff_mode == "expert_choice":
        return_fn = lambda: ExpertChoiceFF(
            **get_expert_choice_args(args),
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
