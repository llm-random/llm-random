import torch
import torch.nn.functional as F

from lizrd.core import llm
from research.conditional.moe_layers.ffs import ContinuousMoE
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF


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


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return lambda: llm.FeedForward(args.dmodel, args.dff)
    elif args.ff_mode == "cont_moe":
        return lambda: ContinuousMoE(
            args.dmodel,
            args.dff,
            args.n_experts,
            args.group_size,
            args.sparsity_dim,
            args.temperature,
        )
    elif args.ff_mode == "expert_choice":
        return lambda: ExpertChoiceFF(
            dmodel=args.dmodel,
            n_experts=args.n_experts,
            expert_size=args.expert_size,
            cutoff=args.cutoff,
            topk=args.topk,
        )
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")
