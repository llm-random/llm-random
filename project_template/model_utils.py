from functools import partial

from typing import Optional, Type, Union, Callable
import torch
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from lizrd.core import llm
from lizrd.text.data import LLMBatch


def make_loss_and_gradient_function(
    loss_checkpoint_chungs: int,
) -> Callable:
    if loss_checkpoint_chungs == 0:
        return calculate_llm_loss_and_gradient
    else:
        return partial(chungized_llm_loss_and_gradient, n_chungs=loss_checkpoint_chungs)


def calculate_single_chung_loss(
    model: torch.nn.Module,
    encoder_output: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
):
    output = model(encoder_output)
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

def chungized_llm_loss_and_gradient(
    batch: LLMBatch,
    model: torch.nn.Module,
    n_chungs: int,
    num_checkpoint_accumulation_steps: int,
) -> tuple[float, dict]:
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

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
            chunged_encoder_output,
            chunged_gt,
            chunged_mask,
        )
        partial_loss = (
            single_chung_loss.mean() / n_chungs / num_checkpoint_accumulation_steps
        )
        if model.training:
            partial_loss.backward()
        total_loss += partial_loss.item()
        total_correct_tokens += single_chung_correct_tokens
        total_masked_tokens += single_chung_masked_tokens

    aux_info = {
        "correct_tokens": total_correct_tokens,
        "total_masked_tokens": total_masked_tokens,
    }

    if model.training:
        # ok, we need to backward one loss (because of torch autograd)
        # the "loss" that has the same gradient as the original cross entropy loss is the sum below
        assert encoder_output_detach.grad.shape == encoder_output.shape
        loss_to_optimize = (encoder_output * encoder_output_detach.grad).sum()
        loss_to_optimize.backward()
    return total_loss, aux_info


def calculate_llm_loss_and_gradient(
    batch: LLMBatch,
    model: torch.nn.Module,
    num_checkpoint_accumulation_steps: int,
) -> tuple[float, dict]:
    def hack_for_python_garbage_collection():
        """we want to have no reference to model output while backpropagating to allow torch to free memory,
        so we wrap loss calculation in a function"""
        input_tokens = batch.input_ids
        gt_tokens = batch.target_ids
        mask = batch.should_calculate_loss

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
        }
        return loss, aux_info

    loss, aux_info = hack_for_python_garbage_collection()
    if model.training:
        loss_to_optimize = loss.clone()
        loss_to_optimize.backward()

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


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    elif args.ff_mode == "swi_glu":
        return_fn = lambda: llm.SwiGLUFeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
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

    selective_precision_modules = get_classes_from_module_names(
        args.fsdp_selective_precision_modules
    )
    if selective_precision_modules is not None:
        ignored_classes += list(selective_precision_modules)

    return ignored_classes
