from typing import Optional, Type, Union, Callable
from functools import partial

import torch
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.profiler import ProfilerAction

from lizrd.core import llm

from research.tokenizex.utils.data import TokenizexBatch
from research.tokenizex.model.input_wise_attention import ManagerMaskSetter
from research.tokenizex.model.input_wise_pe import ManagerPESetter


def make_loss_and_gradient_function(
    loss_checkpoint_chungs: int,
) -> Callable:
    if loss_checkpoint_chungs == 0:
        return calculate_llm_loss_and_gradient
    else:
        raise NotImplementedError("Loss chungization not supported")
        return partial(chungized_llm_loss_and_gradient, n_chungs=loss_checkpoint_chungs)


def run_backward(
    loss: torch.Tensor,
    mixed_precision_dtype: torch.dtype,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    with torch.autocast(
        device_type="cuda", enabled=False, dtype=mixed_precision_dtype
    ):  # dev do i need to have configured pe and att masks here?
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()


def calculate_llm_loss_and_gradient(  # dev TODO
    batch: TokenizexBatch,
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
        tokens_positions = batch.positions
        attention_masks = batch.attention_mask
        deftok_byte_scale = batch.deftok_byte_scale

        with torch.autocast(
            device_type="cuda", enabled=mixed_precision, dtype=mixed_precision_dtype
        ), ManagerMaskSetter(model, attention_masks), ManagerPESetter(
            model, tokens_positions
        ):
            model_output = model(input_tokens)

        # move the gt tokens and mask to the same device as the model output - they should be on the same device for loss calculation
        gt_tokens = gt_tokens.to(model_output.device)

        mask = mask.to(model_output.device)

        mask_loss = F.cross_entropy(
            model_output.flatten(0, -2),  # dev check debug
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

        aux_info["deftok_loss"] = loss.item() * deftok_byte_scale

        return loss, aux_info

    loss, aux_info = hack_for_python_garbage_collection()
    for key, value in aux_info["losses"].items():
        aux_info["losses"][key] = value / num_checkpoint_accumulation_steps

    if model.training:
        loss_to_optimize = loss.clone()
        for value in aux_info["losses"].values():
            loss_to_optimize += value
        run_backward(loss_to_optimize, mixed_precision_dtype, scaler)  # dev checkpoin

    clear_additional_losses(model)
    return loss.item(), aux_info


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
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    elif args.ff_mode == "swi_glu":
        return_fn = lambda: llm.SwiGLUFeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")
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


def update_model_fit_gpu_info(database: Optional[str], params: dict, value: str):
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
