import datetime
import random
import string
from typing import Any, Dict, Optional, List
from lizrd.core.llm import EmbeddingLayer
from research.batch_size_rampup_config import BatchSizeRampupConfig


def tags_to_name(tags: Optional[List[str]]) -> str:
    return "_".join(tags) if tags else ""


def list_to_str(args_list: list):
    args_list = [str(elem) for elem in args_list]
    return ", ".join(args_list)


def make_concise_datetime() -> str:
    now = datetime.datetime.now()
    return str(now.year)[-2:] + "_" + now.strftime("%m-%d_%H:%M:%S")


def get_n_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_expert_parameters(model):
    # Filter for parameters with 'expert_inner_function' in their name and that require gradients
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if "expert_inner_function" in name and p.requires_grad
    )


def get_total_nonembedding_parameters(model):
    n_learnable_parameters = get_n_learnable_parameters(model)

    embedding = [m for m in model.modules() if isinstance(m, EmbeddingLayer)][0]
    head = model.head

    n_learnable_nonembedding_parameters = (
        n_learnable_parameters
        - get_n_learnable_parameters(embedding)
        - get_n_learnable_parameters(head)
    )
    return n_learnable_nonembedding_parameters


def get_active_nonembedding_parameters(args, model):
    total_nonembedding_parameters = get_total_nonembedding_parameters(model)
    if args.ff_mode in ["token_choice", "expert_choice"]:
        all_expert_parameters = get_expert_parameters(model)
        active_expert_parameters = int(all_expert_parameters * args.topk_fraction)
        active_nonembedding_parameters = (
            total_nonembedding_parameters
            - all_expert_parameters
            + active_expert_parameters
        )
    else:
        active_nonembedding_parameters = total_nonembedding_parameters

    return active_nonembedding_parameters


def get_model_configuration_for_active_param_calculation(args):
    dmodel = args.dmodel
    isgated = False
    dff = None
    active_ratio = None
    ismoe = False

    if args.ff_mode in ["vanilla", "vanilla_timed"]:
        isgated = False
        active_ratio = 1
        dff = args.dff

    elif args.ff_mode in ["swi_glu"]:
        isgated = True
        active_ratio = 1
        dff = args.dff

    elif args.ff_mode in ["token_choice", "expert_choice"]:
        active_ratio = args.topk_fraction
        dff = args.total_experts_width
        ismoe = True

        if args.moe_inner_expert in ["relu", "ff"]:
            isgated = False
        elif args.moe_inner_expert in ["swi_glu", "ff_gated", "geglu"]:
            isgated = True
        elif args.moe_inner_expert in ["linear"]:
            return None
        else:
            raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")

    elif args.ff_mode in [
        "expert_choice_with_parallel_ff",
        "token_choice_old",
        "double_choice",
        "expert_choice_old",
        "cont_moe",
        "cont_moe_quick",
        "cont_moe_merge_diff_simple",
        "cont_moe_merge_diff_comm_base",
        "cont_moe_rawmerge",
        "cont_moe_topmerge",
        "cont_moe_nosoft",
        "cont_moe_adatemp",
        "cont_moe_adatemp_positive",
        "cont_moe_ln",
        "cont_moe_final",
        "cont_moe_random_groups",
        "cont_moe_common_weighted_parameters" "cont_moe_separate_weighted_parameters",
        "cont_moe_legacy",
        "kernelized_fc",
    ]:
        return None
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")

    return dmodel, isgated, dff, active_ratio, ismoe


def calculate_from_args_model_parameter_counts(args, vocab_size):
    model_configuration = get_model_configuration_for_active_param_calculation(args)
    if model_configuration is None:
        # when counting parameters for your model type isn't implemented, this function returns 1 for all param-counts.
        # This is a safe value that:
        #   1. won't kill the experiment. We don't want that, because counting parameters is mainly for nice neptune loss plots, not a necessary feature.
        #   2. Values are obviously wrong, so someone should notice that they don't have their parameter logic implemented.
        print(
            "Parameter counting not implemented for this model. Consider adding it to lizrd/support/misc.py calculate_from_args_model_parameter_counts function"
        )
        return (
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        )

    else:
        embedding_parameters = vocab_size * args.dmodel + args.cutoff * args.dmodel
        head_parameters = vocab_size * args.dmodel
        (
            dmodel,
            isgated,
            dff,
            active_ratio,
            ismoe,
        ) = model_configuration

        layer_norm_params = 2 * dmodel

        attention_params = (dmodel**2) * 4  # Q, K, V and O projections

        if isgated:
            ff_total_params = 3 * dmodel * dff
        else:
            ff_total_params = 2 * dmodel * dff

        ff_active_params = (
            ff_total_params * active_ratio
        )  # when not MoE, active_ratio = 1

        print(f"experts active/total: {ff_active_params}/{ff_total_params}")

        router_params = 0
        if ismoe:
            router_params = dmodel * args.n_experts
            print(f"router: {router_params}")

        n_blocks = args.n_blocks
        all_layer_norm_params = (
            n_blocks * layer_norm_params * 2
        )  # 2 because LN in attention and LN in FF
        all_attention_params = n_blocks * attention_params
        all_ff_total_params = n_blocks * ff_total_params
        all_ff_active_params = n_blocks * ff_active_params
        all_router_params = n_blocks * router_params

    return (
        embedding_parameters,
        all_layer_norm_params,
        all_attention_params,
        all_ff_total_params,
        all_ff_active_params,
        all_router_params,
        head_parameters,
    )


def count_tokens_per_step(batch_size, cutoff):
    return batch_size * cutoff


def count_token_to_active_ratio(tokens, active):
    return tokens / active


def generate_random_string(length: int) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def get_argument_attributes(args: Any, params: list) -> Dict[str, Any]:
    if (
        args.model_fit_gpu_info_database_path is not None
        and args.model_fit_gpu_info_params is not None
    ):
        params = args.model_fit_gpu_info_params.split(",")
        return {param: getattr(args, param) for param in params}


def set_seed(seed):
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


def get_ith_chunk(tensor, chunks, i):
    import torch

    list_of_chunks = torch.chunk(tensor, chunks, dim=0)
    return list_of_chunks[i]


def calculate_n_processed_tokens(
    step: int,
    seq_len: int,
    target_batch_size_per_gpu: int,
    n_gpus: int,
    rampup_config: Optional[BatchSizeRampupConfig],
):
    if rampup_config is None:
        return step * n_gpus * target_batch_size_per_gpu * seq_len
    else:
        transition_points = [p * 1e9 for p in rampup_config.transition_points]
        steps_in_previous_intervals = 0
        tokens_in_previous_intervals = 0
        for point, batch_size in zip(transition_points, rampup_config.batch_sizes):
            total_steps_after_this_interval = steps_in_previous_intervals + (
                point - tokens_in_previous_intervals
            ) // (batch_size * seq_len)
            if step < total_steps_after_this_interval:
                # The current step is within this ramp-up interval
                return (
                    tokens_in_previous_intervals
                    + (step - steps_in_previous_intervals) * batch_size * seq_len
                )
            else:
                tokens_in_previous_intervals = (
                    tokens_in_previous_intervals
                    + (total_steps_after_this_interval - steps_in_previous_intervals)
                    * batch_size
                    * seq_len
                )
                steps_in_previous_intervals = total_steps_after_this_interval

        # we have reached the target interval
        return (
            tokens_in_previous_intervals
            + (step - steps_in_previous_intervals)
            * n_gpus
            * target_batch_size_per_gpu
            * seq_len
        )


def calculate_current_batch_size_from_rampup(
    processed_tokens,
    transition_points,
    batch_sizes,
    target_batch_size,
):
    for transition_point, batch_size in zip(transition_points, batch_sizes):
        if (
            processed_tokens < transition_point * 1e9
        ):  # transition points are given in billions of tokens
            return batch_size
    return target_batch_size
