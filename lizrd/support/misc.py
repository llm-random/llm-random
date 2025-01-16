import datetime
import random
import string
from typing import Any, Dict, Optional, List
from lizrd.core.llm import EmbeddingLayer


def tags_to_name(tags: Optional[List[str]]) -> str:
    return "_".join(tags) if tags else ""


def list_to_str(args_list: list):
    args_list = [str(elem) for elem in args_list]
    return ", ".join(args_list)


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
        "projected_vanilla",
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


def convert_steps_to_tokens(
    step: int,
    seq_len: int,
    target_batch_size: int,
    transition_points: list[int] = None,
    batch_sizes: list[int] = None,
):
    if transition_points is None:
        return int(step * target_batch_size * seq_len)

    tokens_per_step_list = [batch_size * seq_len for batch_size in batch_sizes]
    steps_prev = tokens_prev = 0

    for point, tokens_per_step in zip(transition_points, tokens_per_step_list):
        steps_to_transition = point - steps_prev
        point_in_steps = point

        if step < point_in_steps:
            return int(tokens_prev + (step - steps_prev) * tokens_per_step)

        tokens_to_transition = steps_to_transition * tokens_per_step
        tokens_prev += tokens_to_transition
        steps_prev = point_in_steps

    # After all ramp-up intervals
    return int(tokens_prev + (step - steps_prev) * target_batch_size * seq_len)


def convert_tokens_to_steps(
    tokens: int,  # in bilions
    seq_len: int,
    target_batch_size: int,
    transition_points: list[int] = None,
    batch_sizes: list[int] = None,
):
    if transition_points is None:
        return int((tokens) / (target_batch_size * seq_len))

    tokens

    tokens_per_step_list = [batch_size * seq_len for batch_size in batch_sizes]
    steps_prev = tokens_prev = 0

    for point, tokens_per_step in zip(transition_points, tokens_per_step_list):
        steps_to_transition = point - steps_prev
        tokens_to_trasition = steps_to_transition * tokens_per_step
        tokens_current = tokens_prev + tokens_to_trasition

        print(f"tokens_current: {tokens_current}")

        if tokens < tokens_current:
            return int(steps_prev + (tokens - tokens_prev) / tokens_per_step)

        tokens_prev = tokens_current
        steps_prev = point

    # After all ramp-up intervals
    return int(steps_prev + (tokens - tokens_prev) / (target_batch_size * seq_len))


# this function is needed, because both convert tokens to steps and steps to tokens use transition points in steps
def convert_transition_points_in_tokens_to_steps(
    transition_points_in_tokens: list[float], batch_sizes: list[int], seq_len: int
):
    transition_points_in_steps = []
    transition_points_in_tokens = [p * 1e9 for p in transition_points_in_tokens]
    tokens_per_step_list = [batch_size * seq_len for batch_size in batch_sizes]
    steps_prev = tokens_prev = 0

    for point, tokens_per_step in zip(
        transition_points_in_tokens, tokens_per_step_list
    ):
        tokens_to_transition = point - tokens_prev
        steps_to_transition = tokens_to_transition / tokens_per_step
        point_in_steps = steps_prev + steps_to_transition

        transition_points_in_steps.append(int(point_in_steps))

        tokens_prev = point
        steps_prev = point_in_steps
    return transition_points_in_steps


def get_batch_size(
    step,
    target_batch_size: int,
    transition_points: list[int] = None,
    batch_sizes: list[int] = None,
):
    if transition_points is not None:
        for point, batch_size in zip(transition_points, batch_sizes):
            if step < point:
                return batch_size
    return target_batch_size
