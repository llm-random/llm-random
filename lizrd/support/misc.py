import datetime
import random
import string
from typing import Any, Dict, Optional, List


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


def count_parameters(model, args, VOCAB_SIZE):
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_embedding_and_head_params = 2 * VOCAB_SIZE * args.dmodel
    pos_embedding_params = args.cutoff * args.dmodel
    model_n_params -= input_embedding_and_head_params + pos_embedding_params
    return model_n_params


def count_moe_non_emb_active_params(dmodel, effective_dff_x, dff, n_blocks):
    return (
        dmodel**2
        * (4 + 2 * (effective_dff_x if effective_dff_x is not None else dff / dmodel))
        * n_blocks
    )


def count_tokens_per_step(batch_size, cutoff):
    return batch_size * cutoff


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


def calculate_current_batch_size_from_rampup(
    processed_tokens,
    transition_points,
    batch_sizes,
    target_batch_size,
):
    transition_points_in_tokens = [point * 1e9 for point in transition_points]

    if processed_tokens >= transition_points_in_tokens[-1]:
        return target_batch_size

    for i in reversed(range(len(transition_points_in_tokens) - 1)):
        if processed_tokens >= transition_points_in_tokens[i]:
            return batch_sizes[i + 1]

    return batch_sizes[0]
