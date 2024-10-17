import datetime
import random
import string
from typing import Any, Dict, Optional, List
from lizrd.core.llm import EmbeddingLayer


def tags_to_name(tags: Optional[List[str]]) -> str:
    return "_".join(tags) if tags else ""


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


def count_tokens_per_step(batch_size, cutoff):
    return batch_size * cutoff


def count_token_to_active_ratio(tokens_per_step, n_active, args):
    tokens_per_active_ratio = (
        (args.n_steps * tokens_per_step / n_active)
        if args.n_steps is not None
        else None
    )
    return tokens_per_active_ratio


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
