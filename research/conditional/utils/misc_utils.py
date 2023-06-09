import time
from itertools import product

import torch

from research.conditional.moe_layers.continuous_moe import (
    ContinuousMoE,
    ContinuousMoEQuick,
)


def introduce_parser_arguments(parser):
    # core hyperparameters, fixed for all experiments; needs a good reason to change

    parser.add_argument("--use_clearml", action="store_true")
    parser.add_argument("--use_neptune", action="store_true")
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cutoff", type=int, default=128)
    parser.add_argument("--dmodel", type=int, default=768)
    parser.add_argument("--dff", type=int, default=3072)
    parser.add_argument("--n_att_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--logging_interval_light", type=int, default=1000000)
    parser.add_argument("--logging_interval_heavy", type=int, default=1000000)
    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_percent", type=float, default=0.15)
    parser.add_argument("--n_steps", type=int, default=90000)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--tags", nargs="*", type=str, default=None)
    parser.add_argument(
        "--model_type", type=str, choices=["gpt", "bert"], default="bert"
    )

    # parameters usually changed for experiments

    parser.add_argument("--ff_mode", type=str, default="vanilla")
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--n_experts", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--sparsity_dim", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--expert_size", type=int, default=-1)
    parser.add_argument("--topk_fraction", type=float, required=False)
    parser.add_argument("--logging_interval_loss", type=int, default=250)
    parser.add_argument("--every_other_layer", action="store_true")
    parser.add_argument("--expert_random_perm", action="store_true")
    parser.add_argument("--standard_ff_first", action="store_true")
    parser.add_argument("--use_opt_einsum", action="store_true")

    # experimental/legacy parameters

    parser.add_argument("--hack_name", type=str, default=None)
    parser.add_argument("--x_flop", action="store_true")
    parser.add_argument("--x_logarithmic", action="store_true")

    return parser


def get_parameter_size_in_gb(parameter):
    return parameter.data.numel() * parameter.data.element_size() / (1024**3)


def determine_mem_usage():
    dm = 768
    dff = 3072
    n_experts__ = [128]
    group_size__ = [2, 16, 128]
    sparsity_dim = 1
    temperature = 1.0
    batch_size = 600
    seq_len = 128

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for group_size, n_experts in product(group_size__, n_experts__):
        for expertsize in [
            2,
            200,
            400,
        ]:
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"memory_allocated after freeing: {memory_allocated}")
            module = ContinuousMoE(
                dm, dff, n_experts, group_size, sparsity_dim, temperature, expertsize
            ).to(DEVICE)

            input_batch = torch.randn(batch_size, seq_len, dm).to(DEVICE)

            torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats before forward pass

            # Forward pass to compute activations
            activations = module(input_batch)

            # Calculate memory usage
            memory_usage = torch.cuda.max_memory_allocated(DEVICE) / (
                1024**3
            )  # Convert bytes to gigabytes
            print(
                f"n_experts: {n_experts}, group_size: {group_size}, expertsize: {expertsize}",
                "Memory Usage: {:.2f} GB".format(memory_usage),
            )
            # print size of module components: model.lin1, model.lin2, model.controller
            print(
                f"model.lin1: {get_parameter_size_in_gb(module.lin1)}, model.lin2: {get_parameter_size_in_gb(module.lin2)}, model.controller: {get_parameter_size_in_gb(module.controller)}"
            )
            del module, input_batch, activations
            breakpoint()


def einsum_opt_v_normal():
    dm = 768
    dff = 3072
    n_experts = 32
    group_size = 8
    sparsity_dim = 1
    expertsize = 100
    use_opt_einsum = [True, False]
    temperature = 1.0
    batch_size = 600
    seq_len = 128

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    breakpoint()
    for use_opt in use_opt_einsum:
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"memory_allocated after freeing: {memory_allocated}")
        module = ContinuousMoEQuick(
            dm,
            dff,
            n_experts,
            group_size,
            sparsity_dim,
            temperature,
            expertsize,
            use_opt_einsum=use_opt,
        ).to(DEVICE)

        input_batch = torch.randn(batch_size, seq_len, dm).to(DEVICE)
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats before forward pass

        for _ in range(10):
            time_start = time.time()
            activations = module(input_batch)
            loss = torch.sum(activations)
            loss.backward()
            time_end = time.time()
            print(f"use_opt: {use_opt}; Time taken: {time_end - time_start}")
            # Calculate memory usage
            memory_usage = torch.cuda.memory_allocated(DEVICE) / (
                1024**3
            )  # Convert bytes to gigabytes
            print(
                f"use_opt: {use_opt}",
                "Memory Usage: {:.2f} GB".format(memory_usage),
            )
            torch.cuda.empty_cache()
            del loss, activations


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-8), dim=-1)
