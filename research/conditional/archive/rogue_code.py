import time
from itertools import product

import torch

from research.conditional.moe_layers.continuous_moe import (
    ContinuousMoE,
    ContinuousMoEQuick,
)


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


def set_highest_index_one(tensor: torch.Tensor) -> torch.Tensor:
    # Get the index of the highest value in the last dimension
    _, indices = torch.max(tensor, dim=-1)

    # Create a new tensor filled with zeros, with the same shape as the input tensor
    result_tensor = torch.zeros_like(tensor)

    # Calculate index shape for the new tensor
    result_shape = list(range(tensor.dim() - 1))

    # Set 1 at the index of the highest value in each sub-array
    result_tensor.scatter_(
        -1, indices.view(tensor.shape[:-1] + (1,)).expand(tensor.shape), 1
    )

    return result_tensor
