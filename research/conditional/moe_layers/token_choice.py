from typing import Optional

import torch
import torch.nn.functional as F
from fancy_einsum import einsum

from lizrd.core.initialization import get_init_fun
from research.conditional.utils.layer_manager import (
    LoggingLayer,
    measure_time,
    time_measured,
)
from research.conditional.moe_layers.moe_gating import TokenGating


class TokenChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        expert_inner_function: LoggingLayer,
        doutput: Optional[int] = None,
        routing_top_k: int = 1,
        use_einsum: bool = False,
    ):
        """
        Args:
            dmodel: dimension of the input
            doutput: dimension of the output (default: dmodel)
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
            expert_logic: expert logic layer, takes input of shape (n_experts, capacity, dmodel) and returns output of shape (n_experts, capacity, dmodel)
        """
        super().__init__()

        self.dmodel = dmodel
        self.doutput = self.dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.expert_inner_function = expert_inner_function
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.router = TokenGating(
            dmodel=dmodel,
            n_experts=n_experts,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=routing_top_k,
            use_einsum=use_einsum,
        )

    @time_measured("assign_tokens_to_input")
    def routing_emit(self, x, tokens_per_expert_indices, tokens_per_expert_values):
        capacity = tokens_per_expert_indices.shape[0]
        indicies_reshaped = tokens_per_expert_indices.T.reshape(
            self.n_experts * capacity
        )
        values_reshaped = tokens_per_expert_values.T.reshape(
            self.n_experts * capacity, 1
        )
        experts_input = x[indicies_reshaped, :] * values_reshaped
        experts_input = experts_input.reshape(self.n_experts, capacity, self.dmodel)
        return experts_input

    @time_measured("assign_tokens_to_output")
    def routing_merge(
        self,
        experts_output,
        masked_expert_gate,
        tokens_per_expert_indices,
        batch_size,
        seq_len,
        x,
    ):
        output = torch.zeros(
            batch_size * seq_len,
            self.doutput,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
        output.index_add_(
            dim=0,
            index=tokens_per_expert_indices.T.flatten(),
            source=experts_output.reshape(
                self.n_experts * experts_output.shape[1], self.doutput
            ),
        )
        output *= masked_expert_gate.sum(dim=1, keepdim=True)
        output = output.reshape(batch_size, seq_len, self.doutput)
        return output

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        (
            tokens_per_expert_indices,
            tokens_per_expert_values,
            masked_expert_gate,
        ) = self.router(x)

        x = x.flatten(start_dim=0, end_dim=1)
        experts_input = self.routing_emit(
            x, tokens_per_expert_indices, tokens_per_expert_values
        )
        experts_output = self.expert_inner_function(experts_input).to(x.dtype)
        output = self.routing_merge(
            experts_output,
            masked_expert_gate,
            tokens_per_expert_indices,
            batch_size,
            seq_len,
            x,
        )
        return output


class ExpertRelu(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        init_type: str,
        init_scale: float,
        use_einsum: bool = False,
        doutput: Optional[int] = None,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.doutput = dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.use_einsum = use_einsum

        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        self.lin1_weight = init(shape=(n_experts, dmodel, expert_size), fan_in=dmodel)
        self.lin2_weight = init(
            shape=(n_experts, expert_size, self.doutput),
            fan_in=int(n_experts * expert_size),
        )

    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape

        assert n_experts == self.n_experts
        assert dmodel == self.dmodel

        with measure_time(self, "process_by_experts"):
            if self.use_einsum:
                experts_output = einsum(
                    "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                    x,
                    self.lin1_weight,
                )
            else:
                experts_output = torch.matmul(x, self.lin1_weight)

            experts_output = F.relu(experts_output)
            if self.use_einsum:
                experts_output = einsum(
                    "n_experts capacity expert_size, n_experts expert_size doutput -> n_experts capacity doutput",
                    experts_output,
                    self.lin2_weight,
                )
            else:
                experts_output = torch.matmul(experts_output, self.lin2_weight)

        assert experts_output.shape == (n_experts, capacity, self.doutput)
        return experts_output


class ExpertSwiGLU(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        init_type: str,
        init_scale: float,
        use_einsum: bool = False,
        doutput: Optional[int] = None,
    ):
        super().__init__()
        self.doutput = dmodel if doutput is None else doutput
        self.use_einsum = use_einsum

        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        self.lin1_weight = init(shape=(n_experts, dmodel, expert_size), fan_in=dmodel)
        self.lin2_weight = init(
            shape=(n_experts, expert_size, self.doutput),
            fan_in=int(n_experts * expert_size),
        )
        self.swi_glu_gate_weight = init(
            shape=(n_experts, dmodel, expert_size), fan_in=dmodel
        )

    def forward(self, x: torch.Tensor):
        with measure_time(self, "process_by_experts"):
            if self.use_einsum:
                experts_output = einsum(
                    "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                    x,
                    self.lin1_weight,
                )

                swi_glu_gate = einsum(
                    "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                    x,
                    self.swi_glu_gate_weight,
                )
            else:
                experts_output = torch.matmul(x, self.lin1_weight)
                swi_glu_gate = torch.matmul(x, self.swi_glu_gate_weight)

            experts_output = F.silu(experts_output) * swi_glu_gate

            if self.use_einsum:
                experts_output = einsum(
                    "n_experts capacity expert_size, n_experts expert_size doutput -> n_experts capacity doutput",
                    experts_output,
                    self.lin2_weight,
                )
            else:
                experts_output = torch.matmul(experts_output, self.lin2_weight)
        return experts_output
