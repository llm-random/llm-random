from typing import Optional
import torch

from lizrd.core.misc import (
    LoggingLayer,
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
        zloss_weight: float = 0,
        routing_top_k: int = 1,
        use_einsum: bool = False,
        get_router_values_from: str = "weights",
        moe_values_exp: Optional[int] = 1,
        detach_gate: bool = False,
        **_,
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
        self.n_experts = n_experts
        self.expert_inner_function = expert_inner_function
        self.doutput = self.expert_inner_function.doutput
        self.gating = TokenGating(
            dmodel=dmodel,
            n_experts=n_experts,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=routing_top_k,
            use_einsum=use_einsum,
            get_router_values_from=get_router_values_from,
            detach_gate=detach_gate,
            expert_inner_function=self.expert_inner_function,
            moe_values_exp=moe_values_exp,
            zloss_weight=zloss_weight,
        )

    @time_measured("assign_tokens_to_input")
    def extract(self, x, token_indicies):
        capacity = token_indicies.shape[0]
        token_indicies = token_indicies.T.reshape(self.n_experts * capacity)
        experts_input = x[token_indicies, :]
        experts_input = experts_input.reshape(self.n_experts, capacity, self.dmodel)
        return experts_input

    @time_measured("assign_tokens_to_output")
    def merge(
        self,
        experts_output,
        token_expert_values,
        token_expert_indices,
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
        experts_output *= token_expert_values.T.unsqueeze(-1)
        output.index_add_(
            dim=0,
            index=token_expert_indices.T.flatten(),
            source=experts_output.reshape(
                self.n_experts * experts_output.shape[1], self.doutput
            ),
        )
        output = output.reshape(batch_size, seq_len, self.doutput)
        return output

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        token_expert_indices, token_expert_values = self.gating(x)

        x = x.flatten(start_dim=0, end_dim=1)
        experts_input = self.extract(x, token_expert_indices)
        experts_output = self.expert_inner_function(experts_input).to(x.dtype)
        output = self.merge(
            experts_output,
            token_expert_values,
            token_expert_indices,
            batch_size,
            seq_len,
            x,
        )
        return output
