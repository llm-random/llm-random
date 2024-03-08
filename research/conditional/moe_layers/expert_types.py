from typing import Optional

import torch
from fancy_einsum import einsum

from lizrd.core.initialization import get_init_fun
from lizrd.core.misc import resolve_activation_name
from research.conditional.utils.layer_manager import LoggingLayer, time_measured


class ExpertFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        init_type: str,
        init_scale: float,
        use_einsum: bool = False,
        doutput: Optional[int] = None,
        activation_name: str = "relu",
    ):
        super().__init__()
        self.dmodel = dmodel
        self.doutput = dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.use_einsum = use_einsum
        self.activation = resolve_activation_name(activation_name)

        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        self.lin1_weight = init(shape=(n_experts, dmodel, expert_size), fan_in=dmodel)
        self.lin2_weight = init(
            shape=(n_experts, expert_size, self.doutput),
            fan_in=int(n_experts * expert_size),
        )

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape

        assert n_experts == self.n_experts
        assert dmodel == self.dmodel

        if self.use_einsum:
            experts_output = einsum(
                "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                x,
                self.lin1_weight,
            )
        else:
            experts_output = torch.matmul(x, self.lin1_weight)

        experts_output = self.activation(experts_output)
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


class ExpertGated(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        init_type: str,
        init_scale: float,
        use_einsum: bool = False,
        doutput: Optional[int] = None,
        activation_name: str = "silu",
    ):
        super().__init__()
        self.doutput = dmodel if doutput is None else doutput
        self.use_einsum = use_einsum
        self.activation = resolve_activation_name(activation_name)

        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        self.lin1_weight = init(shape=(n_experts, dmodel, expert_size), fan_in=dmodel)
        self.lin2_weight = init(
            shape=(n_experts, expert_size, self.doutput),
            fan_in=int(n_experts * expert_size),
        )
        self.gate_weight = init(shape=(n_experts, dmodel, expert_size), fan_in=dmodel)

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        if self.use_einsum:
            experts_output = einsum(
                "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                x,
                self.lin1_weight,
            )

            gate = einsum(
                "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                x,
                self.gate_weight,
            )
        else:
            experts_output = torch.matmul(x, self.lin1_weight)
            gate = torch.matmul(x, self.gate_weight)

        experts_output = self.activation(experts_output) * gate

        if self.use_einsum:
            experts_output = einsum(
                "n_experts capacity expert_size, n_experts expert_size doutput -> n_experts capacity doutput",
                experts_output,
                self.lin2_weight,
            )
        else:
            experts_output = torch.matmul(experts_output, self.lin2_weight)
        return experts_output


class ExpertLinear(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        init_type: str,
        init_scale: float,
        doutput: Optional[int] = None,
    ):
        super().__init__()
        self.dmodel = dmodel
        assert doutput is None or doutput == expert_size
        self.doutput = expert_size
        self.n_experts = n_experts

        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        self.lin1_weight = init(shape=(n_experts, dmodel, expert_size), fan_in=dmodel)

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape
        assert n_experts == self.n_experts
        assert dmodel == self.dmodel

        experts_output = torch.matmul(x, self.lin1_weight)
        assert experts_output.shape == (n_experts, capacity, self.doutput)
        return experts_output
