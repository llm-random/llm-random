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
        topk: int = 1,
        use_topk_initialization: bool = False,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.doutput = dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.use_einsum = use_einsum
        self.expert_size = expert_size
        self.activation = resolve_activation_name(activation_name)

        fan_in_factor = topk if use_topk_initialization else n_experts
        self.init_fun = get_init_fun(init_type=init_type, init_scale=init_scale)
        self._lin1_weight = self.init_fun(
            shape=(n_experts, dmodel, expert_size), fan_in=dmodel
        )
        self._lin2_weight = self.init_fun(
            shape=(n_experts, expert_size, self.doutput),
            fan_in=int(fan_in_factor * expert_size),
        )

    @property
    def lin1_weight(self):
        return self._lin1_weight.reshape(self.n_experts, self.dmodel, -1)

    @property
    def lin2_weight(self):
        return self._lin2_weight.reshape(self.n_experts, -1, self.doutput)

    def double_n_experts(self):
        self.n_experts = 2 * self.n_experts

    def half_n_experts(self):
        self.n_experts = self.n_experts // 2

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape

        assert n_experts == self.n_experts
        assert dmodel == self.dmodel

        # maybe remove these einsums that just multiply two tensors? This will never be faster than torch.matmul
        # unlike of course einsums with >2 tensors
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


class ExpertGated(ExpertFF):
    def __init__(
        self,
        *args,
        activation_name: str = "silu",
        **kwargs,
    ):
        super().__init__(*args, activation_name=activation_name, **kwargs)
        self.gate_weight = self.init_fun(
            shape=(self.n_experts, self.dmodel, self.expert_size), fan_in=self.dmodel
        )

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

        experts_output = self.activation(gate) * experts_output

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

    def double_n_experts(self):
        raise NotImplementedError

    def half_n_experts(self):
        raise NotImplementedError

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape
        assert (n_experts, dmodel) == (self.n_experts, self.dmodel)

        experts_output = torch.matmul(x, self.lin1_weight)
        assert experts_output.shape == (n_experts, capacity, self.doutput)
        return experts_output
