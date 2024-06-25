from typing import Optional

import torch
from fancy_einsum import einsum
from typing import Literal


from lizrd.core.initialization import get_init_fun
from lizrd.core.misc import resolve_activation_name
from lizrd.core.misc import LoggingLayer, time_measured
from lizrd.core.kan import KanFF


class ExpertKAN(LoggingLayer):
    def __init__(
        self,
        kan_type: str,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        use_topk_initialization,
        init_type: Literal["kaiming_uniform", "truncated_normal"] = "kaiming_uniform",
        init_scale: float = 0.1,
        init_scale_base: float = 1.0,
        init_scale_spline: float = 1.0,
        init_scale_noise: float = 0.1,
        latent_factor: float = 1.0,
        parameter_matched: str = "true",
        activation_name: str = "relu",
        topk: int = 1,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.activation = resolve_activation_name(activation_name)
        self.doutput = dmodel

        print(f"\nexpert_size = dff = {expert_size}\n")

        self.kan_experts = torch.nn.ModuleList(
            [
                KanFF(
                    dmodel=dmodel,
                    dff=expert_size,
                    kan_type=kan_type,
                    init_type=init_type,
                    init_scale=init_scale,
                    init_scale_base=init_scale_base,
                    init_scale_spline=init_scale_spline,
                    init_scale_noise=init_scale_noise,
                    latent_factor=latent_factor,
                    parameter_matched=parameter_matched,
                )
                for _ in range(n_experts)
            ]
        )

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape

        assert n_experts == self.n_experts
        assert dmodel == self.dmodel

        results = []
        for i in range(n_experts):
            results.append(self.kan_experts[i](x[i, :, :]))

        output = torch.stack(results, dim=0)

        assert output.shape == x.shape

        return output


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
        self.lin1_weight = self.init_fun(
            shape=(n_experts, dmodel, expert_size), fan_in=dmodel
        )
        self.lin2_weight = self.init_fun(
            shape=(n_experts, expert_size, self.doutput),
            fan_in=int(fan_in_factor * expert_size),
        )

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
        fan_in: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.doutput = expert_size
        self.n_experts = n_experts

        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        self.lin1_weight = init(
            shape=(n_experts, dmodel, expert_size), fan_in=fan_in or dmodel
        )

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape
        assert (n_experts, dmodel) == (self.n_experts, self.dmodel)

        experts_output = torch.matmul(x, self.lin1_weight)
        assert experts_output.shape == (n_experts, capacity, self.doutput)
        return experts_output
