from typing import Optional

import torch
from fancy_einsum import einsum

from lizrd.core.initialization import get_init_fun
from lizrd.core.misc import resolve_activation_name
from lizrd.core.misc import LoggingLayer, time_measured


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
        self.get_lin1 = lambda: self.lin1_weight
        self.get_lin2 = lambda: self.lin2_weight
        self.get_gate_weight = lambda: self.gate_weight

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        if self.use_einsum:
            experts_output = einsum(
                "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                x,
                self.get_lin1(),
            )

            gate = einsum(
                "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                x,
                self.get_gate_weight(),
            )
        else:
            experts_output = torch.matmul(x, self.get_lin1())
            gate = torch.matmul(x, self.get_gate_weight())

        experts_output = self.activation(gate) * experts_output

        if self.use_einsum:
            experts_output = einsum(
                "n_experts capacity expert_size, n_experts expert_size doutput -> n_experts capacity doutput",
                experts_output,
                self.get_lin2(),
            )
        else:
            experts_output = torch.matmul(experts_output, self.get_lin2())
        return experts_output


class ExpertClustered(ExpertGated):
    def __init__(self, clustering_interval, clustering_iters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clustering_interval = clustering_interval
        self.clustering_iters = clustering_iters
        self.curent_step = 0
        self.neuron_expert_perm = torch.arange(self.n_experts * self.expert_size)
        self.get_gate_weight = lambda: self.permute_weights(self.gate_weight)
        self.get_lin1 = lambda: self.permute_weights(self.lin1_weight)
        self.get_lin2 = lambda: self.permute_weights(self.lin2_weight, False)

    def permute_weights(self, weight, transpose=True):
        if transpose:
            weight = weight.permute(0, 2, 1)
        weight = weight.reshape(self.n_experts * self.expert_size, -1)
        weight = weight[self.neuron_expert_perm]
        weight = weight.reshape(self.n_experts, self.expert_size, -1)
        if transpose:
            weight = weight.permute(0, 2, 1)
        return weight

    @torch.no_grad()
    def calculate_new_matching(self):
        from k_means_constrained import KMeansConstrained

        weight = self.gate_weight.permute(0, 2, 1).reshape(
            self.n_experts * self.expert_size, -1
        )
        weight = torch.nn.functional.normalize(weight)
        inits = self.get_gate_weight().mean(dim=2)
        clf = KMeansConstrained(
            n_clusters=self.n_experts,
            size_min=self.expert_size,
            size_max=self.expert_size,
            n_init=1,
            max_iter=self.clustering_iters,
            init=inits.detach().cpu().numpy(),
        )
        labels = clf.fit_predict(weight.detach().cpu().numpy())
        counts = torch.zeros(self.n_experts)
        for i in range(self.n_experts * self.expert_size):
            self.neuron_expert_perm[i] = labels[i] * self.n_experts + counts[labels[i]]
            counts[labels[i]] += 1

    def check_matching(self):
        if self.training and self.curent_step % self.clustering_interval == 0:
            self.calculate_new_matching()
        if self.training:
            self.curent_step += 1

    def forward(self, x: torch.Tensor):
        return super().forward(x)


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
