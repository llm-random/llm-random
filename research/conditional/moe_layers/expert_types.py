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
        use_mock_gating_g: int = 0,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.doutput = dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.use_einsum = use_einsum
        self.expert_size = expert_size
        self.use_mock_gating_g = use_mock_gating_g
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
        if self.use_mock_gating_g == 1:
            self.mock_gating = self.init_fun(
                shape=(dmodel, self.n_experts),
                fan_in=dmodel,
            )
        assert 0 <= self.use_mock_gating_g <= 1

        self.mock_gating_act = torch.nn.Softmax(dim=2)
        self.get_lin1 = lambda: self.lin1_weight
        self.get_lin2 = lambda: self.lin2_weight

    def pre_mocked_gating(self, x):
        if self.use_mock_gating_g > 0:
            mul = einsum(
                "n_experts_1 capacity dmodel, dmodel n_experts_2 -> n_experts_1 capacity n_experts_2",
                x,
                self.mock_gating,
            )
            mul = self.mock_gating_act(mul)
            mul = mul[torch.arange(self.n_experts), :, torch.arange(self.n_experts)]
            return mul

    def post_mocked_gating(self, x, mul):
        if self.use_mock_gating_g > 0:
            x = x * mul.unsqueeze(2)
        return x

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape

        assert n_experts == self.n_experts
        assert dmodel == self.dmodel

        mul = self.pre_mocked_gating(x)
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

        experts_output = self.post_mocked_gating(experts_output, mul)
        assert experts_output.shape == (n_experts, capacity, self.doutput)
        return experts_output


class ExpertProjectedFF(LoggingLayer):
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
        weights_base_relative: float = 1.0,
        use_layer_norm: bool = False,
        project_only_one_dim: bool = False,
        project_skip_softmax: bool = False,
        project_whole_matrix: bool = False,
        use_id_init: bool = False,
        detach_projection_for_iters: int = 0,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.doutput = dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.use_einsum = use_einsum
        self.expert_size = expert_size
        self.activation = resolve_activation_name(activation_name)
        self.n_neurons = expert_size * n_experts
        self.use_layer_norm = use_layer_norm
        self.project_only_one_dim = project_only_one_dim
        self.project_skip_softmax = project_skip_softmax
        self.project_whole_matrix = project_whole_matrix
        self.detach_projection_for_iters = detach_projection_for_iters
        self.weights_base_size = int(self.n_neurons * weights_base_relative)
        self.use_id_init = use_id_init

        if self.use_layer_norm:
            self.layer_norm_1 = torch.nn.LayerNorm(self.dmodel)
            self.layer_norm_2 = torch.nn.LayerNorm(self.expert_size)

        fan_in_factor = topk if use_topk_initialization else n_experts
        self.init_fun = get_init_fun(init_type=init_type, init_scale=init_scale)

        if self.project_only_one_dim or self.project_whole_matrix:
            self.weights_base_size = int(self.weights_base_size / expert_size)
            self.lin1_weight = self.init_fun(
                shape=(expert_size, self.weights_base_size, dmodel), fan_in=dmodel
            )
            self.lin2_weight = self.init_fun(
                shape=(expert_size, self.weights_base_size, self.doutput),
                fan_in=int(fan_in_factor * expert_size),
            )
        else:
            self.lin1_weight = self.init_fun(
                shape=(self.weights_base_size, dmodel), fan_in=dmodel
            )
            self.lin2_weight = self.init_fun(
                shape=(self.weights_base_size, self.doutput),
                fan_in=int(fan_in_factor * expert_size),
            )
        size = (
            (n_experts, self.weights_base_size)
            if self.project_whole_matrix
            else (n_experts, expert_size, self.weights_base_size)
        )
        if self.use_id_init:
            if self.project_whole_matrix and self.weights_base_size == n_experts:
                init_v = torch.eye(n_experts)
                if not self.project_skip_softmax:
                    init_v *= 10
                self.neuron_selector = torch.nn.Parameter(init_v).requires_grad_(True)
            else:
                raise NotImplementedError(
                    "ID initialization is not implemented for this case"
                )
        else:
            self.neuron_selector = self.init_fun(
                shape=size, fan_in=self.weights_base_size
            )

        self.selector_act = torch.nn.Softmax(dim=1 if self.project_whole_matrix else 2)
        self.iter = 0

    @time_measured("calculate_experts")
    def calculate_experts(self):
        selector = (
            self.neuron_selector
            if self.project_skip_softmax
            else self.selector_act(self.neuron_selector)
        )
        self.iter += 1
        if self.iter < self.detach_projection_for_iters:
            selector = selector.detach()
        if self.project_whole_matrix:
            lin1 = einsum(
                "n_experts weights_base, expert_size weights_base dmodel "
                "-> n_experts dmodel expert_size",
                selector,
                self.lin1_weight,
            )
            lin2 = einsum(
                "n_experts weights_base, expert_size weights_base doutput "
                "-> n_experts expert_size doutput",
                selector,
                self.lin2_weight,
            )
        elif self.project_only_one_dim:
            lin1 = einsum(
                "n_experts expert_size weights_base, expert_size weights_base dmodel "
                "-> n_experts dmodel expert_size",
                selector,
                self.lin1_weight,
            )
            lin2 = einsum(
                "n_experts expert_size weights_base, expert_size weights_base doutput "
                "-> n_experts expert_size doutput",
                selector,
                self.lin2_weight,
            )
        else:
            lin1 = einsum(
                "n_experts expert_size weights_base, weights_base dmodel -> n_experts dmodel expert_size",
                selector,
                self.lin1_weight,
            )
            lin2 = einsum(
                "n_experts expert_size weights_base, weights_base doutput -> n_experts expert_size doutput",
                selector,
                self.lin2_weight,
            )
        if self.use_layer_norm:
            lin1 = self.layer_norm_1(lin1.permute(0, 2, 1)).permute(0, 2, 1)
            lin2 = self.layer_norm_2(lin2.permute(0, 2, 1)).permute(0, 2, 1)
        return lin1, lin2

    def forward(self, x: torch.Tensor):
        n_experts, capacity, dmodel = x.shape

        assert n_experts == self.n_experts
        assert dmodel == self.dmodel
        lin1, lin2 = self.calculate_experts()

        experts_output = self.calculate_ff(capacity, n_experts, x, lin1, lin2)
        return experts_output

    @time_measured("process_by_experts")
    def calculate_ff(self, capacity, n_experts, x, lin1, lin2):
        # maybe remove these einsums that just multiply two tensors? This will never be faster than torch.matmul
        # unlike of course einsums with >2 tensors
        if self.use_einsum:
            experts_output = einsum(
                "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                x,
                lin1,
            )
        else:
            experts_output = torch.matmul(x, lin1)
        experts_output = self.activation(experts_output)
        if self.use_einsum:
            experts_output = einsum(
                "n_experts capacity expert_size, n_experts expert_size doutput -> n_experts capacity doutput",
                experts_output,
                lin2,
            )
        else:
            experts_output = torch.matmul(experts_output, lin2)
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
        self.get_gate_weight = lambda: self.gate_weight

    @time_measured("process_by_experts")
    def forward(self, x: torch.Tensor):
        mul = self.pre_mocked_gating(x)

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

        experts_output = self.post_mocked_gating(experts_output, mul)

        return experts_output


class ExpertClustered(ExpertGated):
    def __init__(self, clustering_interval, clustering_iters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clustering_interval = clustering_interval
        self.clustering_iters = clustering_iters
        self.current_step = 0
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
            self.neuron_expert_perm[i] = (
                labels[i] * self.expert_size + counts[labels[i]]
            )
            counts[labels[i]] += 1

    def check_matching(self):
        if self.training and self.current_step % self.clustering_interval == 0:
            self.calculate_new_matching()
        if self.training:
            self.current_step += 1

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
