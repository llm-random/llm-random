import torch
import torch.nn.functional as F
from fancy_einsum import einsum

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.support.logging import make_histogram
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time


class TokenChoiceRouter(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        use_einsum: bool = False,
    ):
        super().__init__()

        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum
        self.dmodel = dmodel

        self.gate = nn.Parameter(
            get_init_weight(
                shape=(dmodel, n_experts),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len, _ = x.shape
        n_tokens = batch_size * seq_len
        self.update_cache_for_logging("n_tokens", torch.Tensor([n_tokens]))

        x = x.flatten(start_dim=0, end_dim=1)

        with measure_time(self, "expert_embedding"):
            if self.use_einsum:
                gate_out = einsum(
                    "n_tokens dmodel, dmodel n_experts -> n_tokens n_experts",
                    x,
                    self.gate,
                )
            else:
                gate_out = torch.matmul(x, self.gate)

        assert gate_out.shape == (n_tokens, self.n_experts)
        capacity = int(self.capacity_factor * n_tokens / self.n_experts)

        # perform softmax over experts for each token
        with measure_time(self, "softmax"):
            gate_out = torch.softmax(gate_out, dim=1)

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)

        # choose expert for each token
        with measure_time(self, "max_indices"):
            expert_gate, expert_index = torch.max(gate_out, dim=1)

        with measure_time(self, "create_expert_mask"):
            expert_mask = F.one_hot(expert_index, num_classes=self.n_experts)

        with measure_time(self, "calculate expert indexes"):
            position_in_expert = torch.cumsum(expert_mask, dim=0) * expert_mask
            in_capacity_tokens_mask = torch.lt(position_in_expert, capacity)
            expert_mask *= in_capacity_tokens_mask

        with measure_time(self, "calculate aux loss"):
            position_in_expert_mask = position_in_expert.bool()
            tokens_per_expert = position_in_expert_mask.sum(dim=0, dtype=gate_out.dtype)
            load_balancing_loss = calculate_load_balancing_loss(
                self.load_balancing_loss_weight,
                gate_out,
                tokens_per_expert,
                use_einsum=self.use_einsum,
            )

            if "load_balancing_losses" not in self.forward_pass_cache:
                self.forward_pass_cache["load_balancing_losses"] = [load_balancing_loss]
            else:
                self.forward_pass_cache["load_balancing_losses"].append(
                    load_balancing_loss
                )

        # mask out tokens that are not in capacity
        expert_mask_flat = expert_mask.sum(dim=1)
        expert_gate *= expert_mask_flat

        self.update_cache_for_logging("gate_softmax_values", expert_gate)
        self.update_cache_for_logging("max_indices", expert_index)
        self.update_cache_for_logging("tokens_per_expert", tokens_per_expert)
        self.update_cache_for_logging("load_balancing_loss", load_balancing_loss)
        # group tokens indices by expert it should be processed by
        with measure_time(self, "experts_lists"):
            indices_of_tokens_for_expert = [
                single_expert_mask.nonzero(as_tuple=True)[0]
                for single_expert_mask in expert_mask.transpose(0, 1)
            ]

        # create empty input and  assign only tokens to be processed
        experts_input = torch.zeros(
            self.n_experts, capacity, self.dmodel, dtype=x.dtype, device=x.device
        )
        with measure_time(self, "assign_tokens_to_input"):
            for i, indices in enumerate(indices_of_tokens_for_expert):
                experts_input[i, : len(indices)] = x[indices]

        return experts_input, indices_of_tokens_for_expert, expert_gate


class TokenChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        use_einsum: bool = False,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
        """
        super().__init__()

        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum

        self.lin1_weight = nn.Parameter(
            get_init_weight(
                shape=(n_experts, dmodel, expert_size),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )
        self.lin2_weight = nn.Parameter(
            get_init_weight(
                shape=(n_experts, expert_size, dmodel),
                fan_in=int(n_experts * expert_size),
                init_type=init_type,
                scale=init_scale,
            )
        )

        self.router = TokenChoiceRouter(
            dmodel=dmodel,
            n_experts=n_experts,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            use_einsum=use_einsum,
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        experts_input, indices_of_tokens_for_expert, expert_gate = self.router(x)
        x = x.flatten(start_dim=0, end_dim=1)

        with measure_time(self, "process_by_experts"):
            if self.use_einsum:
                experts_output = einsum(
                    "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
                    experts_input,
                    self.lin1_weight,
                )
            else:
                experts_output = torch.matmul(experts_input, self.lin1_weight)

            experts_output = F.relu(experts_output)
            if self.use_einsum:
                experts_output = einsum(
                    "n_experts capacity expert_size, n_experts expert_size dmodel -> n_experts capacity dmodel",
                    experts_output,
                    self.lin2_weight,
                ).to(x.dtype)
            else:
                experts_output = torch.matmul(experts_output, self.lin2_weight)

        output = torch.zeros_like(x)

        with measure_time(self, "assign_tokens_to_output"):
            for i in range(self.n_experts):
                output[indices_of_tokens_for_expert[i]] = experts_output[
                    i, : len(indices_of_tokens_for_expert[i])
                ]

        # multiply output by softmax values
        with measure_time(self, "multiply_output_by_softmax"):
            if self.use_einsum:
                output = einsum(
                    "n_tokens dmodel, n_tokens -> n_tokens dmodel", output, expert_gate
                )
            else:
                output = output * expert_gate.unsqueeze(dim=1)

        output = output.reshape((batch_size, seq_len, self.dmodel))

        return output

    def log_heavy(self):
        return {
            "gradient_of_gate_distribution": make_histogram(self.gate.grad.flatten()),
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()
            ),
            "tokens_per_expert_counts": make_histogram(
                self.logging_cache["tokens_per_expert"]
            ),
            "load_balancing_loss": self.logging_cache["load_balancing_loss"],
        }


def calculate_load_balancing_loss(
    alpha: float,
    softmax_per_token: torch.Tensor,
    n_tokens_in_each_expert: torch.Tensor,
    use_einsum: bool = False,
):
    """
    Calculates the load balancing loss for the token choice layer.

    :param str alpha: aux loss weigth parameter
    :param torch.Tensor softmax_per_token: tensor of shape (tokens, n_experts)
    :param torch.Tensor tokens_in_each_expert: tensor of shape (n_experts)
    """
    n_tokens, n_experts = softmax_per_token.shape
    assert n_experts == n_tokens_in_each_expert.shape[0]

    per_expert_softmax_sum = torch.mean(softmax_per_token, dim=0)

    if use_einsum:
        dot_product = einsum("i, i ->", per_expert_softmax_sum, n_tokens_in_each_expert)
    else:
        dot_product = torch.dot(per_expert_softmax_sum, n_tokens_in_each_expert)
    load_balancing_loss = alpha * n_experts * dot_product / n_tokens
    return load_balancing_loss
