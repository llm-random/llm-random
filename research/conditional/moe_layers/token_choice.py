import torch
import torch.nn.functional as F
from fancy_einsum import einsum

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.support.logging import make_histogram
from lizrd.train import checkpointing
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
        routing_top_k: int = 1,
        use_einsum: bool = False,
        vectorize: bool = True,
    ):
        super().__init__()

        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum
        self.dmodel = dmodel
        self.routing_top_k = routing_top_k
        self._checkpointed_expert_index = None
        self._checkpointed_top_tokens_per_expert_indices = None
        self.vectorize = vectorize

        if vectorize and routing_top_k != 1:
            raise ValueError("vectorize and routing_top_k != 1 are incompatible")

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

        x = x.reshape((n_tokens, self.dmodel))

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
        capacity = int(
            self.capacity_factor * n_tokens * self.routing_top_k / self.n_experts
        )
        if self.vectorize:
            capacity = min(capacity, n_tokens)

        # perform softmax over experts for each token
        with measure_time(self, "softmax"):
            gate_out = torch.softmax(gate_out, dim=1)

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)

        with measure_time(self, "choose_expert"):
            expert_gate, expert_index = self.choose_expert(gate_out)

        with measure_time(self, "create_expert_mask"):
            expanded_expert_mask = F.one_hot(expert_index, num_classes=self.n_experts)
            assert expanded_expert_mask.shape == (
                n_tokens,
                self.routing_top_k,
                self.n_experts,
            )
            expert_mask = expanded_expert_mask.sum(dim=1)
            assert expert_mask.shape == (n_tokens, self.n_experts)

        if self.vectorize:
            with measure_time(self, "experts_lists"):
                (
                    top_tokens_per_expert_values,
                    top_tokens_per_expert_indices,
                ) = expert_mask.topk(k=capacity, dim=0)
            with measure_time(self, "create_truncated_mask"):
                truncated_expert_mask = torch.zeros_like(expert_mask)
                truncated_expert_mask.scatter_(
                    dim=0,
                    index=top_tokens_per_expert_indices,
                    src=top_tokens_per_expert_values,
                )
        else:
            with measure_time(self, "experts_lists"):
                indices_of_tokens_for_expert = [
                    single_expert_mask.nonzero(as_tuple=True)[0][: (capacity - 1)]
                    for single_expert_mask in expert_mask.transpose(0, 1)
                ]
            with measure_time(self, "create_truncated_mask"):
                truncated_expert_mask = torch.zeros_like(expert_mask)
                for i, indices in enumerate(indices_of_tokens_for_expert):
                    truncated_expert_mask[indices, i] = 1

        n_selected_tokens = truncated_expert_mask.sum().item()

        self.update_cache_for_logging(
            "dropped_tokens_ratio",
            ((n_tokens * self.routing_top_k) - n_selected_tokens)
            / (n_tokens * self.routing_top_k),
        )

        with measure_time(self, "calculate aux loss"):
            tokens_per_expert = expert_mask.sum(dim=0, dtype=gate_out.dtype)
            load_balancing_loss = calculate_load_balancing_loss(
                self.load_balancing_loss_weight,
                gate_out,
                tokens_per_expert,
                use_einsum=self.use_einsum,
            )

        if "load_balancing_losses" not in self.forward_pass_cache:
            self.forward_pass_cache["load_balancing_losses"] = [load_balancing_loss]
        else:
            self.forward_pass_cache["load_balancing_losses"].append(load_balancing_loss)

        self.update_cache_for_logging("gate_softmax_values", expert_gate)
        self.update_cache_for_logging("max_indices", expert_index)
        self.update_cache_for_logging("tokens_per_expert", tokens_per_expert)
        self.update_cache_for_logging("load_balancing_loss", load_balancing_loss)

        masked_expert_gate = gate_out * truncated_expert_mask

        # create empty input and  assign only tokens to be processed
        experts_input = torch.zeros(
            self.n_experts, capacity, self.dmodel, dtype=x.dtype, device=x.device
        )
        if self.vectorize:
            with measure_time(self, "assign_tokens_to_input"):
                experts_input = x[
                    top_tokens_per_expert_indices.T.reshape(
                        (self.n_experts * capacity,)
                    ),
                    :,
                ] * top_tokens_per_expert_values.T.reshape(
                    (self.n_experts * capacity, 1)
                )
                experts_input = experts_input.reshape(
                    self.n_experts, capacity, self.dmodel
                )
        else:
            with measure_time(self, "assign_tokens_to_input"):
                for i, indices in enumerate(indices_of_tokens_for_expert):
                    experts_input[i, : len(indices)] = x[indices]
        if self.vectorize:
            return experts_input, top_tokens_per_expert_indices, masked_expert_gate
        else:
            return experts_input, indices_of_tokens_for_expert, masked_expert_gate

    def choose_tokens(self, expert_mask, capacity) -> tuple[torch.Tensor, torch.Tensor]:
        checkpointing_enabled = (
            checkpointing.is_in_first_forward() or checkpointing.is_in_second_forward()
        )
        if checkpointing_enabled:
            if checkpointing.is_in_first_forward():
                with torch.no_grad():
                    (
                        _,
                        top_tokens_per_expert_indices,
                    ) = expert_mask.topk(k=capacity, dim=0)
                    self._checkpointed_top_tokens_per_expert_indices = (
                        top_tokens_per_expert_indices
                    )

            if checkpointing.is_in_second_forward():
                with torch.no_grad():
                    top_tokens_per_expert_indices = (
                        self._checkpointed_top_tokens_per_expert_indices
                    )

            assert isinstance(top_tokens_per_expert_indices, torch.Tensor)
            top_tokens_per_expert_values = torch.gather(
                expert_mask,
                dim=1,
                index=top_tokens_per_expert_indices,
            )
        else:
            (
                top_tokens_per_expert_values,
                top_tokens_per_expert_indices,
            ) = expert_mask.topk(k=capacity, dim=0)
        return top_tokens_per_expert_values, top_tokens_per_expert_indices

    def choose_expert(self, gate_out) -> tuple[torch.Tensor, torch.Tensor]:
        checkpointing_enabled = (
            checkpointing.is_in_first_forward() or checkpointing.is_in_second_forward()
        )
        if checkpointing_enabled:
            if checkpointing.is_in_first_forward():
                with torch.no_grad():
                    _, expert_index = torch.topk(gate_out, k=self.routing_top_k, dim=1)
                    self._checkpointed_expert_index = expert_index

            if checkpointing.is_in_second_forward():
                with torch.no_grad():
                    expert_index = self._checkpointed_expert_index

            assert isinstance(expert_index, torch.Tensor)
            expert_gate = torch.gather(gate_out, dim=1, index=expert_index)
        else:
            expert_gate, expert_index = torch.topk(
                gate_out, k=self.routing_top_k, dim=1
            )
        return expert_gate, expert_index

    def log_light(self):
        return {
            "dropped_tokens_ratio": self.logging_cache["dropped_tokens_ratio"],
            "load_balancing_loss": self.logging_cache["load_balancing_loss"],
        }

    def log_heavy(self):
        return {
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()  # move
            ),
            "tokens_per_expert_counts": make_histogram(
                self.logging_cache["tokens_per_expert"]
            ),
        }


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
        routing_top_k: int = 1,
        use_einsum: bool = False,
        vectorize: bool = True,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
            expert_logic: expert logic layer, takes input of shape (n_experts, capacity, dmodel) and returns output of shape (n_experts, capacity, dmodel)
        """
        super().__init__()

        self.dmodel = dmodel
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.expert_inner_function = expert_inner_function
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum
        self.vectorize = vectorize

        if vectorize and routing_top_k != 1:
            raise ValueError("vectorize and routing_top_k != 1 are incompatible")

        self.router = TokenChoiceRouter(
            dmodel=dmodel,
            n_experts=n_experts,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=routing_top_k,
            use_einsum=use_einsum,
            vectorize=vectorize,
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        if self.vectorize:
            (
                experts_input,
                top_tokens_per_expert_indices,  # [c x n_experts]
                masked_expert_gate,
            ) = self.router(x)
        else:
            (
                experts_input,
                indices_of_tokens_for_expert,  # n_experts lists, each max c long
                masked_expert_gate,
            ) = self.router(x)

        x = x.flatten(start_dim=0, end_dim=1)

        experts_output = self.expert_inner_function(experts_input)

        experts_output = experts_output.to(x.dtype)
        output = torch.zeros_like(x)

        if self.vectorize:
            with measure_time(self, "assign_tokens_to_output"):
                output.index_add_(
                    dim=0,
                    index=top_tokens_per_expert_indices.T.flatten(),
                    source=experts_output.reshape(
                        self.n_experts * experts_output.shape[1], self.dmodel
                    ),
                )
                output *= masked_expert_gate.sum(dim=1, keepdim=True)
        else:
            with measure_time(self, "assign_tokens_to_output"):
                for expert_id in range(self.n_experts):
                    tokens_to_update = indices_of_tokens_for_expert[expert_id]
                    num_of_tokens_to_update = len(tokens_to_update)

                    output[tokens_to_update] += experts_output[
                        expert_id, :num_of_tokens_to_update
                    ] * masked_expert_gate[tokens_to_update, expert_id].unsqueeze(dim=1)

        output = output.reshape((batch_size, seq_len, self.dmodel))

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
    ):
        super().__init__()
        self.dmodel = dmodel
        self.n_experts = n_experts
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

    def forward(self, x: torch.Tensor):
        (n_experts, capacity, dmodel) = x.shape

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
                    "n_experts capacity expert_size, n_experts expert_size dmodel -> n_experts capacity dmodel",
                    experts_output,
                    self.lin2_weight,
                )
            else:
                experts_output = torch.matmul(experts_output, self.lin2_weight)

        assert experts_output.shape == (n_experts, capacity, dmodel)

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
    ):
        super().__init__()

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

        self.swi_glu_gate_weight = nn.Parameter(
            get_init_weight(
                shape=(n_experts, dmodel, expert_size),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
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
                    "n_experts capacity expert_size, n_experts expert_size dmodel -> n_experts capacity dmodel",
                    experts_output,
                    self.lin2_weight,
                )
            else:
                experts_output = torch.matmul(experts_output, self.lin2_weight)

        return experts_output


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
