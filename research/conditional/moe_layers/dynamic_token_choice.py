from typing import Optional, Union

import torch

from lizrd.core.initialization import get_init_fun
from lizrd.train import checkpointing
from research.conditional.utils.layer_manager import (
    LoggingLayer,
    measure_time,
    time_measured,
)
from research.conditional.moe_layers.moe_gating import TokenGating
from fancy_einsum import einsum


class DynamicMoeGating(LoggingLayer):
    def __init__(
        self,
        n_experts,
        dexpert,
        group_by_batch,
        softmax_ungrouped,
        softmax_over,
        use_torch_bmm,
        neurons_embeddings,
        experts_embeddings,
        gate,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.dexpert = dexpert
        self.group_by_batch = group_by_batch
        self.softmax_ungrouped = softmax_ungrouped
        self.softmax_over = softmax_over
        self.use_torch_bmm = use_torch_bmm
        self.neurons_embeddings = neurons_embeddings
        self.experts_embeddings = experts_embeddings
        self.gate = gate
        self._checkpointed_topk_indices: Union[None, torch.Tensor] = None
        assert softmax_over in ["tokens", "experts"]

    def calculate_gate(self, x, batch_size, seq_len):
        with measure_time(self, "make_experts"):
            neuron_scores = torch.einsum(
                "ni,ei->ne", self.neurons_embeddings, self.experts_embeddings
            )
            neuron_weights, neuron_indices = neuron_scores.topk(k=self.dexpert, dim=0)
            neuron_weights = neuron_weights.relu()

        with measure_time(self, "expert_embedding"):
            if self.use_torch_bmm:
                gate = self.gate.unsqueeze(0).expand(batch_size, -1, -1)
                gate_out = torch.bmm(x, gate).permute(2, 0, 1)
                assert gate_out.shape == (self.n_experts, batch_size, seq_len)
            else:
                gate_out = einsum(
                    "batch_size seq_len dmodel, dmodel n_experts "
                    "-> n_experts batch_size seq_len ",
                    x,
                    self.gate,
                )
        # each expert chooses k within dimension 1
        if not self.group_by_batch and not self.softmax_ungrouped:
            gate_out = gate_out.reshape(self.n_experts, batch_size * seq_len)
        # perform softmax either over tokens for each expert or over experts for each token
        with measure_time(self, "softmax"):
            if self.softmax_over == "tokens":
                gate_out = torch.softmax(gate_out, dim=1)
            elif self.softmax_over == "experts":
                gate_out = torch.softmax(gate_out, dim=0)
        if self.softmax_ungrouped:
            gate_out = gate_out.reshape(self.n_experts, batch_size * seq_len)

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)
        return gate_out

    def calculate_topk(self, gate_out, topk):
        is_in_first = checkpointing.is_in_first_forward()
        is_in_second = checkpointing.is_in_second_forward()
        checkpointing_enabled = is_in_first or is_in_second
        if is_in_first and is_in_second:
            raise NotImplementedError(
                "Both first and second forward are = TRUE. You are probably using wrapped and nested checkpointed modules, which is not supported with ExpertGating."
            )
        if checkpointing_enabled:
            # In first forward we discard the first result of topk (topk_values)
            # and instead use gather.
            # This is needed if activation checkpointing is used, because
            # torch aligns tensors in both forward passes by the order in
            # which they are created and that is the easiest way to do that.

            with torch.no_grad():
                if is_in_first:
                    _, topk_indices = torch.topk(gate_out, k=topk, dim=1)
                    self._checkpointed_topk_indices = topk_indices
                if is_in_second:
                    topk_indices = self._checkpointed_topk_indices

            topk_values = gate_out.gather(dim=1, index=topk_indices)
        else:
            topk_values, topk_indices = torch.topk(gate_out, k=topk, dim=1)
        return topk_indices, topk_values


class DynamicTokenGating(DynamicMoeGating):
    def __init__(
        self,
        dmodel: int,
        total_n_neurons: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        routing_top_k: int = 1,
        use_einsum: bool = False,
    ):
        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        # in_projections = torch.randn([total_n_neurons, dm])
        # out_projections = torch.randn([total_n_neurons, dm])
        # neurons_embeddings = torch.randn([total_n_neurons, dm])
        # experts_embeddings = torch.randn([n_exps, dm])

        in_projections = init(shape=(dmodel, n_experts), fan_in=dmodel)
        out_projections = init(shape=(dmodel, n_experts), fan_in=dmodel)
        neurons_embeddings = init(shape=(n_experts, dmodel), fan_in=dmodel)
        experts_embeddings = init(shape=(total_n_neurons, dmodel), fan_in=dmodel)
        # gate = init(shape=(dmodel, n_experts), fan_in=dmodel)

        super().__init__(
            n_experts,
            group_by_batch=False,
            softmax_ungrouped=False,
            softmax_over="experts",
            use_torch_bmm=not use_einsum,
            gate=gate,
        )

        self.dmodel = dmodel
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum
        self.routing_top_k = routing_top_k

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len, _ = x.shape
        n_tokens = batch_size * seq_len
        capacity = min(
            int(self.capacity_factor * n_tokens * self.routing_top_k / self.n_experts),
            n_tokens,
        )
        self.update_cache_for_logging("n_tokens", torch.Tensor([n_tokens]))

        gate_out = self.calculate_gate(x, batch_size, seq_len).T
        assert gate_out.shape == (n_tokens, self.n_experts)

        with measure_time(self, "choose_expert"):
            expert_index, expert_gate = self.calculate_topk(
                gate_out, self.routing_top_k
            )

        self.update_cache_for_logging("gate_softmax_values", expert_gate)
        self.update_cache_for_logging("max_indices", expert_index)

        return self.apply_capacity(capacity, expert_index, gate_out, n_tokens)

    def calculate_balancing_loss(self, gate_out, expert_mask):
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
        self.update_cache_for_logging("tokens_per_expert", tokens_per_expert)
        self.update_cache_for_logging("load_balancing_loss", load_balancing_loss)

    def apply_capacity(self, capacity, expert_index, gate_out, n_tokens):
        # create a mask telling if a token is assigned to an expert
        with measure_time(self, "create_expert_mask"):
            expanded_expert_mask = F.one_hot(expert_index, num_classes=self.n_experts)
            assert expanded_expert_mask.shape == (
                n_tokens,
                self.routing_top_k,
                self.n_experts,
            )
            expert_mask = expanded_expert_mask.sum(dim=1)
            assert expert_mask.shape == (n_tokens, self.n_experts)
        # now apply fixed capacity: for a given expert we can have only capacity tokens
        with measure_time(self, "experts_lists"):
            (
                top_tokens_per_expert_values,
                top_tokens_per_expert_indices,
            ) = expert_mask.topk(k=capacity, dim=0)
        self.log_dropped_tokens(
            top_tokens_per_expert_values,
            top_tokens_per_expert_indices,
            expert_mask,
            n_tokens,
        )
        # from a list of finally chosen tokens, create a mask with their respective values
        expert_values = (
            torch.gather(gate_out, 0, top_tokens_per_expert_indices)
            * top_tokens_per_expert_values
        )
        self.calculate_balancing_loss(gate_out, expert_mask)
        return top_tokens_per_expert_indices, expert_values

    def log_dropped_tokens(
        self,
        top_tokens_per_expert_values,
        top_tokens_per_expert_indices,
        expert_mask,
        n_tokens,
    ):
        # TODO this below is just for logging, we maybe should remove it
        with measure_time(self, "create_truncated_mask"):
            truncated_expert_mask = torch.zeros_like(expert_mask)
            truncated_expert_mask.scatter_(
                dim=0,
                index=top_tokens_per_expert_indices,
                src=top_tokens_per_expert_values,
            )
        n_selected_tokens = truncated_expert_mask.sum().item()
        self.update_cache_for_logging(
            "dropped_tokens_ratio",
            ((n_tokens * self.routing_top_k) - n_selected_tokens)
            / (n_tokens * self.routing_top_k),
        )

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


class DynamicTokenChoiceFF(LoggingLayer):
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

        token_expert_indices, token_expert_values = self.router(x)

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
