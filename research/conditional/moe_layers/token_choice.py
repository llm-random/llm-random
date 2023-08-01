import plotly.express as px
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
from torch.nn import LayerNorm

from lizrd.core import nn
from lizrd.core.misc import get_init_weight
from lizrd.support.logging import make_histogram
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time


class TokenChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_size: size of each expert
            topk_fraction: fraction of tokens that will be chosen for each expert
            random_perm: randomly permute tokens for experts (ablation). Note that
                network can still learn which tokens to choose,
                but not which expert to choose for token
        """
        super().__init__()

        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size

        self.lin1_weights = torch.nn.ParameterList(
            [
                nn.Parameter(get_init_weight((dmodel, expert_size), fan_in=dmodel))
                for _ in range(n_experts)
            ]
        )
        self.lin2_weights = torch.nn.ParameterList(
            [
                nn.Parameter(
                    get_init_weight(
                        (expert_size, dmodel),
                        fan_in=expert_size,
                    )
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Parameter(
            get_init_weight((dmodel, n_experts), fan_in=dmodel)
        ).requires_grad_(True)
        self.ln = LayerNorm(dmodel)

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len, _ = x.shape

        with measure_time(self, "expert_embedding"):
            gate_out = einsum(
                "batch_size seq_len dmodel, dmodel n_experts -> batch_size seq_len n_experts",
                x,
                self.gate,
            )

            # flatten batch_size x seq_len
            gate_out = gate_out.flatten(start_dim=0, end_dim=1)

        # perform softmax over experts for each token
        with measure_time(self, "softmax"):
            gate_out = torch.softmax(gate_out, dim=1)

        self.cache("gate_softmax_all_values", gate_out)

        # choose expert for each token
        with measure_time(self, "max_indices"):
            gate_values, experts_indices = torch.max(gate_out, dim=1)

        #  group tokens by expert it should be processed by
        with measure_time(self, "experts_lists"):
            experts_lists = [
                torch.eq(experts_indices, i).nonzero(as_tuple=True)[0]
                for i in range(self.n_experts)
            ]

        # flatten x s. t. first dimension is tokens instead of batch_size x seq_len
        with measure_time(self, "flatten"):
            x = x.flatten(start_dim=0, end_dim=1)
        final_output = torch.zeros_like(x)
        for i in range(self.n_experts):
            expert_output = einsum(
                "list_size dmodel, dmodel expert_size -> list_size expert_size",
                x[experts_lists[i]],
                self.lin1_weights[i],
            )
            expert_output = F.relu(expert_output)
            expert_output = einsum(
                "list_size expert_size, expert_size dmodel -> list_size dmodel",
                expert_output,
                self.lin2_weights[i],
            )
            final_output[experts_lists[i], :] = expert_output

        self.cache("gate_softmax_values", gate_values)
        self.cache("max_indices", experts_indices)
        self.cache("n_tokens", torch.Tensor([batch_size * seq_len]))

        # multiply final_output by softmax values
        with measure_time(self, "multiply_softmax"):
            final_output = einsum(
                "n_size dmodel, n_size -> n_size dmodel", final_output, gate_values
            )
        final_output = final_output.reshape((batch_size, seq_len, self.dmodel))

        with measure_time(self, "layer_norm"):
            final_output = self.ln(final_output)

        return final_output

    def log_light(self):
        return dict()

    def log_heavy(self):
        # calculate indexes choose counts
        chosen_indexes = self.cached_data["max_indices"].flatten()
        chosen_indexes = torch.cat(
            (
                chosen_indexes,
                torch.Tensor([self.cached_data["n_tokens"] - 1]).type(
                    chosen_indexes.type()
                ),
            )
        )  # make sure bincount takes into account the whole range of indexes
        indexes_choose_counts = chosen_indexes.bincount()

        # make bar plot of values cached in forward with measure_time
        instr_names = list(self.cached_data["time"].keys())
        instr_times = list(self.cached_data["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)

        return {
            "gradient of gate distribution": make_histogram(self.gate.grad.flatten()),
            "gate_softmax_values": make_histogram(
                self.cached_data["gate_softmax_values"].flatten()
            ),
            "gate_softmax_all_values": make_histogram(
                self.cached_data["gate_softmax_all_values"].flatten()
            ),
            "indexes_choose_counts": make_histogram(indexes_choose_counts),
            "instruction_times": times_fig,
        }
