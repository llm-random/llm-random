import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from fancy_einsum import einsum
import plotly.express as px

from lizrd.core import nn
from lizrd.core.misc import get_init_weight
from lizrd.support import ash
from research.conditional.utils.layer_manager import LoggingLayer
from lizrd.support.logging import make_histogram


class ExpertChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        topk_fraction: float,
        random_perm: bool = False,
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
        self.topk_fraction = topk_fraction
        self.random_perm = random_perm

        self.lin1_weight = nn.Parameter(
            get_init_weight((n_experts, dmodel, expert_size), fan_in=dmodel)
        )
        self.lin2_weight = nn.Parameter(
            get_init_weight(
                (n_experts, expert_size, dmodel),
                fan_in=int(n_experts * expert_size * topk_fraction),
            )
        )
        self.gate = nn.Parameter(
            get_init_weight((dmodel, n_experts), fan_in=dmodel)
        ).requires_grad_(True)
        self.ln = LayerNorm(dmodel)

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len = x.shape[0], x.shape[1]
        n_examples = batch_size * seq_len
        topk = round(self.topk_fraction * n_examples)

        # expert embedding
        gate_out = einsum(
            "batch_size seq_len dmodel, dmodel n_experts -> batch_size seq_len n_experts",
            x,
            self.gate,
        )
        # transform such that first dimension corresponds to experts
        gate_out = gate_out.permute(2, 0, 1)
        # flatten batch_size x seq_len
        gate_out = gate_out.flatten(start_dim=1)
        # perform softmax over tokens for each expert
        gate_out = torch.softmax(gate_out, dim=1)
        self.cache("gate_softmax_all_values", gate_out)
        # choose topk tokens for each expert
        topk_values, topk_indices = torch.topk(gate_out, k=topk, dim=1)

        # cache values for logging
        self.cache("gate_softmax_topk_vals", topk_values)
        self.cache("topk_indices", topk_indices)
        self.cache("n_tokens", torch.Tensor([batch_size * seq_len]))

        # Randomly permute tokens for experts if random_perm is True
        # Note this is not total randomness, since topk values are already chosen
        if self.random_perm:
            topk_values = topk_values.flatten()[
                torch.randperm(self.n_experts * topk)
            ].reshape((self.n_experts, topk))

        # flatten x s. t. first dimension is tokens instead of batch_size x seq_len
        x = x.flatten(start_dim=0, end_dim=1)

        # choose the right tokens from x for each expert
        x = torch.index_select(x, dim=0, index=topk_indices.flatten()).reshape(
            (self.n_experts, topk, self.dmodel)
        )
        x = x.reshape((self.n_experts, topk, self.dmodel))

        # feed through ff
        # lin1 maps from (n_experts, topk, dmodel) to (n_experts, topk, exp_size)
        x = einsum(
            "n_exp topk dmodel, n_exp dmodel exp_size -> n_exp topk exp_size",
            x,
            self.lin1_weight,
        )

        x = F.relu(x)

        # lin2 maps from (n_experts, topk, exp_size) to (n_experts, topk, dmodel)
        x = einsum(
            "n_exp topk exp_size, n_exp exp_size dmodel -> n_exp topk dmodel",
            x,
            self.lin2_weight,
        )
        ash.assert_shape("e k m", x, e=self.n_experts, k=topk, m=self.dmodel)

        # multiply by softmax
        ash.assert_shape("e k", topk_values, e=self.n_experts, k=topk)
        x = einsum("n_exp topk dmodel, n_exp topk -> n_exp topk dmodel", x, topk_values)

        # flatten x s. t. first dimension is tokens instead of n_experts x topk
        x = x.flatten(start_dim=0, end_dim=1)

        # add tokens that have been processed by more than one expert
        z = torch.zeros((batch_size * seq_len, self.dmodel)).type(x.type())
        z.index_add_(dim=0, index=topk_indices.flatten().to(int), source=x)

        # reshape to (batch_size, seq_len, dmodel)
        x = z.reshape((batch_size, seq_len, self.dmodel))

        x = self.ln(x)

        return x

    def log_light(self):
        return dict()

    def log_heavy(self):
        # calculate indexes choose counts
        chosen_indexes = self.cached_data["topk_indices"].flatten()
        chosen_indexes = torch.cat(
            (
                chosen_indexes,
                torch.Tensor([self.cached_data["n_tokens"] - 1]).type(
                    chosen_indexes.type()
                ),
            )
        )  # make sure bincount takes into account the whole range of indexes
        indexes_choose_counts = chosen_indexes.bincount()

        return {
            "gradient of gate distribution": make_histogram(self.gate.grad.flatten()),
            "gate_softmax_topk_vals": make_histogram(
                self.cached_data["gate_softmax_topk_vals"].flatten()
            ),
            "gate_softmax_all_values": make_histogram(
                self.cached_data["gate_softmax_all_values"].flatten()
            ),
            "indexes_choose_counts": make_histogram(indexes_choose_counts),
        }
