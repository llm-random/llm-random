import torch
import torch.nn.functional as F
from fancy_einsum import einsum

from lizrd.core import nn
from lizrd.core.misc import Linear, get_init_weight
from lizrd.support import ash


class ExpertChoiceFF(nn.Module):
    def __init__(
        self, dmodel: int, n_experts: int, expert_size: int, cutoff: int, topk: int
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_size: size of each expert
            cutoff: sequence length
            topk: number of tokens that will be chosen for each expert
        """
        super().__init__()

        # make sure that n_experts, topk and expert_size are compatible

        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.width = expert_size * n_experts
        self.topk = topk

        self.lin1_weight = nn.Parameter(
            get_init_weight((dmodel, n_experts, expert_size), fan_in=dmodel)
        )
        self.lin2_weight = nn.Parameter(
            get_init_weight((n_experts, expert_size, dmodel), fan_in=expert_size)
        )
        self.gate = nn.Parameter(
            get_init_weight((cutoff, n_experts), fan_in=dmodel)
        ).requires_grad_(True)

    def forward(self, x: torch.Tensor):
        # x is (batch, cutoff, dmodel)
        batch_size, cutoff = x.shape[0], x.shape[1]
        n_tokens = batch_size * cutoff

        # transform cutoff to n_experts
        # expert embedding
        gate_out = einsum(
            "batch_size cutoff dmodel, cutoff n_experts -> batch_size cutoff n_experts",
            x,
            self.gate,
        )
        # transform such that first dimension corresponds to experts
        gate_out = gate_out.permute(2, 0, 1)
        # flatten batch_size x cutoff
        gate_out = gate_out.flatten(start_dim=1)
        # perform softmax over tokens for each expert
        gate_out = torch.softmax(gate_out, dim=1)
        # choose topk tokens for each expert
        topk_values, topk_indices = torch.topk(gate_out, k=self.topk, dim=1)

        # flatten x s. t. first dimension is tokens instead of batch_size x cutoff
        x = x.flatten(start_dim=0, end_dim=1)

        # save x
        x_before_ff = x

        # choose the right tokens from x for each expert
        x = torch.index_select(x, dim=0, index=topk_indices.flatten()).reshape(
            (self.n_experts, self.topk, self.dmodel)
        )
        ash.assert_shape("e k m", x, e=self.n_experts, k=self.topk, m=self.dmodel)

        # feed through ff
        # lin1 maps from (n_experts, topk, dmodel) to (n_experts, topk, dff/n_experts)
        x = einsum(
            "n_exp topk dmodel, dmodel n_exp exp_size -> n_exp topk exp_size",
            x,
            self.lin1_weight,
        )

        x = F.relu(x)

        # lin2 maps from (...) to (n_experts, topk, dmodel)
        x = einsum(
            "n_exp topk expert_size, n_exp exp_size dmodel -> n_exp topk dmodel",
            x,
            self.lin2_weight,
        )
        ash.assert_shape("e k m", x, e=self.n_experts, k=self.topk, m=self.dmodel)

        # multiply by softmax
        ash.assert_shape("e k", topk_values, e=self.n_experts, k=self.topk)
        x = einsum("n_exp topk dmodel, n_exp topk -> n_exp topk dmodel", x, topk_values)

        # flatten x s. t. first dimension is tokens instead of n_experts x topk
        x = x.flatten(start_dim=0, end_dim=1)

        # add tokens that have been processed by more than one expert
        z = torch.zeros_like(x_before_ff)
        z.index_add_(dim=0, index=topk_indices.flatten().to(int), source=x)

        # reshape to (batch_size, cutoff, dmodel)
        x = z.reshape((batch_size, cutoff, self.dmodel))

        return x
