import torch
import torch.nn.functional as F
from fancy_einsum import einsum

from lizrd.core import nn
from lizrd.core.misc import Linear, get_init_weight


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

        self.lin1 = Linear(dmodel, self.width)
        self.lin2 = Linear(self.width, dmodel)
        self.gate = nn.Parameter(get_init_weight((cutoff, n_experts), fan_in=dmodel)).requires_grad_(
            True
        )

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
        # gate_out = torch.softmax(gate_out, dim=1)
        softmax_gate_out = gate_out
        # choose topk tokens for each expert
        gate_out = torch.topk(gate_out, k=self.topk, dim=1).indices
        gate_out = F.one_hot(gate_out, num_classes=n_tokens)
        # create permutation matrix
        fake_mask = einsum(
            "n_experts topk n_tokens, n_experts n_tokens -> n_experts topk n_tokens",
            gate_out,
            softmax_gate_out,
        )
        perm = gate_out.float() #+ fake_mask - fake_mask.detach()

        # flatten x s. t. first dimension is tokens instead of batch_size x cutoff
        # x = x.flatten(start_dim=0, end_dim=1)

        # save x
        x_before_ff = x
        perm_2 = torch.randperm(x.shape[0])
        # x = x[perm_2]

        # permute tokens according to permutation matrix
        # x = einsum(
        #     "n_elems dmodel, n_experts topk n_elems -> n_experts topk dmodel", x, perm
        # )
        # flatten dim of expert and topk to get input to ff
        # x = x.flatten(start_dim=0, end_dim=1)

        # feed through ff
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        # ------------------ HERE SHOULD BE (FROM Simon) ------------------
        # #lin1 maps from (seq_len, batch, n_experts, dmodel) to (seq_len, batch, n_experts, dff/n_experts)
        # mid_act = misc.einsum("B S e d, d e f -> B S e f", x, self.lin1)
        # ash.assert_shape("B S e f", mid_act, e=self.n_experts, f=self.expertsize)

        # # relu
        # mid_act = torch.relu_(mid_act)

        # # lin2 maps from (batch, seqlen, n_experts, dff/n_experts) to (batch, seqlen, n_experts, dmodel)
        # out = misc.einsum("B S e f, d e f -> B S e d", mid_act, self.lin2)
        # ash.assert_shape("B S e d", out, e=self.n_experts, d=self.dm)
        # ----------------- END SHOULD BE ---------------------

        # add tokens that have been processed by more than one expert
        perm = perm.flatten(start_dim=0, end_dim=1)
        id = torch.eye(n_tokens, device=x.device)
        chosen_examples = einsum(
            "n_elems n_elems, expert_layer_width n_elems -> expert_layer_width n_elems",
            id,
            perm,
        )
        # x = einsum(
        #     "expert_layer_width n_elems, expert_layer_width dmodel -> n_elems dmodel",
        #     chosen_examples,
        #     x,
        # )
        # add tokens that have not been processed
        # not_chosen_examples = (chosen_examples.sum(dim=0) == 0).float()
        # x += einsum(
        #     "n_elems dmodel, n_elems -> n_elems dmodel",
        #     x_before_ff,
        #     not_chosen_examples,
        # )
        # inv = torch.empty_like(perm_2)
        # inv[perm_2] = torch.arange(perm_2.size(0), device=perm_2.device)
        # x = x[inv]

        # again expand token dimension to batch_size x cutoff
        # x = x.reshape(batch_size, cutoff, self.dmodel)

        return x
