import torch
from fancy_einsum import einsum

from lizrd.core import nn
from lizrd.core.misc import Linear, get_init_weight


class ExpertChoiceFF(nn.Module):
    def __init__(
        self, dmodel: int, n_experts: int, expert_size: int, cutoff: int, topk: int
    ):
        super().__init__()
        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.width = dmodel * n_experts
        self.topk = topk

        self.lin1 = Linear(dmodel, self.width)
        self.lin2 = Linear(self.width, dmodel)
        self.gate = nn.Parameter(get_init_weight((cutoff, cutoff * n_experts)))

    def forward(self, x: torch.Tensor):
        # x is (batch, cutoff, dmodel)
        # transform cutoff to n_experts
        gate_out = einsum(
            "batch cutoff dmodel, cutoff gate_dim -> batch cutoff dmodel n_experts",
            x,
            self.gate,
        )

        # perform softmax over dmodel (every expert sums to 1)
        gate_out = gate_out.softmax(dim=-2)

        # TODO: potem wystarczy zgodnie z tą macierzą przepermutować x i już normalnie warstwa
        # ale jak uczyć warstwę gatingu? musi być to coś różniczkowalnego
