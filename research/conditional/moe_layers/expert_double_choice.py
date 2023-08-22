import torch
from torch.nn import LayerNorm

from lizrd.core import nn
from lizrd.core.misc import get_init_weight
from research.conditional.utils.layer_manager import measure_time
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF


class ExpertDoubleChoiceFF(ExpertChoiceFF):
    def __init__(
        self,
        gating_on_start: bool = False,
        second_ln: bool = False,
        nonlinearity_first: bool = False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.second_ln = second_ln
        self.nonlinearity_first = nonlinearity_first
        self.gating_on_start = gating_on_start
        gate_dmodel = self.dmodel if gating_on_start else self.expert_size
        self.gate_2 = nn.Parameter(
            get_init_weight((gate_dmodel, self.n_experts), fan_in=gate_dmodel)
        ).requires_grad_(True)

        if self.second_ln:
            self.ln_2 = LayerNorm(self.expert_size)

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len = x.shape[0], x.shape[1]
        gating_x = x
        topk, topk_indices, topk_values = self.expert_gating(
            x, batch_size, seq_len, self.gate
        )

        x, one_hot = self.extract_chosen_tokens(x, topk, topk_indices, batch_size)
        x = self.feed_linear(x, self.lin1_weight, "ff_1")
        if self.nonlinearity_first:
            x = self.nonlinearity(x)
        x = self.gating_postprocess(
            x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
        )
        if self.second_ln:
            with measure_time(self, "layer_norm_mid"):
                x = self.ln_2(x)

        if not self.nonlinearity_first:
            x = self.nonlinearity(x)

        # second routing
        gating_x = gating_x if self.gating_on_start else x
        topk, topk_indices_2, topk_values_2 = self.expert_gating(
            gating_x, batch_size, seq_len, self.gate_2
        )
        x, one_hot = self.extract_chosen_tokens(x, topk, topk_indices_2, batch_size)
        x = self.feed_linear(x, self.lin2_weight, "ff_2")
        x = self.gating_postprocess(
            x, batch_size, topk, seq_len, topk_values_2, topk_indices_2, one_hot
        )

        with measure_time(self, "layer_norm"):
            x = self.ln(x)

        return x
