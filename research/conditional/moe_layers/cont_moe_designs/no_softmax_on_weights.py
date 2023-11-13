import torch
from torch import nn
import lizrd.core.initialization
from lizrd.support import ash
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass


@ash.check("... dinp -> ... dout")
class ContinuousMoENosoftmax(ContinuousMoeBaseClass):
    """
    The merging and emitting is done with just a linear layer, no softmax.
    """

    def get_merge_and_emit_weights(self, x):
        merge_weights = torch.einsum("B S c d, d e -> B S e c", x, self.controller)
        emit_weights = merge_weights
        return merge_weights, emit_weights

    def merge_map_emit(self, x, merge_weights, emit_weights):
        x = torch.einsum(
            "B S c d, B S e c-> B S e d",
            x,
            merge_weights,
        )
        x = self.layernorm1(x)
        x = torch.einsum(
            "B S e d, d e f -> B S e f",
            x,
            self.lin1,
        )
        x = torch.relu_(x)
        x = torch.einsum(
            "B S e f, d e f -> B S e d",
            x,
            self.lin2,
        )
        x = self.layernorm2(x)
        x = torch.einsum(
            "B S e d, B S e c -> B S c d",
            x,
            emit_weights,
        )
        return x

    def init_core_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
        self.lin1 = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts, self.expert_size),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

        self.lin2 = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts, self.expert_size),
                fan_in=self.expert_size,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.layernorm1 = torch.nn.LayerNorm(self.dm)
        self.layernorm2 = torch.nn.LayerNorm(self.dm)

    def log_heavy(self):
        return {}
