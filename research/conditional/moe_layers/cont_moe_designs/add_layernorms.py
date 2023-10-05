import torch

from lizrd.core import misc, nn
import lizrd.core.init
from lizrd.support import ash
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass


@ash.check("... dinp -> ... dout")
class ContinuousMoELayernorm(ContinuousMoeBaseClass):
    def merge_map_emit(self, x, merge_weights, emit_weights):
        x = misc.einsum(
            "B S c d, B S e c-> B S e d",
            x,
            merge_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = self.layernorm1(x)
        x = misc.einsum(
            "B S e d, d e f -> B S e f",
            x,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = torch.relu_(x)
        x = misc.einsum(
            "B S e f, d e f -> B S e d",
            x,
            self.lin2,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = self.layernorm2(x)
        x = misc.einsum(
            "B S e d, B S e c -> B S c d",
            x,
            emit_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        return x

    def init_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
        self.lin1 = nn.Parameter(
            lizrd.core.init.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.dm
            )
        )

        self.lin2 = nn.Parameter(
            lizrd.core.init.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.expert_size
            )
        )
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            lizrd.core.init.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )
        self.layernorm1 = nn.LayerNorm(self.dm)
        self.layernorm2 = nn.LayerNorm(self.dm)

    def log_heavy(self):
        return {}
