from lizrd.core import misc
import torch.nn as nn
from lizrd.core.initialization import get_init_weight
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


class ContinuousMoEMergeDifferentlyCommonBase(ContinuousMoeBaseClass):
    """
    Both the merge and emit logits are computed as base + merge/emit. The init variance is set so the sum of base + merge/emit is as usual
    """

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_merge + self.controller_base
        )
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_emit + self.controller_base
        )
        self.update_cache_for_logging("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights

    def init_core_parameters(self):
        self.lin1 = nn.Parameter(
            get_init_weight(
                (self.dm, self.n_experts, self.expert_size),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.lin2 = nn.Parameter(
            get_init_weight(
                (self.dm, self.n_experts, self.expert_size),
                fan_in=self.expert_size,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.controller_base = nn.Parameter(
            get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm * 2,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.controller_merge = nn.Parameter(
            get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm * 2,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.controller_emit = nn.Parameter(
            get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm * 2,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
