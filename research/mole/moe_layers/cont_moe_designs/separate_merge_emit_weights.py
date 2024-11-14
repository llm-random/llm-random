from lizrd.core import misc
import torch.nn as nn
import lizrd.core.initialization
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


class ContinuousMoEMergeDifferentlySimple(ContinuousMoeBaseClass):
    """
    Emits tokens with separate weights, instead of using the weights from the merging step.
    """

    def init_core_parameters(self):
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
        self.controller_merge = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        self.controller_emit = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_merge)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_emit)
        self.update_cache_for_logging("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights
