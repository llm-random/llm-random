import torch

from lizrd.core import misc
from research.mole.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.mole.utils.misc_tools import stable_softmax_temperature


class ContinuousMoERawmerge(ContinuousMoeBaseClass):
    """
    The rawmerge means that the emitting step is done with weights = 1
    """

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_weights = torch.ones_like(merge_weights)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights
