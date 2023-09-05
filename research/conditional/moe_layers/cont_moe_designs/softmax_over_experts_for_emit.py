import dataclasses
from lizrd.core import misc
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoESoftmaxOverExperts(ContinuousMoeBaseClass):

    """
    merge weights are calculated based on attention keys from previous layer. The representation is averaged over the head dimension.
    """

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S g d, d e -> B S e g", x, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_weights = stable_softmax_temperature(
            merge_logits, self.temperature, dim=-2
        )
        return merge_weights, emit_weights
