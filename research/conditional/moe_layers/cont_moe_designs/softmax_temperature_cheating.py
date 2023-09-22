import dataclasses
from lizrd.core import misc
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import (
    stable_softmax_temperature,
    straight_through_tensor,
)


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEPassThroughTemp(ContinuousMoeBaseClass):
    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S g d, d e -> B S e g", x, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        forward_merge_weights = stable_softmax_temperature(
            merge_logits, self.temperature
        )
        backward_merge_weights = stable_softmax_temperature(merge_logits, 1.0)
        merge_weights = straight_through_tensor(
            forward_tensor=forward_merge_weights, backward_tensor=backward_merge_weights
        )
        self.update_cache_for_logging("merge_weights", merge_weights)
        return merge_weights, merge_weights
