from lizrd.core import misc
from lizrd.support import ash
from research.conditional.archive.rogue_code import set_highest_index_one
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


@ash.check("... dinp -> ... dout")
class ContinuousMoETopmerge(ContinuousMoeBaseClass):
    """
    The emit only sends the output to the token that had the highest logit in the group.
    """

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_weights = set_highest_index_one(merge_weights).to(x.device)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights
