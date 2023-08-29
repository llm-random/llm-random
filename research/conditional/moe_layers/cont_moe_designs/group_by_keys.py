import dataclasses

import torch

from lizrd.core import misc
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEGroupByKeys(ContinuousMoeBaseClass):

    """
    merge weights are calculated based on attention keys from previous layer. The representation is averaged over the head dimension.
    """

    def get_merge_and_emit_weights(self, x):
        keys_from_prev_layer = self.get_from_forward_pass_cache(
            "attention_keys", self.block_number, "attention"
        )
        # shape of keys_from_prev_layer: (batch_size, sequence_length, n_heads, head_size)
        keys = torch.flatten(keys_from_prev_layer, start_dim=-2)
        keys = self.reshape_into_token_groups(keys)
        merge_logits = misc.einsum("B S c d, d e -> B S e c", keys, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_logits = misc.einsum("B S c d, d e -> B S e c", keys, self.controller)
        self.update_cache_for_logging("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights