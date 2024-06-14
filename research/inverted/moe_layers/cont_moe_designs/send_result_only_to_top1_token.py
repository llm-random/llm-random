import torch
from lizrd.core import misc
from research.inverted.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.inverted.utils.misc_tools import stable_softmax_temperature


def set_highest_index_one(tensor: torch.Tensor) -> torch.Tensor:
    # Get the index of the highest value in the last dimension
    _, indices = torch.max(tensor, dim=-1)

    # Create a new tensor filled with zeros, with the same shape as the input tensor
    result_tensor = torch.zeros_like(tensor)

    # Calculate index shape for the new tensor
    result_shape = list(range(tensor.dim() - 1))

    # Set 1 at the index of the highest value in each sub-array
    result_tensor.scatter_(
        -1, indices.view(tensor.shape[:-1] + (1,)).expand(tensor.shape), 1
    )

    return result_tensor


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
