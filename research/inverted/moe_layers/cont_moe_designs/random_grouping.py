import dataclasses

import einops
from research.inverted.moe_layers.continuous_moe import ContinuousMoeBaseClass

from research.inverted.utils.misc_tools import generate_shuffler_unshuffler


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoERandomGroups(ContinuousMoeBaseClass):
    batch_size: int = -1
    seqlen: int = -1
    mix_whole_batch: bool = False
    different_group_for_every_expert: bool = False

    def reshape_into_token_groups_random(self, x, shuffler):
        # we want to split the input into groups of size self.group_size according to sparsity_dim
        if self.sparsity_dim == 0:
            # gather tokens from the same position in each sequence (mixes data from different examples within a batch)
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            x = shuffler(x)
            # gather tokens from the same sequence (does not mix data from different examples within a batch)
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")
        return x

    def reshape_into_original_random(self, x, unshuffler):
        if self.sparsity_dim == 0:
            out = einops.rearrange(x, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(x, "B S c d -> B (S c) d")
            out = unshuffler(out)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")
        return out

    def forward(self, x):
        assert self.batch_size == x.shape[0]
        assert self.seqlen == x.shape[1]
        shuffler, unshuffler = generate_shuffler_unshuffler(
            self.batch_size, self.seqlen, self.mix_whole_batch
        )
        x = self.reshape_into_token_groups_random(x, shuffler)
        merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        x = self.merge_map_emit(x, merge_weights, emit_weights)
        x = self.reshape_into_original_random(x, unshuffler)
        return x
