import dataclasses
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import entropy


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEEntropyLoss(ContinuousMoeBaseClass):
    def forward(self, x):
        x = self.reshape_into_token_groups(x)
        merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        x = self.merge_map_emit(x, merge_weights, emit_weights)
        x = self.reshape_into_original(x)
        mean_entropy = self.get_entropy(merge_weights, emit_weights)
        self.update_forward_pass_cache("mean_entropy", mean_entropy)
        return x

    def get_entropy(self, merge_weights, emit_weights):
        entropy_merge = entropy(merge_weights).mean()
        entropy_emit = entropy(emit_weights).mean()
        mean_entropy = 0.5 * (entropy_merge + entropy_emit)
        return mean_entropy
