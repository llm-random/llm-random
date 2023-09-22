import dataclasses

import torch

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
        if "contmoe_merge_weights_entropy" not in self.forward_pass_cache:
            self.forward_pass_cache["contmoe_merge_weights_entropy"] = [mean_entropy]
        else:
            self.forward_pass_cache["contmoe_merge_weights_entropy"].append(
                mean_entropy
            )

        return x

    def get_entropy(self, merge_weights, emit_weights):
        device = merge_weights.device
        entropy_merge = entropy(merge_weights).mean().to(device) / torch.log(
            torch.Tensor([self.group_size])
        ).to(device)
        entropy_emit = entropy(emit_weights).mean().to(device) / torch.log(
            torch.Tensor([self.group_size])
        ).to(device)
        self.mean_entropy = 0.5 * (entropy_merge + entropy_emit)
        return self.mean_entropy

    def log_heavy(self):
        log = super().log_heavy()
        log["mean_entropy"] = self.mean_entropy
        return log
