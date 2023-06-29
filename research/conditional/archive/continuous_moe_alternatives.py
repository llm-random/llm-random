import dataclasses

import torch

from lizrd.core import misc, nn
from lizrd.support import ash
from research.conditional.archive.rogue_code import set_highest_index_one
from research.conditional.moe_layers.continuous_moe import (
    stable_softmax_temperature,
    ContinuousMoeBaseClass,
)


class ContinuousMoEQuickMergeDifferentlySimple(ContinuousMoeBaseClass):
    """
    Emits tokens with separate weights, instead of using the weights from the merging step.
    """

    def init_parameters(self):
        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.dm
            )
        )
        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.expert_size
            )
        )
        self.controller_merge = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )
        self.controller_emit = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_merge)
        self.cache("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.cache("merge_weights", merge_weights)
        emit_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_emit)
        self.cache("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature)
        self.cache("emit_weights", emit_weights)
        return merge_weights, emit_weights


class ContinuousMoEQuickMergeDifferentlyCommonBase(ContinuousMoeBaseClass):
    """
    Both the merge and emit logits are computed as base + merge/emit. The init variance is set so the sum of base + merge/emit is as usual
    """

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_merge + self.controller_base
        )
        self.cache("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.cache("merge_weights", merge_weights)
        emit_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_emit + self.controller_base
        )
        self.cache("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature)
        self.cache("emit_weights", emit_weights)
        return merge_weights, emit_weights

    def init_parameters(self):
        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.dm
            )
        )
        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.expert_size
            )
        )
        self.controller_base = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )
        self.controller_merge = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )
        self.controller_emit = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )


class ContinuousMoEQuickRawmerge(ContinuousMoeBaseClass):
    """
    The rawmerge means that the emitting step is done with weights = 1
    """

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        self.cache("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.cache("merge_weights", merge_weights)
        emit_weights = torch.ones_like(merge_weights)
        self.cache("emit_weights", emit_weights)
        return merge_weights, emit_weights


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickTopmerge(ContinuousMoeBaseClass):
    """
    The emit only sends the output to the token that had the highest logit in the group.
    """

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        self.cache("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.cache("merge_weights", merge_weights)
        emit_weights = set_highest_index_one(merge_weights).to(x.device)
        self.cache("emit_weights", emit_weights)
        return merge_weights, emit_weights


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickNosoftmax(ContinuousMoeBaseClass):
    """
    The merging and emitting is done with just a linear layer, no softmax.
    """

    def get_merge_and_emit_weights(self, x):
        merge_weights = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        emit_weights = merge_weights
        return merge_weights, emit_weights

    def log_heavy(self):
        return {}


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEQuickAdaTemp(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    share_by_experts: bool = True
    share_by_emit_merge: bool = True

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_merge)
        self.cache("merge_logits", merge_logits)
        merge_logits /= self.temperature_merge
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature_merge)
        self.cache("merge_weights", merge_weights)
        emit_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_emit)
        emit_logits /= self.temperature_emit
        self.cache("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature_emit)
        self.cache("emit_weights", emit_weights)
        return merge_weights, emit_weights

    def init_parameters(self):
        if self.share_by_experts:
            if self.share_by_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = nn.Parameter(torch.ones(1))
        else:
            if self.share_by_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts))
                self.temperature_merge = nn.Parameter(torch.ones(self.n_experts))

        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.dm
            )
        )
        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.expert_size
            )
        )
        self.controller_base = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )
        self.controller_merge = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )
        self.controller_emit = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )
