import dataclasses

import numpy as np
import torch
from plotly import express as px
import einops
from lizrd.core import misc, nn
from lizrd.support import ash
from lizrd.support.logging import make_histogram
from research.conditional.archive.rogue_code import set_highest_index_one, entropy
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

    def merge_map_emit(self, x, merge_weights, emit_weights):
        x = misc.einsum(
            "B S c d, B S e c-> B S e d",
            x,
            merge_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = self.layernorm1(x)
        x = misc.einsum(
            "B S e d, d e f -> B S e f",
            x,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = torch.relu_(x)
        x = misc.einsum(
            "B S e f, d e f -> B S e d",
            x,
            self.lin2,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = self.layernorm2(x)
        x = misc.einsum(
            "B S e d, B S e c -> B S c d",
            x,
            emit_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        return x

    def init_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
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
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )
        self.layernorm1 = nn.LayerNorm(self.dm)
        self.layernorm2 = nn.LayerNorm(self.dm)

    def log_heavy(self):
        return {}


@ash.check("... dinp -> ... dout")
class ContinuousMoELayernorm(ContinuousMoeBaseClass):
    def merge_map_emit(self, x, merge_weights, emit_weights):
        x = misc.einsum(
            "B S c d, B S e c-> B S e d",
            x,
            merge_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = self.layernorm1(x)
        x = misc.einsum(
            "B S e d, d e f -> B S e f",
            x,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = torch.relu_(x)
        x = misc.einsum(
            "B S e f, d e f -> B S e d",
            x,
            self.lin2,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = self.layernorm2(x)
        x = misc.einsum(
            "B S e d, B S e c -> B S c d",
            x,
            emit_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        return x

    def init_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
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
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )
        self.layernorm1 = nn.LayerNorm(self.dm)
        self.layernorm2 = nn.LayerNorm(self.dm)

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
        emit_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_merge)
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
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = nn.Parameter(torch.ones(self.n_experts, 1))

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

    def log_heavy(self):
        log = {}
        if self.group_size == 1:
            return log
        merge_weights = torch.flatten(
            self.cached_data["merge_weights"], start_dim=0, end_dim=-2
        )
        merge_logits = torch.flatten(
            self.cached_data["merge_logits"], start_dim=0, end_dim=-2
        )
        sample_weight_distros = merge_weights[:5]
        sample_logits_distros = merge_logits[:5]

        for i, sample in enumerate(sample_weight_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"merge_weights/sample_{i}"] = fig

        for i, sample in enumerate(sample_logits_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"merge_logits/sample_{i}"] = fig

        ent = entropy(merge_weights)
        max_entropy = np.log(self.n_experts)
        normalised_ent = ent / max_entropy
        log["merge_weights/normalised_entropy"] = make_histogram(
            normalised_ent, title="merge logits entropy (normalised to [0,1])"
        )

        log[
            "merge_weights/merge_temperature"
        ] = self.temperature_merge.data.flatten().tolist()
        log[
            "merge_weights/emit_temperature"
        ] = self.temperature_emit.data.flatten().tolist()

        return log


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEFinal(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    share_by_experts: bool = True
    share_by_emit_merge: bool = True

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_merge + self.controller_base
        )
        self.cache("merge_logits", merge_logits)
        merge_logits /= self.temperature_merge
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature_merge)
        self.cache("merge_weights", merge_weights)
        emit_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_emit + self.controller_base
        )
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
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = nn.Parameter(torch.ones(self.n_experts, 1))

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


def generate_shuffler_unshuffler():
    def shuffle_3d_tensor(tensor):
        shuffled_indices = torch.randperm(tensor.shape[1])
        shuffled_tensor = tensor[:, shuffled_indices, :]
        return shuffled_tensor, shuffled_indices

    def unshuffle_3d_tensor(shuffled_tensor, shuffled_indices):
        return shuffled_tensor[:, shuffled_indices.argsort(), :]

    return shuffle_3d_tensor, unshuffle_3d_tensor


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoERandomGroups(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    mix_whole_batch: bool = False
    different_group_for_every_expert: bool = True

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
        shuffler, unshuffler = generate_shuffler_unshuffler()
        x = self.reshape_into_token_groups_random(x, shuffler)
        merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        x = self.merge_map_emit(x, merge_weights, emit_weights)
        x = self.reshape_into_original_random(x, unshuffler)
        return x
