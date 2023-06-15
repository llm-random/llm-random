import einops
import torch

from lizrd.core import misc, nn
from lizrd.support import ash
from research.conditional.moe_layers.continuous_moe import (
    stable_softmax_temperature,
    set_highest_index_one,
)
from research.conditional.utils.layer_manager import LoggingLayer


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickMergeDifferentlySimple(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    Emits tokens with separate weights, instead of using the weights from the merging step.
    """

    def __init__(
        self,
        dm,
        dff,
        n_experts,
        group_size,
        sparsity_dim,
        temperature,
        expert_size,
        use_opt_einsum,
    ):
        super().__init__()
        self.dm = dm
        self.dff = dff
        self.n_experts = n_experts
        self.group_size = group_size
        self.sparsity_dim = sparsity_dim
        self.temperature = temperature
        if expert_size is None:
            print(
                f"expert_size is None, setting it to dff // n_experts = {dff // n_experts}"
            )
            expert_size = dff // n_experts
        self.expert_size = expert_size
        self.use_opt_einsum = use_opt_einsum
        self.init_parameters()

    def forward(self, x):
        if self.sparsity_dim == 0:
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        controller_logits_merge = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller_merge,
            use_opt_einsum=self.use_opt_einsum,
        )
        controller_logits_emit = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller_emit,
            use_opt_einsum=self.use_opt_einsum,
        )

        controller_weights_merge = stable_softmax_temperature(
            controller_logits_merge, self.temperature
        )
        controller_weights_emit = stable_softmax_temperature(
            controller_logits_emit, self.temperature
        )

        x = misc.einsum(
            "B S c d, B S e c, d e f -> B S e f",
            x,
            controller_weights_merge,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )

        mid_act = torch.relu_(x)

        out = misc.einsum(
            "B S e f, d e f, B S e c -> B S c d",
            mid_act,
            self.lin2,
            controller_weights_emit,
            use_opt_einsum=self.use_opt_einsum,
        )

        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        return out

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


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickMergeDifferentlyCommonBase(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    Both the merge and emit logits are computed as base + merge/emit.

    """

    def __init__(
        self,
        dm,
        dff,
        n_experts,
        group_size,
        sparsity_dim,
        temperature,
        expert_size,
        use_opt_einsum,
    ):
        """
        1. Groups tokens into groups of fixed size,
        2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
        3. Each expert processes the token constructed above to output a token of size dmodel
        4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.

        :param dm: usual dmodel
        :param dff: usual dff, though it's never explicitly used alone: dff = n_experts * expert_size
        :param n_experts: number of experts
        :param group_size: number of tokens to aggregate into one "token mix"
        :param sparsity_dim: dimension over which to aggregate: 0 for batch, 1 for sequence
        :param temperature: temperature for softmax
        """
        super().__init__()
        self.dm = dm
        self.dff = dff
        self.n_experts = n_experts
        self.group_size = group_size
        self.sparsity_dim = sparsity_dim
        self.temperature = temperature
        if expert_size is None:
            print(
                f"expert_size is None, setting it to dff // n_experts = {dff // n_experts}"
            )
            expert_size = dff // n_experts
        self.expert_size = expert_size
        self.use_opt_einsum = use_opt_einsum
        self.init_parameters()

    def forward(self, x):
        if self.sparsity_dim == 0:
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        controller_logits_common = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller_common,
            use_opt_einsum=self.use_opt_einsum,
        )
        controller_logits_merge = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller_merge,
            use_opt_einsum=self.use_opt_einsum,
        )
        controller_logits_emit = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller_emit,
            use_opt_einsum=self.use_opt_einsum,
        )

        controller_weights_merge = stable_softmax_temperature(
            controller_logits_merge + controller_logits_common, self.temperature
        )
        controller_weights_emit = stable_softmax_temperature(
            controller_logits_emit + controller_logits_common, self.temperature
        )

        x = misc.einsum(
            "B S c d, B S e c, d e f -> B S e f",
            x,
            controller_weights_merge,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )

        mid_act = torch.relu_(x)

        out = misc.einsum(
            "B S e f, d e f, B S e c -> B S c d",
            mid_act,
            self.lin2,
            controller_weights_emit,
            use_opt_einsum=self.use_opt_einsum,
        )

        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        return out

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
        self.controller_common = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )
        self.controller_merge = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )
        self.controller_emit = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickRawmerge(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    The rawmerge means that the emitting step is done with weights = 1
    """

    def __init__(
        self,
        dm,
        dff,
        n_experts,
        group_size,
        sparsity_dim,
        temperature,
        expert_size,
        use_opt_einsum,
    ):
        """
        1. Groups tokens into groups of fixed size,
        2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
        3. Each expert processes the token constructed above to output a token of size dmodel
        4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.

        :param dm: usual dmodel
        :param dff: usual dff, though it's never explicitly used alone: dff = n_experts * expert_size
        :param n_experts: number of experts
        :param group_size: number of tokens to aggregate into one "token mix"
        :param sparsity_dim: dimension over which to aggregate: 0 for batch, 1 for sequence
        :param temperature: temperature for softmax
        """
        super().__init__()
        self.dm = dm
        self.dff = dff
        self.n_experts = n_experts
        self.group_size = group_size
        self.sparsity_dim = sparsity_dim
        self.temperature = temperature
        if expert_size is None:
            print(
                f"expert_size is None, setting it to dff // n_experts = {dff // n_experts}"
            )
            expert_size = dff // n_experts
        self.expert_size = expert_size
        self.use_opt_einsum = use_opt_einsum
        self.init_parameters()

    def forward(self, x):
        if self.sparsity_dim == 0:
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        controller_logits = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller,
            use_opt_einsum=self.use_opt_einsum,
        )

        controller_weights = stable_softmax_temperature(
            controller_logits, self.temperature
        )

        x = misc.einsum(
            "B S c d, B S e c, d e f -> B S e f",
            x,
            controller_weights,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )

        mid_act = torch.relu_(x)

        out = misc.einsum(
            "B S e f, d e f, B S e c -> B S c d",
            mid_act,
            self.lin2,
            controller_weights * 0 + 1,
            use_opt_einsum=self.use_opt_einsum,
        )

        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        return out

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
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickTopmerge(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    The emit only sends the output to the token that had the highest logit in the group.
    """

    def __init__(
        self,
        dm,
        dff,
        n_experts,
        group_size,
        sparsity_dim,
        temperature,
        expert_size,
        use_opt_einsum,
    ):
        """
        1. Groups tokens into groups of fixed size,
        2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
        3. Each expert processes the token constructed above to output a token of size dmodel
        4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.

        :param dm: usual dmodel
        :param dff: usual dff, though it's never explicitly used alone: dff = n_experts * expert_size
        :param n_experts: number of experts
        :param group_size: number of tokens to aggregate into one "token mix"
        :param sparsity_dim: dimension over which to aggregate: 0 for batch, 1 for sequence
        :param temperature: temperature for softmax
        """
        super().__init__()
        self.dm = dm
        self.dff = dff
        self.n_experts = n_experts
        self.group_size = group_size
        self.sparsity_dim = sparsity_dim
        self.temperature = temperature
        if expert_size is None:
            print(
                f"expert_size is None, setting it to dff // n_experts = {dff // n_experts}"
            )
            expert_size = dff // n_experts
        self.expert_size = expert_size
        self.use_opt_einsum = use_opt_einsum
        self.init_parameters()

    def forward(self, x):
        if self.sparsity_dim == 0:
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        controller_logits = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller,
            use_opt_einsum=self.use_opt_einsum,
        )

        merge_weights = stable_softmax_temperature(controller_logits, self.temperature)
        emit_weights = set_highest_index_one(merge_weights).to(x.device)

        x = misc.einsum(
            "B S c d, B S e c, d e f -> B S e f",
            x,
            merge_weights,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )

        mid_act = torch.relu_(x)

        out = misc.einsum(
            "B S e f, d e f, B S e c -> B S c d",
            mid_act,
            self.lin2,
            emit_weights,
            use_opt_einsum=self.use_opt_einsum,
        )

        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        return out

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
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickNosoftmax(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    The emit only sends the output to the token that had the highest logit in the group.
    """

    def __init__(
        self,
        dm,
        dff,
        n_experts,
        group_size,
        sparsity_dim,
        temperature,
        expert_size,
        use_opt_einsum,
    ):
        """
        1. Groups tokens into groups of fixed size,
        2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
        3. Each expert processes the token constructed above to output a token of size dmodel
        4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.

        :param dm: usual dmodel
        :param dff: usual dff, though it's never explicitly used alone: dff = n_experts * expert_size
        :param n_experts: number of experts
        :param group_size: number of tokens to aggregate into one "token mix"
        :param sparsity_dim: dimension over which to aggregate: 0 for batch, 1 for sequence
        :param temperature: temperature for softmax
        """
        super().__init__()
        self.dm = dm
        self.dff = dff
        self.n_experts = n_experts
        self.group_size = group_size
        self.sparsity_dim = sparsity_dim
        self.temperature = temperature
        if expert_size is None:
            print(
                f"expert_size is None, setting it to dff // n_experts = {dff // n_experts}"
            )
            expert_size = dff // n_experts
        self.expert_size = expert_size
        self.use_opt_einsum = use_opt_einsum
        self.init_parameters()

    def forward(self, x):
        if self.sparsity_dim == 0:
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        merge_weights = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller_merge,
            use_opt_einsum=self.use_opt_einsum,
        )
        emit_weights = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller_emit,
            use_opt_einsum=self.use_opt_einsum,
        )

        x = misc.einsum(
            "B S c d, B S e c, d e f -> B S e f",
            x,
            merge_weights,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )

        mid_act = torch.relu_(x)

        out = misc.einsum(
            "B S e f, d e f, B S e c -> B S c d",
            mid_act,
            self.lin2,
            emit_weights,
            use_opt_einsum=self.use_opt_einsum,
        )

        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        return out

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


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuickAdaTemp(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    """

    def __init__(
        self,
        dm,
        dff,
        n_experts,
        group_size,
        sparsity_dim,
        temperature,
        expert_size,
        use_opt_einsum,
    ):
        """
        1. Groups tokens into groups of fixed size,
        2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
        3. Each expert processes the token constructed above to output a token of size dmodel
        4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.

        :param dm: usual dmodel
        :param dff: usual dff, though it's never explicitly used alone: dff = n_experts * expert_size
        :param n_experts: number of experts
        :param group_size: number of tokens to aggregate into one "token mix"
        :param sparsity_dim: dimension over which to aggregate: 0 for batch, 1 for sequence
        :param temperature: temperature for softmax
        """
        super().__init__()
        self.dm = dm
        self.dff = dff
        self.n_experts = n_experts
        self.group_size = group_size
        self.sparsity_dim = sparsity_dim
        self.temperature = temperature
        if expert_size is None:
            print(
                f"expert_size is None, setting it to dff // n_experts = {dff // n_experts}"
            )
            expert_size = dff // n_experts
        self.expert_size = expert_size
        self.use_opt_einsum = use_opt_einsum
        self.init_parameters()

    def forward(self, x):
        if self.sparsity_dim == 0:
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        controller_logits = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller,
            use_opt_einsum=self.use_opt_einsum,
        )

        controller_weights = stable_softmax_temperature(
            controller_logits, self.learnable_temp
        )

        x = misc.einsum(
            "B S c d, B S e c, d e f -> B S e f",
            x,
            controller_weights,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )

        mid_act = torch.relu_(x)

        out = misc.einsum(
            "B S e f, d e f, B S e c -> B S c d",
            mid_act,
            self.lin2,
            controller_weights,
            use_opt_einsum=self.use_opt_einsum,
        )

        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        return out

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
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

        self.learnable_temp = nn.Parameter(torch.ones(1))
