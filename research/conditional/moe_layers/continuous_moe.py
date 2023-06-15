import einops
import numpy as np
import torch
from plotly import express as px

from lizrd.core import misc, nn
from lizrd.support import ash
from lizrd.support.logging import make_histogram
from research.conditional.utils.layer_manager import LoggingLayer, measure_time


def stable_softmax_temperature(x, temperature, dim=-1):
    x = x / temperature
    x = x - x.max(dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True)
    return x


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-8), dim=-1)


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


class FeedForwardTimed(LoggingLayer):
    def __init__(self, dmodel, dff):
        super().__init__()
        self.dmodel = dmodel
        self.dff = dff
        self.logging_ff_pre_relu = misc.Linear(dmodel, dff)
        self.relu = nn.ReLU(inplace=True)
        self.logging_ff_post_relu = misc.Linear(dff, dmodel)

    def forward(self, x):
        with measure_time(self, "logging_ff_pre_relu"):
            x = self.logging_ff_pre_relu(x)
        with measure_time(self, "relu"):
            x = self.relu(x)
        with measure_time(self, "logging_ff_post_relu"):
            x = self.logging_ff_post_relu(x)
        return x

    def log_heavy(self):
        instr_names = list(self.cached_data["time"].keys())
        instr_times = list(self.cached_data["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)
        out = {"instruction_times_plot": times_fig}
        out.update(self.cached_data["time"])
        return out


@ash.check("... dinp -> ... dout")
class ContinuousMoE(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    """

    def __init__(
        self, dm, dff, n_experts, group_size, sparsity_dim, temperature, expert_size
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
        if expert_size < 0:
            print(
                f"expert_size is None, setting it to dff // n_experts = {dff // n_experts}"
            )
            expert_size = dff // n_experts
        self.expert_size = expert_size
        self.init_parameters()

    def forward(self, x):
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3
        # assert shape: x is of shape (batch, seq_len, dmodel)
        ash.assert_shape("B S d", x, d=self.dm)

        # 1. Groups tokens into groups of fixed size,

        # we want to split the input into groups of size self.group_size according to sparsity_dim
        if self.sparsity_dim == 0:
            # gather tokens from the same position in each sequence (mixes data from different examples within a batch)
            x = einops.rearrange(x, "(B c) S d -> B S c d", c=self.group_size)
        elif self.sparsity_dim == 1:
            # gather tokens from the same sequence (does not mix data from different examples within a batch)
            x = einops.rearrange(x, "B (S c) d -> B S c d", c=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        # - mind that either batch or seqlen has been split into groups, so it's not the same sizes as in the input
        ash.assert_shape("B S c d", x, d=self.dm, c=self.group_size)

        # controller weights hold normalised weights for every group x expert pair
        controller_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        self.cache("controller_logits", controller_logits)

        # print memory usage change
        # print(f"1. post controller logits memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        ash.assert_shape(
            "B S e c", controller_logits, e=self.n_experts, c=self.group_size
        )
        # apply softmax over "group_size" dimension
        controller_weights = stable_softmax_temperature(
            controller_logits, self.temperature
        )
        self.cache("controller_weights", controller_weights)
        # print(f"2. post controller weights memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        # 2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,

        # aggregate x according to controller weights
        # for every group in x, we aggregate the tokens according to the controller weights
        x = torch.einsum("B S c d, B S e c -> B S e d", x, controller_weights)
        ash.assert_shape("B S e d", x, e=self.n_experts, d=self.dm)

        # print(f"3. post aggregation memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        # 3. Each expert processes the token constructed above to output a token of size dmodel

        # lin1 maps from (seq_len, batch, n_experts, dmodel) to (seq_len, batch, n_experts, dff/n_experts)
        mid_act = misc.einsum("B S e d, d e f -> B S e f", x, self.lin1)
        ash.assert_shape("B S e f", mid_act, e=self.n_experts, f=self.expert_size)

        # print(f"4. post lin1 memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        # relu
        mid_act = torch.relu_(mid_act)

        # print(f"5. post relu memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        # lin2 maps from (batch, seqlen, n_experts, dff/n_experts) to (batch, seqlen, n_experts, dmodel)
        out = misc.einsum("B S e f, d e f -> B S e d", mid_act, self.lin2)

        # print(f"6. post lin2 memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        ash.assert_shape("B S e d", out, e=self.n_experts, d=self.dm)

        # 4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.

        # distribute expert outputs according to controller weights
        # (batch, seqlen, n_experts, dmodel) * (batch, seqlen, sparsity, n_experts) -> (batch, seqlen, sparsity, dmodel)
        out = torch.einsum("B S e d, B S e c -> B S c d", out, controller_weights)
        ash.assert_shape("B S c d", out, d=self.dm, c=self.group_size)

        # print(f"7. post distribution memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        # rearrange
        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        # print(
        #     f"8. post rearrange memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB"
        # )

        # assert shape: out is of shape (batch, seq_len, dmodel)
        ash.assert_shape("B S d", out, d=self.dm)
        return out

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
        # controller: dmodel to n_experts
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

    def log_light(self):
        return {}

    def log_heavy(self):
        log = {}
        controller_weights = torch.flatten(
            self.cached_data["controller_weights"], start_dim=0, end_dim=-2
        )
        controller_logits = torch.flatten(
            self.cached_data["controller_logits"], start_dim=0, end_dim=-2
        )
        sample_weight_distros = controller_weights[:5]
        sample_logits_distros = controller_logits[:5]

        for i, sample in enumerate(sample_weight_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"controller_weights/sample_{i}"] = fig

        for i, sample in enumerate(sample_logits_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"controller_logits/sample_{i}"] = fig

        ent = entropy(controller_weights)
        max_entropy = np.log(self.n_experts)
        normalised_ent = ent / max_entropy
        log["controller_weights/normalised_entropy"] = make_histogram(
            normalised_ent, title="controller weights entropy (normalised to [0,1])"
        )

        return log


@ash.check("... dinp -> ... dout")
class ContinuousMoEQuick(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input. The token merging and linear mapping operations are fused into 1 einsum, which is less memory intensive than the original implementation.
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

        controller_logits = misc.einsum(
            "B S c d, d e -> B S e c",
            x,
            self.controller,
            use_opt_einsum=self.use_opt_einsum,
        )
        ash.assert_shape(
            "B S e c", controller_logits, e=self.n_experts, c=self.group_size
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
            controller_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        if self.sparsity_dim == 0:
            out = einops.rearrange(out, "B S c d -> (B c) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(out, "B S c d -> B (S c) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")

        ash.assert_shape("B S d", out, d=self.dm)
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

    def log_light(self):
        return {}

    def log_heavy(self):
        log = {}
        controller_weights = torch.flatten(
            self.cached_data["controller_weights"], start_dim=0, end_dim=-2
        )
        controller_logits = torch.flatten(
            self.cached_data["controller_logits"], start_dim=0, end_dim=-2
        )
        sample_weight_distros = controller_weights[:5]
        sample_logits_distros = controller_logits[:5]

        for i, sample in enumerate(sample_weight_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"controller_weights/sample_{i}"] = fig

        for i, sample in enumerate(sample_logits_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"controller_logits/sample_{i}"] = fig

        ent = entropy(controller_weights)
        max_entropy = np.log(self.n_experts)
        normalised_ent = ent / max_entropy
        log["controller_weights/normalised_entropy"] = make_histogram(
            normalised_ent, title="controller weights entropy (normalised to [0,1])"
        )

        return log
