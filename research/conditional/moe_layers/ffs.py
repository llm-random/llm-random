import einops
import torch
from torch.nn import functional as F

from lizrd.core import misc
from lizrd.core import nn
from lizrd.core.llm import SplitLastAxis, Transpose, MergeLastAxis
from lizrd.core.misc import EinMix
from lizrd.support import ash
from lizrd.support.logging import make_histogram
from lizrd.support.profile import Timer, TimerLayer
from research.conditional.utils.layer_manager import LoggingLayer


@ash.check("... d -> ... d")
class RewrittenSplitFF(nn.Module):
    def __init__(self, register_list, dm, dff, nexperts, sparsity, expertsize):
        super(RewrittenSplitFF, self).__init__()
        register_list.append(self)
        assert dff == nexperts * expertsize
        self.dm = dm
        self.dff = dff
        self.sparsity = sparsity
        self.nexperts = nexperts
        self.expertsize = expertsize

        self.controller = nn.Parameter(misc.get_init_weight((dm, nexperts), fan_in=dm))
        self.f1 = nn.Parameter(
            misc.get_init_weight((dm, nexperts, expertsize), fan_in=dm)
        )
        self.f2 = nn.Parameter(
            misc.get_init_weight(
                (nexperts, expertsize, dm), fan_in=(expertsize * nexperts / sparsity)
            )
        )

        self.f1b = nn.Parameter(misc.get_init_bias((nexperts, expertsize)))
        # self.f2b = nn.Parameter(misc.get_init_bias(
        #     (nexperts, dm)))

        self.cp = "d e"
        self.f1p = "d e s"
        self.f2p = "e s d"
        self.og_batched_act = "... b t d"
        self.batched_act = "... t d"

        self.cout = "... t e"
        self.inner = "... e s"
        self.inner2 = "... e d"

    def forward(self, x):
        # TODO: I want to, if model is in .train(), then also run all things
        # that we need for
        with Timer("rewrittenFF", disable_inner=False):
            # BATCH, embedding
            ash.assert_shape("... B d", x, d=self.dm)
            # batch, set, embedding <-- this is just reshape
            with Timer("grouping", disable=True):
                grouped = einops.rearrange(
                    x, f"... (b t) d -> {self.og_batched_act}", t=self.sparsity
                )

            with Timer("Controller"):
                ## CONTROLLER:
                # batch, set1, embedding <-- this is starting point
                cont_logits = misc.einsum(
                    f"{self.batched_act}, {self.cp} -> {self.cout}",
                    grouped,
                    self.controller,
                )

                # adding bias to controller output is actually dangerous, not only unnecessary
                # cont_logits += self.controller_bias  # This is unnecessary, but it's a reminder

                # biases in the controller are not needed, because they are added to
                # every token in a given expert, and expert chooses the token with max value

                # batch, set1, set2(experts), expertsets  <--- this comes from linear
                # batch, set1, set2(experts), expertsets <--- sample on 1st dimension (set1)
                # In lieu of adding noise, we can prioritize earlier tokens. This breaks symmetry.

                # TODO: add sampling ?
                cont_logits += torch.reshape(
                    torch.linspace(
                        start=0, end=1e-6, steps=self.sparsity, device=x.device
                    ),  # to break symmetry
                    (-1, 1),
                )
                with Timer("contprocessing", disable=True):
                    cont_probs = F.softmax(cont_logits, dim=-2)
                    # cont_permutation = cont_logits
                    cont_permutation = torch.eq(
                        cont_logits, torch.max(cont_logits, dim=-2, keepdim=True)[0]
                    )
                    cont_permutation = cont_permutation * 1.0  # convert to float tensor
                    cont_permutation = (
                        cont_permutation * cont_probs
                    )  # multiply by probability for training!

            with Timer("FF", disable_inner=True):
                with Timer("ff1", disable_inner=True):
                    with Timer("ff1main"):
                        inner = misc.einsum(
                            f"{self.batched_act}, {self.f1p}, {self.cout} -> {self.inner}",
                            grouped,
                            self.f1,
                            cont_permutation,
                            use_opt_einsum=True,
                        )
                    with Timer("ff1b"):
                        inner += self.f1b

                with Timer("relu", disable=True):
                    inner = torch.relu_(inner)

                with Timer("ff2", disable_inner=True):
                    # with Timer('alternative'):
                    #     result_unpermuted = misc.einsum(f'{self.inner},{self.cout},{self.f2p} -> {self.batched_act}',
                    #                                     inner, cont_permutation, self.f2,
                    #                                     use_opt_einsum=True)
                    with Timer("intermediate"):
                        intermediate = misc.einsum(
                            f"{self.inner},{self.f2p} -> {self.inner2}", inner, self.f2
                        )
                    # with Timer('ff2bias'):
                    #     intermediate += self.f2b
                    with Timer("unpermuting"):
                        result_unpermuted = misc.einsum(
                            f"{self.inner2}, {self.cout} -> {self.batched_act}",
                            intermediate,
                            cont_permutation,
                        )

            # final reshape
            # BATCH, embedding
            with Timer("ungrouping", disable=True):
                result_final = einops.rearrange(
                    result_unpermuted, f"{self.og_batched_act} -> ... (b t) d"
                )

            return result_final


@ash.check("... d -> ... d")
class SimpleSplitFF(nn.Module):
    def __init__(
        self,
        register_list,
        dm,
        dff,
        expertsets,
        nexperts,
        expertsize,
        controller_loss_weight=1.0,
    ):
        super(SimpleSplitFF, self).__init__()
        # register_list will be used, together with some get_loss function, to compute loss
        # this will require gradients to be already in place!
        register_list.append(self)

        assert dff == expertsets * nexperts * expertsize
        self.dm = dm
        self.dff = dff
        # self.expertsets = expertsets
        # self.nexperts = nexperts
        sparsity = nexperts
        self.sparsity = sparsity
        totalexperts = expertsets * nexperts
        self.totalexperts = totalexperts
        del expertsets, nexperts
        self.expertsize = expertsize

        # assert expertsets == nexperts  # TODO: remove, it shouldn't be necessary

        self.controller = nn.Parameter(
            misc.get_init_weight((dm, totalexperts), fan_in=dm)
        )
        self.cp = "d e"
        self.gp = "... t d"
        self.cout = "... t e"
        self.inner = "... e f"

        self.bias = nn.Parameter(misc.get_init_bias((totalexperts, expertsize)))

        self.f1p = "d e f"
        self.f1 = nn.Parameter(
            misc.get_init_weight((dm, totalexperts, expertsize), fan_in=dm)
        )

        self.f2p = "e f d"
        self.f2 = nn.Parameter(
            misc.get_init_weight(
                (totalexperts, expertsize, dm),
                fan_in=(expertsize * totalexperts / sparsity),
            )
        )
        # TODO(jaszczur): check if the above is correct regarding fan_in

        self.inner2 = "... e d"

        self.ogp = self.gp.replace("...", "... b")
        self.controller_loss_weight = controller_loss_weight
        # self.controller_bias = nn.Parameter(misc.get_init_bias(
        #     (totalexperts, )))

        self.last_x = None
        self.last_permutation = None

    def forward(self, x):
        self.last_x = x.detach()
        # TODO: I want to, if model is in .train(), then also run all things
        # that we need for
        with Timer("batchedFF", disable_inner=False):
            # BATCH, embedding
            ash.assert_shape("... B d", x, d=self.dm)
            # batch, set, embedding <-- this is just reshape
            grouped = einops.rearrange(x, f"... (b t) d -> {self.ogp}", t=self.sparsity)

            ## CONTROLLER:
            # batch, set1, embedding <-- this is starting point
            cont_logits = misc.einsum(
                f"{self.gp}, {self.cp} -> {self.cout}", grouped, self.controller
            )

            # adding bias to controller output is actually dangerous, not only unnecessary
            # cont_logits += self.controller_bias  # This is unnecessary, but it's a reminder

            # biases in the controller are not needed, because they are added to
            # every token in a given expert, and expert chooses the token with max value

            # batch, set1, set2(experts), expertsets  <--- this comes from linear
            # batch, set1, set2(experts), expertsets <--- sample on 1st dimension (set1)
            # In lieu of adding noise, we can prioritize earlier tokens. This breaks symmetry.

            # TODO: add sampling ?
            cont_logits += torch.reshape(
                torch.linspace(
                    start=0, end=1e-6, steps=self.sparsity, device=x.device
                ),  # to break symmetry
                (-1, 1),
            )
            raise ValueError("the line below is a bug")
            cont_probs = F.softmax(cont_logits, dim=1)
            # cont_permutation = cont_logits
            cont_permutation = torch.eq(
                cont_logits, torch.max(cont_logits, dim=-3, keepdim=True)[0]
            )
            cont_permutation = cont_permutation * 1.0  # convert to float tensor
            cont_permutation = (
                cont_permutation * cont_probs
            )  # multiply by probability for training!
            self.last_permutation = cont_permutation.detach()

            inner = misc.einsum(
                f"{self.gp}, {self.f1p}, {self.cout} -> {self.inner}",
                grouped,
                self.f1,
                cont_permutation,
                use_opt_einsum=True,
            )

            inner = inner + self.bias

            inner = torch.relu_(inner)

            intermediate = misc.einsum(
                f"{self.inner},{self.f2p} -> {self.inner2}", inner, self.f2
            )
            result_unpermuted = misc.einsum(
                f"{self.inner2}, {self.cout} -> {self.gp}",
                intermediate,
                cont_permutation,
            )

            # final reshape
            # BATCH, embedding
            result_final = einops.rearrange(
                result_unpermuted, f"{self.ogp} -> ... (b t) d"
            )

            return result_final


@ash.check("... d -> ... d")
class BatchSplitFF(nn.Module):
    def __init__(
        self,
        register_list,
        dm,
        dff,
        expertsets,
        nexperts,
        expertsize,
        controller_loss_weight=1.0,
    ):
        super(BatchSplitFF, self).__init__()
        # register_list will be used, together with some get_loss function, to compute loss
        # this will require gradients to be already in place!
        register_list.append(self)

        assert dff == expertsets * nexperts * expertsize
        self.dm = dm
        self.dff = dff
        # self.expertsets = expertsets
        # self.nexperts = nexperts
        sparsity = nexperts
        self.sparsity = sparsity
        totalexperts = expertsets * nexperts
        self.totalexperts = totalexperts
        del expertsets, nexperts
        self.expertsize = expertsize

        # assert expertsets == nexperts  # TODO: remove, it shouldn't be necessary

        self.controller = nn.Parameter(
            misc.get_init_weight((dm, totalexperts), fan_in=dm)
        )
        self.cp = "d e"
        self.gp = "... t d"
        self.cout = "... t e"
        self.inner = "... e f"

        self.bias = nn.Parameter(misc.get_init_bias((totalexperts, expertsize)))

        self.f1p = "d e f"
        self.f1 = nn.Parameter(
            misc.get_init_weight((dm, totalexperts, expertsize), fan_in=dm)
        )

        self.f2p = "e f d"
        self.f2 = nn.Parameter(
            misc.get_init_weight(
                (totalexperts, expertsize, dm),
                fan_in=(expertsize * totalexperts / sparsity),
            )
        )
        # TODO(jaszczur): check if the above is correct regarding fan_in

        self.inner2 = "... e d"

        self.ogp = self.gp.replace("...", "... b")
        self.controller_loss_weight = controller_loss_weight
        self.controller_bias = nn.Parameter(misc.get_init_bias((totalexperts,)))

        self.register_full_backward_hook(BatchSplitFF.backward_hook_batch_split_ff)
        self.last_x = None
        self.last_permutation = None

    def backward_hook_batch_split_ff(self, grad_input, grad_output):
        # for now we completely ignore which experts were activated etc.
        x = self.last_x.detach()
        grad_output = grad_output[0].detach()
        del grad_input
        with torch.no_grad():
            combined_x = x.reshape(-1, x.shape[-1])
            combined_gradient = grad_output.reshape(-1, grad_output.shape[-1])

            grouped = einops.rearrange(
                combined_x,
                f"(b e t) d -> b e t d",
                e=self.totalexperts,
                t=self.sparsity,
            )
            # TODO(jaszczur): add permutation here! for x and for gradient

            inner = misc.einsum(
                f"... e t d, {self.f1p}-> ... e t f",
                grouped,
                self.f1,
                use_opt_einsum=True,
            )

            inner = inner + self.bias.view(self.totalexperts, 1, self.expertsize)

            inner = torch.relu_(inner)

            intermediate = misc.einsum(
                f"... e t f, {self.f2p} -> ... e t d", inner, self.f2
            )
            result_unpermuted = intermediate

            gradient_grouped = einops.rearrange(
                combined_gradient,
                f"(b e t) d -> b e t d",
                e=self.totalexperts,
                t=self.sparsity,
            )
            gradient_similarity = misc.einsum(
                f"... e t d, ... e t d-> ... e t", gradient_grouped, result_unpermuted
            )
            # TODO(jaszczur): the line below is unnecessary, but it's a reminder
            gradient_similarity = (
                gradient_similarity / self.dm**0.5
            )  # This is to normalize outputs
            best_choice = (
                gradient_similarity == gradient_similarity.max(dim=-1, keepdim=True)[0]
            ) * 1.0
            ash.assert_shape("... e t", best_choice)
        # # for now we completely ignore which experts were activated etc.
        # x = self.last_x.detach()
        # grad_output = grad_output[0].detach()
        # del grad_input
        # with torch.no_grad():
        #     combined_x = x.reshape(-1, x.shape[-1])
        #     combined_gradient = grad_output.reshape(-1, grad_output.shape[-1])
        #
        #     grouped = einops.rearrange(combined_x, f'(b t) d -> b t d',
        #                                t=self.sparsity)
        #     # TODO(jaszczur): add permutation here! for x and for gradient
        #
        #     inner = misc.einsum(f'... t d, {self.f1p}, ... t e-> ... e f',
        #                         grouped, self.f1, self.last_permutation,
        #                         use_opt_einsum=True)
        #
        #     inner = inner + self.bias.view(self.totalexperts, 1, self.expertsize)
        #
        #     inner = torch.relu_(inner)
        #
        #     intermediate = misc.einsum(f'... e f, {self.f2p} -> ... e d',
        #                                inner, self.f2)
        #     result_unpermuted = misc.einsum('... e d, ... t e -> ... t d',
        #                                     intermediate, self.last_permutation)
        #
        #     gradient_grouped = einops.rearrange(combined_gradient, f'(b t) d -> b t d',
        #                                         e=self.totalexperts, t=self.sparsity)
        #     gradient_similarity = misc.einsum(f'... e t d, ... e t d-> ... e t',
        #                                       gradient_grouped, result_unpermuted)
        #     # TODO(jaszczur): the line below is unnecessary, but it's a reminder
        #     gradient_similarity = gradient_similarity / self.dm ** 0.5  # This is to normalize outputs
        #     best_choice = (gradient_similarity == gradient_similarity.max(dim=-1, keepdim=True)[0]) * 1.0
        #     ash.assert_shape('... e t', best_choice)

        with torch.enable_grad():
            ## CONTROLLER:
            # batch, set1, embedding <-- this is starting point
            cont_logits = misc.einsum(
                f"... e t d, {self.cp} -> ... e t", grouped.detach(), self.controller
            )
            # cont_logits_transposed = cont_logits.transpose(1, -1)
            # best_choice_transposed = best_choice.detach().transpose(1, -1)
            cont_logits += self.controller_bias.view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(cont_logits, best_choice)
            loss *= self.controller_loss_weight
            loss.backward()

        # print(f'loss: {loss.item()}')
        # print(f'grad: {self.controller.grad.norm().item()}')

    def forward(self, x):
        self.last_x = x.detach()
        # TODO: I want to, if model is in .train(), then also run all things
        # that we need for
        with Timer("batchedFF", disable_inner=False):
            # BATCH, embedding
            ash.assert_shape("... B d", x, d=self.dm)
            # batch, set, embedding <-- this is just reshape
            grouped = einops.rearrange(x, f"... (b t) d -> {self.ogp}", t=self.sparsity)

            ## CONTROLLER:
            # batch, set1, embedding <-- this is starting point
            cont_logits = misc.einsum(
                f"{self.gp}, {self.cp} -> {self.cout}", grouped, self.controller
            )
            cont_logits += (
                self.controller_bias
            )  # This is unnecessary, but it's a reminder
            # biases in the controller are not needed, because they are added to
            # every token in a given expert, and expert chooses the token with max value

            # batch, set1, set2(experts), expertsets  <--- this comes from linear
            # batch, set1, set2(experts), expertsets <--- sample on 1st dimension (set1)
            # In lieu of adding noise, we can prioritize earlier tokens. This breaks symmetry.

            cont_logits += torch.reshape(
                torch.linspace(
                    start=0, end=1e-6, steps=self.sparsity, device=x.device
                ),  # to break symmetry
                (-1, 1),
            )
            # cont_permutation = cont_logits
            cont_permutation = torch.eq(
                cont_logits, torch.max(cont_logits, dim=-3, keepdim=True)[0]
            )
            cont_permutation = cont_permutation * 1.0  # convert to float tensor
            self.last_permutation = cont_permutation.detach()

            inner = misc.einsum(
                f"{self.gp}, {self.f1p}, {self.cout} -> {self.inner}",
                grouped,
                self.f1,
                cont_permutation,
                use_opt_einsum=True,
            )

            inner = inner + self.bias

            inner = torch.relu_(inner)

            intermediate = misc.einsum(
                f"{self.inner},{self.f2p} -> {self.inner2}", inner, self.f2
            )
            result_unpermuted = misc.einsum(
                f"{self.inner2}, {self.cout} -> {self.gp}",
                intermediate,
                cont_permutation,
            )

            # final reshape
            # BATCH, embedding
            result_final = einops.rearrange(
                result_unpermuted, f"{self.ogp} -> ... (b t) d"
            )

            return result_final


def stable_softmax_temperature(x, temperature, dim=-1):
    x = x / temperature
    x = x - x.max(dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True)
    return x


@ash.check("... dinp -> ... dout")
class ContinuousMoE(LoggingLayer):
    """
    Continuous mixture of experts. Each expert attends to some subset of the input.
    """

    def __init__(
        self, dm, dff, n_experts, group_size, sparsity_dim, temperature, expertsize=None
    ):
        """
        1. Groups tokens into groups of fixed size,
        2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
        3. Each expert processes the token constructed above to output a token of size dmodel
        4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.

        :param dm: usual dmodel
        :param dff: usual dff, though it's never explicitly used alone: dff = n_experts * expertsize
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
        self.expertsize = dff // n_experts if expertsize is None else expertsize
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

        # print memory usage change
        # print(f"1. post controller logits memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # taken_up_memory = torch.cuda.memory_allocated() / 1024 ** 3

        self.cache("controller_logits", controller_logits)

        ash.assert_shape(
            "B S e c", controller_logits, e=self.n_experts, c=self.group_size
        )
        # apply softmax over "group_size" dimension
        controller_weights = stable_softmax_temperature(
            controller_logits, self.temperature
        )
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
        ash.assert_shape("B S e f", mid_act, e=self.n_experts, f=self.expertsize)

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

        print(
            f"8. post rearrange memory usage: {taken_up_memory} GB -> {torch.cuda.memory_allocated() / 1024 ** 3} GB"
        )

        # assert shape: out is of shape (batch, seq_len, dmodel)
        ash.assert_shape("B S d", out, d=self.dm)
        return out

    def init_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expertsize), fan_in=self.dm
            )
        )

        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expertsize), fan_in=self.expertsize
            )
        )
        # controller: dmodel to n_experts
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

    def log_light(self):
        return {
            "controller_logits_mean": self.cached_data["controller_logits"]
            .mean()
            .item(),
        }

    def log_heavy(self):
        return {
            "controller_logits_distribution": make_histogram(
                self.cached_data["controller_logits"]
            ),
        }


@ash.check("... dinp -> ... dout")
class FactoredDense(nn.Module):
    def __init__(self, dinput, doutput, modules):
        super(FactoredDense, self).__init__()
        assert doutput % modules == 0
        dmodule = doutput // modules

        self.gating = nn.Parameter(misc.get_init_weight((modules, dinput), fan_in=1))
        self.projection = nn.Parameter(
            misc.get_init_weight((dinput, dmodule), fan_in=dinput)
        )
        self.bias = nn.Parameter(misc.get_init_bias(doutput))

    def forward(self, x):
        y = misc.einsum(
            "... d, m d, d f -> ... m f",
            x,
            self.gating,
            self.projection,
            use_opt_einsum=True,
        )
        y = einops.rearrange(y, "... modules dmodule -> ... (modules dmodule)")
        y = y + self.bias
        return y


@ash.check("... dinp -> ... dinp")
def PermutationDense(dinput):
    sqdi = int(round(dinput**0.5))
    assert sqdi * sqdi == dinput

    # wtflayers = []
    # for repeat in range(3):
    #     for variant in ['a c', 'c a', 'b c', 'c b']:
    #         for v2 in ['a b c', 'a c b', 'b a c', 'b c a',
    #                    'c a b', 'c b a']:
    #             layer = TimerLayer(
    #                 f"{variant};{v2}",
    #                 EinMix(f"... a b -> ... {variant}",
    #                        weight_shape=v2,
    #                        a=sqdi, b=sqdi, c=sqdi))
    #             wtflayers.append(layer)
    # random.shuffle(wtflayers)

    return nn.Sequential(
        # nn.Sequential(
        #     SplitLastAxis(sqdi, sqdi),
        #     nn.Sequential(*wtflayers),
        #     MergeLastAxis(),
        # ),
        TimerLayer(
            "verA",
            nn.Sequential(
                SplitLastAxis(sqdi, sqdi),
                misc.EinMix(
                    "... a b -> ... a c", weight_shape="a b c", a=sqdi, b=sqdi, c=sqdi
                ),
                Transpose(),
                misc.EinMix(
                    "... a b -> ... a c", weight_shape="a b c", a=sqdi, b=sqdi, c=sqdi
                ),
                Transpose(),
                misc.EinMix(
                    "... a b -> ... a c", weight_shape="a b c", a=sqdi, b=sqdi, c=sqdi
                ),
                MergeLastAxis(),
            ),
        ),
        # TimerLayer('verB', nn.Sequential(
        #     SplitLastAxis(sqdi, sqdi),
        #     EinMix('... a b -> ... a c',
        #            weight_shape='a b c',
        #            a=sqdi, b=sqdi, c=sqdi),
        #     EinMix('... a b -> ... a c',
        #            weight_shape='a b c',
        #            a=sqdi, b=sqdi, c=sqdi),
        #     EinMix('... a b -> ... a c',
        #            weight_shape='a b c',
        #            a=sqdi, b=sqdi, c=sqdi),
        #     MergeLastAxis(),
        # )),
        # EinSumLayer('...')
        # TimerLayer('verB', nn.Sequential(
        #     EinMix('... (a b) -> ... (a B)',
        #            weight_shape='a b B',
        #            a=sqdi, b=sqdi, B=sqdi),
        #     EinMix('... (a b) -> ... (a B)',
        #            weight_shape='a b B',
        #            a=sqdi, b=sqdi, B=sqdi),
        #     EinMix('... (a b) -> ... (a B)',
        #            weight_shape='a b B',
        #            a=sqdi, b=sqdi, B=sqdi),
        #            )),
        # EinMix('... (b a) -> ... (A b)',
        #        weight_shape='b a A',
        #        a=sqdi, A=sqdi, b=sqdi),
        # EinMix('... (a b) -> ... (B a)',
        #        weight_shape='a b B',
        #        a=sqdi, b=sqdi, B=sqdi),
        # EinMix('... (A B) -> ... (A B)',
        #        weight_shape='A B',
        #        A=sqdi, B=sqdi),
    )


@ash.check("... -> ...")
def NoopDense():
    return nn.Sequential()


@ash.check("... dinp -> ... dout")
def FactoredDense(dinput, doutput, modules):
    assert doutput % modules == 0
    dmodule = doutput // modules
    return nn.Sequential(
        EinMix(
            "... dinp -> ... modules dinp",
            weight_shape="modules dinp",  # bias_shape='modules dinp',
            dinp=dinput,
            modules=modules,
        ),
        EinMix(
            "... modules dinp -> ... (modules dmodule)",
            weight_shape="dinp dmodule",  # bias_shape='modules dmodule',
            modules=modules,
            dinp=dinput,
            dmodule=dmodule,
        ),
    )


class GeneralizedReLU(nn.Module):
    def __init__(self, ndim, bias=False):
        super(GeneralizedReLU, self).__init__()
        assert bias is False  # we assume bias was already added
        self.ndim = ndim
        self.alpha = nn.Parameter(torch.zeros(ndim))
        self.beta = nn.Parameter(torch.ones(ndim))
        self.a = nn.Parameter(torch.zeros(ndim))
        self.b = nn.Parameter(torch.zeros(ndim))

    def forward(self, x):
        above_zero = self.beta * x + self.b
        below_zero = self.alpha * x + self.a
        result = torch.where(x > 0.0, above_zero, below_zero)
        return result
