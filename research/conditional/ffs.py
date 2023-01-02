import einops
import torch
from lizrd.core import nn
from torch.nn import functional as F

from lizrd.core import misc
from lizrd.core.bert import SplitLastAxis, Transpose, MergeLastAxis
from lizrd.support import ash
from lizrd.support.profile import Timer, TimerLayer
from lizrd.core.misc import EinMix


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
