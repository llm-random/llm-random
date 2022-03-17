import torch
import torch.nn as nn

import einops
from einops.layers.torch import Rearrange

from misc import einsum, EinMix, check_layer_funs, Sum
import ash
from profile import TimerLayer, Timer


@ash.check('... d -> ... d')
def FeedForward(dmodel, dff):
    # TODO: replace with Linears
    return TimerLayer('denseFF', nn.Sequential(
        TimerLayer('Linear1', nn.Linear(dmodel, dff), off=True),
        nn.ReLU(inplace=True),
        TimerLayer('Linear2', nn.Linear(dff, dmodel), off=True),
    ))


@ash.check('... d -> ... d')
class BatchSplitFF(nn.Module):
    def __init__(self, register_list, dm, dff, expertsets, nexperts, expertsize):
        super(BatchSplitFF, self).__init__()
        # register_list will be used, together with some get_loss function, to compute loss
        # this will require gradients to be already in place!
        register_list.append(self)

        assert dff == expertsets * nexperts * expertsize
        self.dm = dm
        self.dff = dff
        self.expertsets = expertsets
        self.nexperts = nexperts
        self.expertsize = expertsize

        # assert expertsets == nexperts  # TODO: remove, it shouldn't be necessary

        self.controller = nn.Parameter(torch.Tensor(
            dm, nexperts, expertsets,
        ))
        self.cp = 'd e s'
        self.gp = '... t d'
        self.cout = '... t e s'
        self.inner = '... e s f'

        self.bias = nn.Parameter(torch.Tensor(
            nexperts, expertsets, expertsize
        ))

        self.f1p = 'd e s f'
        self.f1 = nn.Parameter(torch.Tensor(
            dm, nexperts, expertsets, expertsize
        ))

        self.f2p = 'e s f d'
        self.f2 = nn.Parameter(torch.Tensor(
            nexperts, expertsets, expertsize, dm
        ))

        self.inner2 = '... e s d'

        self.ogp = self.gp.replace('...', '... b')

    def forward(self, x):
        # TODO: I want to, if model is in .train(), then also run all things
        # that we need for
        with Timer('batchedFF', disable_inner=False):
            #BATCH, embedding
            ash.assert_shape('... B d', x, d=self.dm)
            # batch, set, embedding <-- this is just reshape
            with Timer('grouped'):
                grouped = einops.rearrange(x, f'... (b t) d -> {self.ogp}',
                                           t=self.nexperts)

            ## CONTROLLER:
            # batch, set1, embedding <-- this is starting point
            with Timer('cont_logits'):
                cont_logits = einsum(f'{self.gp}, {self.cp} -> {self.cout}',
                                     grouped, self.controller)
                # biases in the controller are not needed, because they are added to
                # every token in a given expert, and expert chooses the token with max value


            # batch, set1, set2(experts), expertsets  <--- this comes from linear
            # batch, set1, set2(experts), expertsets <--- sample on 1st dimension (set1)
            # In lieu of adding noise, we can prioritize earlier tokens. This breaks symmetry.

            with Timer('cont_probs'):
                cont_logits += torch.reshape(
                    torch.linspace(start=0, end=1e-6, steps=self.nexperts,
                                   device=x.device),  # to break symmetry
                    (-1, 1, 1),
                )
                # cont_permutation = cont_logits
                cont_permutation = torch.eq(cont_logits, torch.max(cont_logits, dim=-3, keepdim=True)[0])
                cont_permutation = cont_permutation * 1.  # convert to float tensor

            with Timer('Einsum 1'):
                inner = einsum(f'{self.gp}, {self.f1p}, {self.cout} -> {self.inner}',
                               grouped, self.f1, cont_permutation,
                               use_opt_einsum=True)

            with Timer('bias'):
                inner = inner + self.bias

            with Timer('ReLU'):
                inner = torch.relu_(inner)

            with Timer('Einsum 2'):
                intermediate = einsum(f'{self.inner},{self.f2p} -> {self.inner2}',
                                      inner, self.f2)
                result_unpermuted = einsum(f'{self.inner2}, {self.cout} -> {self.gp}',
                                           intermediate, cont_permutation)

            # final reshape
            # BATCH, embedding
            with Timer('rearrange final'):
                result_final = einops.rearrange(result_unpermuted, f'{self.ogp} -> ... (b t) d')
            return result_final


@ash.check('... d -> ... d')
class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out + x


# @ash.check('... dinp -> ... dout')
# def FactoredDense(dinput, doutput, modules):
#     assert doutput % modules == 0
#     dmodule = doutput // modules
#     return nn.Sequential(
#         EinMix('... dinp -> ... modules dinp', weight_shape='modules dinp',# bias_shape='modules dinp',
#                dinp=dinput, modules=modules),
#         EinMix('... modules dinp -> ... (modules dmodule)', weight_shape='dinp dmodule',# bias_shape='modules dmodule',
#                modules=modules, dinp=dinput, dmodule=dmodule),
#     )

@ash.check('... dinp -> ... dout')
class FactoredDense(nn.Module):
    def __init__(self, dinput, doutput, modules):
        super(FactoredDense, self).__init__()
        assert doutput % modules == 0
        dmodule = doutput // modules

        self.gating = nn.Parameter(torch.Tensor(
            modules, dinput,
        ))
        self.projection = nn.Parameter(torch.Tensor(
            dinput, dmodule
        ))

    def forward(self, x):
        y = einsum('... d, m d, d f -> ... m f',
                   x, self.gating, self.projection,
                   use_opt_einsum=True)
        y = einops.rearrange(y, '... modules dmodule -> ... (modules dmodule)')
        return y


@ash.check('... dinp -> ... dinp')
def PermutationDense(dinput):
    sqdi = int(round(dinput**0.5))
    assert sqdi * sqdi == dinput
    return nn.Sequential(
        EinMix('... (a b) -> ... (B a)',
               weight_shape='a b B',
               a=sqdi, b=sqdi, B=sqdi),
        EinMix('... (B a) -> ... (A B)',
               weight_shape='B a A',
               a=sqdi, A=sqdi, B=sqdi),
        # EinMix('... (A B) -> ... (A B)',
        #        weight_shape='A B',
        #        A=sqdi, B=sqdi),
    )


@ash.check('... -> ...')
def NoopDense():
    return nn.Sequential()


@ash.check('... dinp -> ... dout')
def LowRank(dinput, doutput, dlowrank):
    return nn.Sequential(
        nn.Linear(dinput, dlowrank, bias=False),
        nn.Linear(dlowrank, doutput)
    )


@ash.check('... d -> ... d')
class Attention(nn.Module):
    def __init__(self, dmodel, heads, dhead=None, layer_fun=None):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads
        self.heads = heads
        self.dhead = dhead
        self.dmodel = dmodel
        if layer_fun is None:
            layer_fun = lambda: EinMix('... dmodel -> ... (heads dhead)',
                                       weight_shape='dmodel heads dhead', bias_shape='heads dhead',
                                       dmodel=dmodel, heads=heads, dhead=dhead)
        layer_fun_and_reshape = lambda: nn.Sequential(
            layer_fun(),
            Rearrange('... (heads dhead) -> ... heads dhead',
                      heads=heads, dhead=dhead)
        )

        self.Q = layer_fun_and_reshape()
        self.K = layer_fun_and_reshape()
        self.V = layer_fun_and_reshape()

        # self.A = Reduce('... seqlen1 heads dhead, ... seqlen2 heads dhead -> ... heads seqlen1 seqlen2')

        self.D = nn.Sequential(
            Rearrange('... heads dhead -> ... (heads dhead)',
                      heads=heads, dhead=dhead),
            layer_fun()
        )

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        a = torch.einsum('... l h d, ... L h d -> ... h l L',
                         q, k) * (1 / self.dhead ** 0.5)
        a = a * (1 / self.dhead ** 0.5)
        a = torch.softmax(a, dim=-1)
        prefinal = torch.einsum('... h l L, ... L h d -> ... l h d', a, v)
        output = self.D(prefinal)
        return output


@ash.check('... d -> ... d')
def ResidualBlock(dmodel, layer):
    return Residual(nn.Sequential(
        nn.LayerNorm(dmodel),
        layer,
        # nn.LayerNorm(dmodel),
    ))


@ash.check('... d -> ... d')
def EncoderBlock(dmodel, *layers):
    residual_layers = []
    for layer in layers:
        residual_layers.append(ResidualBlock(dmodel, layer))
    return nn.Sequential(*residual_layers)


@ash.check('... d -> ... d')
def EncoderTower(n_blocks, dmodel, *layer_funs):
    check_layer_funs(*layer_funs)
    encoder_blocks = []
    for i_block in range(n_blocks):
        layers = [layer_fun() for layer_fun in layer_funs]
        encoder_blocks.append(EncoderBlock(dmodel, *layers))
    return nn.Sequential(*encoder_blocks)


@ash.check('... -> ... d')
def TokenEmbedding(vocab_size, embedding_dim):
    return nn.Embedding(vocab_size, embedding_dim)


@ash.check('... -> ... d')
class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.layer = nn.Embedding(max_length, embedding_dim)
        # TODO(jaszczur): add initialization as positional encoding

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.layer(positions)
        return embeddings


@ash.check('... -> ... d')
def EmbeddingLayer(*layers):
    return Sum(*layers)


@ash.check('... inp -> ... out')
def PredictionHead(embedding_dim, output_size):
    return nn.Linear(embedding_dim, output_size)


@ash.check('... -> ... out')
def BERT(embedding_layer, encoder_tower, head):
    return nn.Sequential(
        embedding_layer,
        encoder_tower,
        head
    )
