import torch
import lizrd.core.nn as nn

from einops.layers.torch import Rearrange

from lizrd.core import misc
from lizrd.support import ash
from lizrd.support.profile import TimerLayer


@ash.check('... d -> ... d')
def FeedForward(dmodel, dff):
    return TimerLayer('denseFF', nn.Sequential(
        TimerLayer('Linear1', misc.Linear(dmodel, dff), off=True),
        nn.ReLU(inplace=True),
        TimerLayer('Linear2', misc.Linear(dff, dmodel), off=True),
    ))


@ash.check('... d -> ... d')
class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out + x


@ash.check('... dinp -> ... a b')
class SplitLastAxis(nn.Module):
    def __init__(self, a, b):
        super(SplitLastAxis, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        a, b = self.a, self.b
        assert x.shape[-1] == a * b
        result = x.view(x.shape[:-1] + (a, b))
        assert result.shape[-2:] == (a, b)
        # print("wtf", x.shape, result.shape)
        return result


@ash.check('... a b -> ... dout')
class MergeLastAxis(nn.Module):
    def forward(self, x):
        result = x.reshape(x.shape[:-2] + (-1,))
        # print('wtf', x.shape, result.shape)
        return result


@ash.check('... a b -> ... b a')
class Transpose(nn.Module):
    def forward(self, x):
        # return einops.rearrange(x, '... a b -> ... b a')
        return torch.transpose(x, -1, -2)


@ash.check('... dinp -> ... dout')
def LowRank(dinput, doutput, dlowrank):
    return nn.Sequential(
        misc.Linear(dinput, dlowrank, bias=False),
        misc.Linear(dlowrank, doutput)
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
            layer_fun = lambda: misc.EinMix('... dmodel -> ... (heads dhead)',
                                            weight_shape='dmodel heads dhead', bias_shape='heads dhead',
                                            dmodel=dmodel, heads=heads, dhead=dhead)
        layer_fun_and_reshape = lambda: nn.Sequential(
            TimerLayer('QKVproj', layer_fun()),
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
                         q, k)
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
    misc.check_layer_funs(*layer_funs)
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
    return misc.Sum(*layers)


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
