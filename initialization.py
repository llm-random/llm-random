
from torch import nn

import bert
import misc


class PassThrough(nn.Module):
    def __init__(self, val_mult=1.0, grad_mult=1.0):
        super(PassThrough, self).__init__()
        self.val_mult = val_mult
        self.grad_mult = grad_mult

    def forward(self, x):
        valresult = x.detach() * self.val_mult
        gradresult = (x - x.detach()) * self.grad_mult
        return valresult + gradresult


def FixedLinear(dinput, doutput, relu=False):
    linlayer = nn.Linear(dinput, doutput, bias=False)
    scaling = 2 if relu else 1
    limit = 3 ** 0.5
    nn.init.uniform_(linlayer.weight, -limit, +limit)
    block = nn.Sequential(
        PassThrough(grad_mult=(1.0/doutput)**0.5),
        linlayer,
        nn.ReLU() if relu else misc.Noop(),
        PassThrough(val_mult=(scaling/dinput)**0.5,
                    grad_mult=(scaling)**0.5),
    )
    return block


def StandardLinear(dinput, doutput, relu=False):
    linlayer = nn.Linear(dinput, doutput, bias=False)
    scaling = 6 if relu else 3
    limit = (scaling / dinput) ** 0.5
    nn.init.uniform_(linlayer.weight, -limit, +limit)
    block = nn.Sequential(
        linlayer,
        nn.ReLU() if relu else misc.Noop(),
    )
    return block


# unused
# def FixedReLU():
#     return nn.Sequential(
#         nn.ReLU(),
#         PassThrough(grad_mult=2.0 ** 0.5,
#                     val_mult=2.0 ** 0.5),
#     )


def FixedFeedForward(dmodel, dff):
    return nn.Sequential(
        FixedLinear(dmodel, dff, relu=True),
        FixedLinear(dff, dmodel),
    )


def StandardFeedForward(dmodel, dff):
    return nn.Sequential(
        StandardLinear(dmodel, dff, relu=True),
        StandardLinear(dff, dmodel),
    )


def FixedBERT(max_length, dm, vocab_size, dff, heads, n_blocks, output_size):
    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm),
        bert.TokenEmbedding(vocab_size, dm)
    )

    layer_fun = lambda: FixedLinear(dm, dm, relu=False)

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        (lambda: FixedFeedForward(dm, dff)),
        (lambda: bert.Attention(dm, heads, layer_fun=layer_fun)),
    )

    head = FixedLinear(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)
    return model


def StandardBERT(max_length, dm, vocab_size, dff, heads, n_blocks, output_size):
    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm),
        bert.TokenEmbedding(vocab_size, dm)
    )

    layer_fun = lambda: StandardLinear(dm, dm, relu=False)

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        (lambda: StandardFeedForward(dm, dff)),
        (lambda: bert.Attention(dm, heads, layer_fun=layer_fun)),
    )

    head = StandardLinear(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)
    return model
