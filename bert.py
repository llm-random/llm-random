import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from time import sleep
from opt_einsum import contract
import opt_einsum
from einops.layers.torch import Rearrange, Reduce
#https://einops.rocks/3-einmix-layer/

from einops.layers.torch import EinMix as OGEinMix
import einops


def einsum(subscript, *operands, **kwargs):
    return torch.einsum(subscript, *operands, **kwargs)
    # return contract(subscript, *operands, **kwargs)

import gc
from typing import List


class EinMix(nn.Module):
    def __init__(self, signature, weight_shape=None, bias_shape=None, **kwargs):
        super(EinMix, self).__init__()
        self.change_anything = False
        if '...' in signature:
            self.change_anything = True
            self.og_signature = signature
            signature = signature.replace('...', 'squeezed')
        self.layer = OGEinMix(signature, weight_shape=weight_shape, bias_shape=bias_shape, **kwargs)

    def forward(self, x):
        if not self.change_anything:
            return self.layer(x)
        # else
        beginning, end = self.og_signature.split('->')
        beginning = beginning.split()
        end = end.split()
        assert beginning[0] == end[0] == '...'
        contracted_dims = len(x.shape)-len(beginning)+1
        ellipsis_shape = list(x.shape[:contracted_dims])
        newx = torch.reshape(x, [-1] + list(x.shape[contracted_dims:]))
        output = self.layer(newx)
        newoutput = torch.reshape(output, ellipsis_shape + list(output.shape[1:]))
        return newoutput


def Dense(dinp, dout):
    return EinMix('... dinp -> ... dout',
                  weight_shape='dinp dout', bias_shape='dout',
                  dinp=dinp, dout=dout)


def FeedForward(dmodel, dff):
    # TODO: replace with Linears
    return TimerLayer('denseFF', nn.Sequential(
        nn.Linear(dmodel, dff),
        # EinMix('... dm -> ... dff',
        #        weight_shape='dm dff', bias_shape='dff',
        #        dm=dmodel, dff=dff),
        nn.ReLU(inplace=True),
        nn.Linear(dff, dmodel),
        # EinMix('... dff -> ... dm',
        #        weight_shape='dff dm', bias_shape='dm',
        #        dm=dmodel, dff=dff)
    ))


# class BatchSplitFF(nn.Module):
#     def __init__(self, register_list, dm, dff, expertsets, nexperts, expertsize):
#         super(BatchSplitFF, self).__init__()
#         # register_list will be used, together with some get_loss function, to compute loss
#         # this will require gradients to be already in place!
#         register_list.append(self)
#
#         assert dff == expertsets * nexperts * expertsize
#         self.dm = dm
#         self.dff = dff
#         self.expertsets = expertsets
#         self.nexperts = nexperts
#         self.expertsize = expertsize
#
#         # assert expertsets == nexperts  # TODO: remove, it shouldn't be necessary
#
#         self.contr = EinMix('... tokens d -> ... tokens experts sets',
#                             weight_shape='d experts sets', bias_shape='experts sets',
#                             d=dm, tokens=self.nexperts, experts=self.nexperts, sets=self.expertsets)
#
#         self.f1 = EinMix('... experts sets d -> ... experts sets expertsize',
#                          weight_shape='experts sets d expertsize',
#                          bias_shape='experts sets expertsize',
#                          d=dm, experts=self.nexperts, sets=self.expertsets,
#                          expertsize=self.expertsize)
#         self.f2 = EinMix('... experts sets expertsize -> ... experts sets d',
#                          weight_shape='experts sets expertsize d',
#                          bias_shape='experts sets d',
#                          d=dm, experts=self.nexperts, sets=self.expertsets,
#                          expertsize=self.expertsize)
#
#     def forward(self, x):
#         #BATCH, embedding
#         assert len(x.shape) >= 2
#         assert x.shape[-1] == self.dm
#         #batch, set, embedding <-- this is just reshape
#         grouped = einops.rearrange(x, '... (b g) d -> ... b g d',
#                                    g=self.nexperts)
#
#         ## CONTROLLER:
#         # batch, set1, embedding <-- this is starting point
#         cont_logits = self.contr(grouped)
#         # batch, set1, set2(experts), expertsets  <--- this comes from linear
#         # batch, set1, set2(experts), expertsets <--- sample on 1st dimension (set1)
#         # In lieu of adding noise, we can prioritize earlier tokens. This breaks symmetry.
#         cont_logits += torch.reshape(
#             torch.linspace(start=0, end=1e-6, steps=self.nexperts,
#                            device=x.device),  # to break symmetry
#             (-1, 1, 1),
#         )
#         cont_permutation = torch.eq(cont_logits, torch.max(cont_logits, dim=-3, keepdim=True)[0])
#         cont_permutation = cont_permutation * 1.  # convert to float tensor
#
#         # transformation with controller
#         # batch, set2(experts), expertsets, embedding
#         permuted = torch.einsum(
#             '... t d, ... t e s -> ... e s d',
#             grouped, cont_permutation
#         )
#
#         # f1 weight: set2(experts), expertsets, embedding, expertsize
#         inner = self.f1(permuted)
#         inner = torch.relu(inner)
#         # batch, set2(experts), expertsets, expertsize
#         # ReLU
#         # batch, set2(experts), expertsets, expertsize
#         result_permuted = self.f2(inner)
#         # f2 weight: set2(experts), expertsets, expertsize, embedding
#         # batch, set2(experts), expertsets, embedding
#
#         # back from the controller, transformation
#         # batch, set1, embedding
#         result_unpermuted = torch.einsum(
#             '... e s d, ... t e s -> ... t d',
#             result_permuted, cont_permutation
#         )
#
#         # final reshape
#         # BATCH, embedding
#         result_final = einops.rearrange(result_unpermuted, '... b g d -> ... (b g) d')
#         return result_final


GLOBAL_TIMERS = dict()
GLOBAL_NAMES = []
GLOBAL_DEPTHS = dict()
CURRENT_DEPTH = [0]
DISABLED = False


class TimerLayer(nn.Module):
    def __init__(self, name, layer):
        super(TimerLayer, self).__init__()
        self.name = name
        self.layer = layer

    def forward(self, *args, **kwargs):
        with Timer(self.name):
            result = self.layer(*args, **kwargs)
        return result


class Timer(object):
    def __init__(self, name, disable_inner=False):
        global DISABLED
        self.name = name
        self.disable_inner = disable_inner
        self.i_disabled = False
        if not DISABLED:
            if name not in GLOBAL_TIMERS:
                GLOBAL_TIMERS[self.name] = []
                # GLOBAL_TIMERS[self.name+'T'] = []
            if self.name not in GLOBAL_NAMES:
                GLOBAL_NAMES.append(name)
        # self.start = torch.cuda.Event(enable_timing=True)
        # self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        global DISABLED
        if not DISABLED:
            if self.disable_inner:
                DISABLED = True
                self.i_disabled = True
            torch.cuda.synchronize()
            GLOBAL_DEPTHS[self.name] = CURRENT_DEPTH[0]
            CURRENT_DEPTH[0] += 1
            self.start_time = time.time()
            # self.start.record()

    def __exit__(self, *args):
        global DISABLED
        if not DISABLED or self.i_disabled:
            if self.disable_inner:
                DISABLED = False
                self.i_disabled = False
            # self.end.record()
            torch.cuda.synchronize()
            self.end_time = time.time()

            CURRENT_DEPTH[0] -= 1
            GLOBAL_TIMERS[self.name].append(self.end_time - self.start_time)
            # GLOBAL_TIMERS[self.name+'T'].append(self.start.elapsed_time(self.end)/1000)


def reset_times():
    global GLOBAL_TIMERS, GLOBAL_DEPTHS, GLOBAL_NAMES
    GLOBAL_TIMERS = dict()
    GLOBAL_DEPTHS = dict()
    GLOBAL_NAMES = []


def print_times(reset=True):
    for name in GLOBAL_NAMES:
        values = GLOBAL_TIMERS[name]
        depth = GLOBAL_DEPTHS[name]
        print(f'{" "*depth + name:18}: {round(sum(values), 3)} +/- {round(np.std(values)*len(values)**0.5, 2)}')
    print('\n\n\n')
    if reset:
        reset_times()


class AltBatchSplitFF(nn.Module):
    def __init__(self, register_list, dm, dff, expertsets, nexperts, expertsize):
        super(AltBatchSplitFF, self).__init__()
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

        self.f1p = 'd e s f'
        self.f1 = nn.Parameter(torch.Tensor(
            dm, nexperts, expertsets, expertsize
        ))

        self.f1Q = nn.Parameter(torch.Tensor(
            dm, nexperts*expertsets*expertsize
        ))

        self.f2p = 'e s f d'
        self.f2 = nn.Parameter(torch.Tensor(
            nexperts, expertsets, expertsize, dm
        ))

        self.ogp = self.gp.replace('...', '... b')

    def forward(self, x):
        with Timer('batchedFF'):
            #BATCH, embedding
            assert len(x.shape) >= 2
            assert x.shape[-1] == self.dm
            # batch, set, embedding <-- this is just reshape
            with Timer('grouped'):
                grouped = einops.rearrange(x, '... (b t) d -> {}'.format(self.ogp),
                                           t=self.nexperts)

            ## CONTROLLER:
            # batch, set1, embedding <-- this is starting point
            with Timer('cont_logits'):
                cont_logits = einsum('{}, {} -> {}'.format(self.gp, self.cp, self.cout),
                                     grouped, self.controller)


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
                # inner = einsum('... t d, ... t e s, e s f d -> ... e s f',
                #                grouped, cont_permutation, self.f2)

                with Timer('Einsum 1A-1'):
                    intermediate = einsum('... t d, d e s f -> ... t e s f',
                                          grouped, self.f1)
                with Timer('Einsum 1B'):
                    inner = einsum('... t e s f, ... t e s -> ... e s f',
                                   intermediate, cont_permutation)

                with Timer('Einsum 1A-T'):
                    intermediate = einsum('... t d, d E -> ... t E',
                                          grouped, self.f1Q)

                # with Timer('Einsum Q1A'):
                #     intermediate = einsum('... t d, ... t e s -> ... t e s d',
                #                           grouped, cont_permutation)
                # with Timer('Einsum Q1B'):
                #     inner = einsum('... t e s d, d e s f -> ... e s f',
                #                    intermediate, self.f1)
                #
                # with Timer('Einsum Q1A-2'):
                #     intermediate = einsum('... t d, ... t e s -> ... t d e s',
                #                           grouped, cont_permutation)
                # with Timer('Einsum Q1B'):
                #     inner = einsum('... t e s d, d e s f -> ... e s f',
                #                    intermediate, self.f1)

                # inner = einsum('{}, {}, {} -> {}'.format(self.gp, self.cout, self.f1p, self.inner),
                #                grouped, cont_permutation, self.f1)

            with Timer('ReLU'):
                inner = torch.relu_(inner)

            with Timer('Einsum 2'):
                # result_unpermuted = einsum(
                #     '{}, {} -> {}'.format(self.cout, self.f2p, self.gp),
                #     cont_permutation, self.f2
                # ) * torch.mean(inner)

                # intermediate = einsum('... e s f, ... t e s -> ... e s f t',
                #                       inner, cont_permutation)
                # result_unpermuted = einsum('... e s f t, e s f d -> ... t d',
                #                            intermediate, self.f2)

                with Timer('Einsum 2A'):
                    intermediate = einsum('... e s f, e s f d -> ... e s d',
                                          inner, self.f2)
                with Timer('Einsum 2B'):
                    result_unpermuted = einsum('... e s d, ... t e s -> ... t d',
                                               intermediate, cont_permutation)


                # intermediate = einsum('... t e s, e s f d -> ... e s f t d',
                #                       cont_permutation, self.f2)
                # result_unpermuted = einsum('... e s f t d, ... e s f -> ... t d',
                #                            intermediate, inner)


                # result_unpermuted = einsum('... e s f, ... t e s, e s f d -> ... t d',
                #                       inner, cont_permutation, self.f2)

                # result_unpermuted = einsum(
                #     '{}, {}, {} -> {}'.format(self.inner, self.cout, self.f2p, self.gp),
                #     inner, cont_permutation, self.f2
                # )

            # final reshape
            # BATCH, embedding
            with Timer('rearrange final'):
                result_final = einops.rearrange(result_unpermuted, '{} -> ... (b t) d'.format(self.ogp))
            return result_final


class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out + x


class Attention(nn.Module):
    def __init__(self, dmodel, heads, dhead=None):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads
        self.heads = heads
        self.dhead = dhead
        self.dmodel = dmodel
        layer_fun = lambda: EinMix('... dm -> ... heads dhead',
                                   weight_shape='dm heads dhead', bias_shape='heads dhead',
                                   dm=dmodel, heads=heads, dhead=dhead)
        self.Q = layer_fun()
        self.K = layer_fun()
        self.V = layer_fun()

        # self.A = Reduce('... seqlen1 heads dhead, ... seqlen2 heads dhead -> ... heads seqlen1 seqlen2')

        self.D = EinMix('... heads dhead -> ... dm',
                   weight_shape='heads dhead dm', bias_shape='dm',
                   dm=dmodel, heads=heads, dhead=dhead)

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


def ResidualBlock(dmodel, layer):
    return Residual(nn.Sequential(
        nn.LayerNorm(dmodel),
        layer,
        # nn.LayerNorm(dmodel),
    ))


def EncoderBlock(dmodel, *layers):
    residual_layers = []
    for layer in layers:
        residual_layers.append(ResidualBlock(dmodel, layer))
    return nn.Sequential(*residual_layers)


def check_layer_funs(*layer_funs):
    for layer_fun in layer_funs:
        if isinstance(layer_fun, nn.Module):
            raise TypeError('Expected layer function/lambda, got nn.Module: {}'
                            .format(type(layer_fun)))


def EncoderTower(n_blocks, dmodel, *layer_funs):
    check_layer_funs(*layer_funs)
    encoder_blocks = []
    for i_block in range(n_blocks):
        layers = [layer_fun() for layer_fun in layer_funs]
        encoder_blocks.append(EncoderBlock(dmodel, *layers))
    return nn.Sequential(*encoder_blocks)


class StopGradient(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return stop_gradient(x)


def stop_gradient(x):
    return x.detach()


class StopValuePassGradient(nn.Module):
    def __init__(self):
        super(StopValuePassGradient, self).__init__()

    def forward(self, x):
        return x - x.detach()


class Aggregate(nn.Module):
    def __init__(self, function, *layers):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        result = None
        for layer in self.layers:
            if result is None:
                result = layer(x)
            else:
                result = self.function(result, layer(x))
        return result


def Sum(*layers):
    return Aggregate((lambda x, y: x+y), *layers)


def GradientLike(value_layer, gradient_layer):
    return Sum(
        StopGradient(value_layer),
        StopValuePassGradient(gradient_layer),
    )


def TokenEmbedding(vocab_size, embedding_dim):
    return nn.Embedding(vocab_size, embedding_dim)


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


def EmbeddingLayer(*layers):
    return Sum(*layers)


def PredictionHead(embedding_dim, output_size):
    return nn.Linear(embedding_dim, output_size)


def BERT(embedding_layer, encoder_tower, head):
    return nn.Sequential(
        embedding_layer,
        encoder_tower,
        head
    )
