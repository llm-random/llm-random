import torch
from lizrd.core import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from time import sleep
from opt_einsum import contract
import opt_einsum
from einops.layers.torch import Rearrange, Reduce

# https://einops.rocks/3-einmix-layer/

import gc
from typing import List

IGNORE_ASSERTS = True


if IGNORE_ASSERTS:

    def assert_shape(x, shape: List[int]):
        pass

else:

    def assert_shape(x, shape):
        for a, b in zip(x.shape, shape):
            if b is not None and a != b:
                raise AssertionError(
                    "invalid shape; got {} but expected {}.".format(x.shape, shape)
                )


if IGNORE_ASSERTS:

    def assertf(condition: bool):
        pass

else:

    def assertf(condition: bool):
        assert condition


# class Einsum(nn.Module):
#     def __init__(self, text, *sizes):
#         self.text = text
#         self.sizes = sizes
#
#     def forward(self, *tensors):
#         assert len(tensors) == len(self.sizes)
#         for tensor, size in zip(tensors, self.sizes):
#             assert_shape(tensor, size)
#         output = torch.einsum(self.text, *tensors)
#         return output

OPTIMIZED = False

CONTRACT_CACHE = dict()


def add_or_get_contract(text, *tensors):
    shapes = tuple(tuple(tensor.shape) for tensor in tensors)
    key = (text, shapes)

    if key not in CONTRACT_CACHE:
        print("adding {}".format(key))
        path = opt_einsum.contract_path(text, *tensors, optimize="optimal")
        print(path[1])
        print("...")
        CONTRACT_CACHE[key] = opt_einsum.contract_expression(
            text, *shapes, optimize="optimal"
        )
    return CONTRACT_CACHE[key]


def einsum(text, *tensors):
    if OPTIMIZED:
        expr = add_or_get_contract(text, *tensors)
        return expr(*tensors)
    else:
        return torch.einsum(text, *tensors)


class DenseFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super(DenseFF, self).__init__()
        self.f1 = nn.Linear(d_model, d_ff)
        self.f2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        ff = self.f1(x)
        ff = F.relu(ff)
        out = self.f2(ff)
        return out


class DenseFFEinsum(nn.Module):
    def __init__(self, d_model, d_ff):
        super(DenseFFEinsum, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.f1 = nn.Parameter(torch.Tensor(d_ff, d_model))
        self.f2 = nn.Parameter(torch.Tensor(d_ff, d_model))

    def forward(self, x):
        inner = einsum("bm,nm->bn", x, self.f1)
        inner = F.relu(inner)
        output = einsum("bn,nm->bm", inner, self.f2)
        assert_shape(output, (None, self.d_model))
        return output


class LowRank(nn.Module):
    def __init__(self, d_model, d_lowrank, d_output=None):
        super(LowRank, self).__init__()
        if d_output is None:
            d_output = d_model

        self.f1 = nn.Parameter(torch.Tensor(d_model, d_lowrank))
        self.f2 = nn.Parameter(torch.Tensor(d_lowrank, d_output))
        # self.contract =

    def forward(self, x):
        out = einsum("bm,ml,lo->bo", x, self.f1, self.f2)
        # out = contract('bm,ml,lo->bo', x, self.f1, self.f2)
        # lowrank = torch.einsum('bm,ml->bl', x, self.f1)
        # out = torch.einsum('bl,lo->bo', lowrank, self.f2)
        return out


def stop_gradient(x):
    return x.detach()


class GradientsLike(nn.Module):
    def __init__(self):
        super(GradientsLike, self).__init__()

    def forward(self, x):
        return x - stop_gradient(x)


class SparseController(nn.Module):
    def __init__(self, d_model, d_lowrank, d_ff, N):
        super(SparseController, self).__init__()
        assertf(d_ff % N == 0)
        self.lowrank = LowRank(d_model, d_lowrank, d_ff)
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_lowrank = d_lowrank

    def forward(self, x):
        N = self.N
        assert_shape(x, (BATCH_SIZE, self.d_model))
        out = self.lowrank(x)

        out = out.view(BATCH_SIZE, -1, N)
        assertf(out.shape == (BATCH_SIZE, self.d_ff // N, N))

        # probs = F.softmax(out, dim=-1)
        # TODO(jaszczur): change to discrete
        # result = probs

        result = out
        assertf(result.shape == (BATCH_SIZE, self.d_ff // N, N))
        return result


class SparseFF(nn.Module):
    def __init__(self, d_model, d_ff, d_lowrank, N):
        super(SparseFF, self).__init__()
        assertf(d_ff % N == 0)

        n_expertsets = d_ff // N

        self.d_model = d_model
        self.d_ff = d_ff
        self.d_lowrank = d_lowrank
        self.N = N
        self.controller = SparseController(d_model, d_lowrank, d_ff, N)

        self.f1 = nn.Parameter(torch.Tensor(n_expertsets, N, d_model))
        # TODO(jaszczur): add biases
        # self.f1 = nn.Linear(d_model, d_ff)
        # self.f2 = nn.Linear(d_ff, d_model)
        self.f2 = nn.Parameter(torch.Tensor(n_expertsets, N, d_model))

    def forward(self, x):
        N = self.N
        assertf(x.shape == (BATCH_SIZE, self.d_model))
        controller_output = self.controller(x)
        if self.training:
            inner = einsum("bm,enm->ben", x, self.f1)
            # inner = self.f1(x)
            # inner = inner.view(BATCH_SIZE, self.d_ff//N, N)

            assert_shape(inner, (BATCH_SIZE, self.d_ff // N, N))
            assert_shape(controller_output, (BATCH_SIZE, self.d_ff // N, N))
            inner = F.relu(inner) * controller_output

            output = einsum("ben,enm->bm", inner, self.f2)
            # inner = inner.view(BATCH_SIZE, self.d_ff)
            # output = self.f2(inner)
            assert_shape(output, (BATCH_SIZE, self.d_model))
            return output
        else:
            controller_indexes = torch.argmax(controller_output, dim=-1, keepdim=True)

            assertf(BATCH_SIZE == 1)
            assert_shape(controller_indexes, (BATCH_SIZE, self.d_ff // N, 1))
            controller_indexes = controller_indexes.view(self.d_ff // N)
            assert_shape(self.f1, (self.d_ff // N, N, self.d_model))

            rangeE = torch.arange(self.d_ff // N)

            f1p = self.f1[rangeE, controller_indexes]
            f2p = self.f2[rangeE, controller_indexes]
            # f1p = torch.index_select(self.f1, -1, controller_indexes)
            # f2p = torch.index_select(self.f2, -1, controller_indexes)

            assert_shape(f1p, (self.d_ff // N, self.d_model))
            assert_shape(f2p, (self.d_ff // N, self.d_model))

            inner = einsum("bm,em->be", x, f1p)

            assert_shape(inner, (BATCH_SIZE, self.d_ff // N))

            inner = F.relu(inner)
            output = einsum("be,em->bm", inner, f2p)
            assert_shape(output, (BATCH_SIZE, self.d_model))
            return output


class BatchSplitFF(nn.Module):
    def __init__(self, dmodel, dff, expertsize, nexperts, expertsets):
        super(BatchSplitFF, self).__init__()
        assertf(expertsize * nexperts * expertsets == dff)
        self.dmodel = dmodel
        self.dff = dff
        self.expertsize = expertsize
        self.nexperts = nexperts
        self.expertsets = expertsets

        self.f1 = nn.Parameter(torch.Tensor(nexperts, expertsets, expertsize, dmodel))
        # TODO(jaszczur): add biases
        # self.f1 = nn.Linear(d_model, d_ff)
        # self.f2 = nn.Linear(d_ff, d_model)
        self.f2 = nn.Parameter(torch.Tensor(nexperts, expertsets, expertsize, dmodel))

        self.batch_size = BATCH_SIZE

    def forward(self, x):
        assertf(self.batch_size % self.nexperts == 0)
        assert_shape(x, (self.batch_size, self.dmodel))

        grouped = x.view((-1, self.nexperts, self.dmodel))
        inner = einsum("Bnd,nsed->Bnse", grouped, self.f1)
        assert_shape(inner, (None, self.nexperts, self.expertsets, self.expertsize))
        inner = F.relu(inner)
        output = einsum("Bnse,nsed->Bnd", inner, self.f2)
        assert_shape(output, (None, self.nexperts, self.dmodel))
        ungrouped = output.reshape((-1, self.dmodel))
        assert_shape(ungrouped, (self.batch_size, self.dmodel))
        return ungrouped


class NewBatchSplitFF(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        # BATCH, embedding
        # batch, set, embedding <-- this is just reshape

        ## CONTROLLER:
        # batch, set1, embedding <-- this is starting point
        # batch, set1, set2(experts), expertsets  <--- this comes from linear
        # batch, set1, set2(experts), expertsets <--- sample on 1st dimension (set1)

        # transformation with controller
        # batch, set2(experts), expertsets, embedding

        # f1 weight: set2(experts), expertsets, embedding, expertsize
        # batch, set2(experts), expertsets, expertsize
        # ReLU
        # batch, set2(experts), expertsets, expertsize
        # f2 weight: set2(experts), expertsets, expertsize, embedding
        # batch, set2(experts), expertsets, embedding

        # back from the controller, transformation
        # batch, set1, embedding

        # final reshape
        # BATCH, embedding

        pass


class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.fflayer = layer

    def forward(self, x):
        return self.fflayer(x) + x


class Model(nn.Module):
    def __init__(self, layers, d_model, d_ff, d_lowrank, sparsity, version):
        super(Model, self).__init__()
        if "batch" in version:
            layer_fun = lambda: BatchSplitFF(d_model, d_ff, 32, 32, 4)
        elif "sparse" in version:
            layer_fun = lambda: SparseFF(d_model, d_ff, d_lowrank, sparsity)
        elif "einsum" in version:
            layer_fun = lambda: DenseFFEinsum(d_model, d_ff)
        else:
            layer_fun = lambda: DenseFF(d_model, d_ff)
        self.layers = nn.ModuleList([Residual(layer_fun()) for i in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# model = Model(3, 128, 4*128, 32, 16, 'sparse')

CUDA = torch.device("cuda")


def timemodel(batch, sample, layers, d_model, d_ff, d_lowrank, sparsity, version):
    model = Model(layers, d_model, d_ff, d_lowrank, sparsity, version)
    # model = torch.jit.script(model)
    sample = [torch.Tensor(np.random.random((batch, d_model))) for x in range(sample)]
    if GPU:
        model.to(CUDA)
        sample = [x.to(CUDA) for x in sample]
    if "train" in version:
        model.train()
    else:
        model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(REPEAT):
            for s in sample:
                r = model(s)
    return time.time() - start


BATCH_SIZE = 64
SAMPLE = 100
REPEAT = 1
LAYERS = 20
DMODEL = 1024
DFF = 4 * 1024
DLOWRANK = 16 * 16
SPARSITY = 16
GPU = True


# print(torch.cuda.mem_get_info())
# torch.cuda.empty_cache()
# print(torch.cuda.mem_get_info())
#
# gc.collect()
# torch.cuda.empty_cache()
# print(torch.cuda.mem_get_info())


def standard_sleep():
    sleep(0.2)


if __name__ == "__main__":
    # standard_sleep()
    # BATCH_SIZE = 1
    # # print("dense-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "dense-ff"))
    #
    # standard_sleep()
    # BATCH_SIZE = 32
    # print(BATCH_SIZE)
    # print("batch-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "batch-ff"))
    # print("dense-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "dense-ff"))
    # print("batch-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "batch-ff"))

    # standard_sleep()
    # BATCH_SIZE = 256
    # print(BATCH_SIZE)
    # print("batch-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "batch-ff"))
    # print("dense-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "dense-ff"))
    # print("batch-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "batch-ff"))
    #
    standard_sleep()
    BATCH_SIZE = 4096
    print(BATCH_SIZE)
    print(
        "batch-ff",
        timemodel(
            BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "batch-ff"
        ),
    )
    print(
        "dense-ff",
        timemodel(
            BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "dense-ff"
        ),
    )
    print(
        "batch-ff",
        timemodel(
            BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "batch-ff"
        ),
    )
    print(
        "dense-ff",
        timemodel(
            BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "dense-ff"
        ),
    )


# if __name__ == "__main__":
#     standard_sleep()
#     print("sparse-eval", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "sparse-eval"))
#     standard_sleep()
#     print("sparse-eval", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "sparse-eval"))
#     standard_sleep()
#     print("dense-einsum", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "dense-einsum"))
#     standard_sleep()
#     print("dense-ff", timemodel(BATCH_SIZE, SAMPLE, LAYERS, DMODEL, DFF, DLOWRANK, SPARSITY, "dense-ff"))
