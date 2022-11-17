import research.conditional.ffs
from lizrd.core import misc
from lizrd.core import bert
import torch

from lizrd.support import profile


def test_basic(self):
    batch, seql, dm, heads, dff = 3, 12, 32, 4, 64

    vocab_size, max_length = 107, 33
    output_size = 3
    n_blocks = 2

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm), bert.TokenEmbedding(vocab_size, dm)
    )

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        (lambda: research.conditional.ffs.BatchSplitFF([], dm, dff, 4, 4, 4)),
        (lambda: bert.Attention(dm, heads)),
    )

    head = bert.PredictionHead(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)

    input = torch.randint(0, vocab_size, (batch, seql))

    output = model(input)

    self.assertShape(output, (batch, seql, output_size))


CUDA = torch.device("cuda")

USE_CUDA = True
DO_BACKWARD = True


class NoopEnter(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_one_sparse(log_total_dff):
    logexpertsets = 2
    logexpertsize = 5
    lognexperts = log_total_dff - logexpertsets - logexpertsize
    expertsets = 2**logexpertsets
    expertsize = 2**logexpertsize
    nexperts = 2**lognexperts
    print(logexpertsets, logexpertsize, lognexperts)
    print(expertsets, expertsize, nexperts)
    # try:
    main_tests(
        "simplesparse",
        disable_inner=False,
        expertsets=expertsets,
        expertsize=expertsize,
        nexperts=nexperts,
    )
    print("IMPORTANT", round(sum(profile.GLOBAL_TIMERS["batchedFF"]), 3))
    profile.print_times()
    # except:
    #     print("FAILED")
    print("", flush=True)


def test_all_sparse(log_total_dff):
    keys = ["rewrittenFF", "Controller", "FF"]
    tables = {
        key: [["-"] * (log_total_dff + 1) for _ in range(log_total_dff + 1)]
        for key in keys
    }

    for logexpertsets in range(0, log_total_dff + 1):
        for logexpertsize in range(0, log_total_dff + 1 - logexpertsets):
            lognexperts = log_total_dff - logexpertsets - logexpertsize
            expertsets = 2**logexpertsets
            expertsize = 2**logexpertsize
            nexperts = 2**lognexperts
            print(logexpertsets, logexpertsize, lognexperts)
            print(expertsets, expertsize, nexperts)
            try:
                main_tests(
                    "rewritten",
                    disable_inner=False,
                    expertsets=expertsets,
                    expertsize=expertsize,
                    nexperts=nexperts,
                )
                for key in keys:
                    time = round(sum(profile.GLOBAL_TIMERS[key]), 3)
                    tables[key][logexpertsize][logexpertsets] = time
                profile.print_times()
            except:
                print("FAILED")
                for key in keys:
                    tables[key][logexpertsize][logexpertsets] = "nan"
            print("\n\n\n")
            for key in keys:
                print(f"\n\n{key}")
                for row in tables[key]:
                    for cell in row:
                        print(cell, end="\t")
                    print()
            print("", flush=True)


def main_tests(version, disable_inner=False, expertsets=4, expertsize=64, nexperts=8):
    # multiplier = 32
    # one_size = 16
    # expertsets = one_size
    # total_dff = 2048

    total_dff = expertsets * expertsize * nexperts

    # expertsets = 16
    # nexperts = int((total_dff/expertsets)**0.5)
    # expertsize = 8
    # nexperts = one_size
    # nexperts = 16
    # expertsize = one_size * 4
    # expertsize = 16
    assert expertsets * nexperts * expertsize == total_dff
    dff = total_dff

    # DM = 512
    # DFF = DM * 4
    # BLOCKS = 4
    # HEADS = 8

    batch, seql, dm, heads = 64, 128, 512, 8
    vocab_size, max_length = 30522, 128
    output_size = 30522
    n_blocks = 4
    samples = 100
    warmup = 10

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm), bert.TokenEmbedding(vocab_size, dm)
    )

    if version == "sparse":
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.ffs.BatchSplitFF(
                    [], dm, dff, expertsets, nexperts, expertsize
                )
            ),
            (lambda: profile.TimerLayer("attention", bert.Attention(dm, heads))),
        )
    elif version == "rewritten":
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.ffs.RewrittenSplitFF(
                    [], dm, dff, expertsets * nexperts, nexperts, expertsize
                )
            ),
            (lambda: profile.TimerLayer("attention", bert.Attention(dm, heads))),
        )
    elif version == "simplesparse":
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.ffs.SimpleSplitFF(
                    [], dm, dff, expertsets, nexperts, expertsize
                )
            ),
            (lambda: profile.TimerLayer("attention", bert.Attention(dm, heads))),
        )
    elif version == "sparse+qkv":
        modules = 4
        sparse_linear_projection = lambda: research.conditional.ffs.FactoredDense(
            dm, dm, modules
        )
        sparse_linear_projection = (
            lambda func=sparse_linear_projection: profile.TimerLayer(
                "projection", func()
            )
        )
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.ffs.BatchSplitFF(
                    [], dm, dff, expertsets, nexperts, expertsize
                )
            ),
            (
                lambda: profile.TimerLayer(
                    "attention",
                    bert.Attention(dm, heads, layer_fun=sparse_linear_projection),
                )
            ),
        )
    elif version == "sparse+lowrank":
        lowrank = 16
        sparse_linear_projection = lambda: bert.LowRank(dm, dm, lowrank)
        sparse_linear_projection = (
            lambda func=sparse_linear_projection: profile.TimerLayer(
                "projection", func()
            )
        )
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.ffs.BatchSplitFF(
                    [], dm, dff, expertsets, nexperts, expertsize
                )
            ),
            (
                lambda: profile.TimerLayer(
                    "attention",
                    bert.Attention(dm, heads, layer_fun=sparse_linear_projection),
                )
            ),
        )
    elif version == "sparse+perm":
        sparse_linear_projection = lambda: research.conditional.ffs.PermutationDense(dm)
        sparse_linear_projection = (
            lambda func=sparse_linear_projection: profile.TimerLayer(
                "projection", func()
            )
        )
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.ffs.BatchSplitFF(
                    [], dm, dff, expertsets, nexperts, expertsize
                )
            ),
            (
                lambda: profile.TimerLayer(
                    "attention",
                    bert.Attention(dm, heads, layer_fun=sparse_linear_projection),
                )
            ),
        )
    elif version == "sparse+noop":
        sparse_linear_projection = lambda: research.conditional.ffs.NoopDense()
        sparse_linear_projection = (
            lambda func=sparse_linear_projection: profile.TimerLayer(
                "projection", func()
            )
        )
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.ffs.BatchSplitFF(
                    [], dm, dff, expertsets, nexperts, expertsize
                )
            ),
            (
                lambda: profile.TimerLayer(
                    "attention",
                    bert.Attention(dm, heads, layer_fun=sparse_linear_projection),
                )
            ),
        )
    elif version == "dense":
        sparse_linear_projection = lambda: misc.DenseEinMix(dm, dm)
        sparse_linear_projection = (
            lambda func=sparse_linear_projection: profile.TimerLayer(
                "projection", func()
            )
        )
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (lambda: bert.FeedForward(dm, dff)),
            (
                lambda: profile.TimerLayer(
                    "attention",
                    bert.Attention(dm, heads, layer_fun=sparse_linear_projection),
                )
            ),
        )
    else:
        raise ValueError("Unrecognized type of FF: {}".format(version))

    head = bert.PredictionHead(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)
    model = profile.TimerLayer("model", model)
    model.train()

    inputs = [
        torch.randint(0, vocab_size, (batch, seql)) for s in range(samples + warmup)
    ]
    if USE_CUDA:
        model.to(CUDA)
        inputs = [x.to(CUDA) for x in inputs]

    with (NoopEnter() if DO_BACKWARD else torch.no_grad()):
        for input in inputs[:warmup]:
            output = model(input)
            loss = torch.sum(output)
            if DO_BACKWARD:
                loss.backward()
                # optimizer.step()
                torch.sum(output).item()  # to make sure everything is computed
        profile.reset_times()
        with profile.Timer(f"{version}", disable_inner=disable_inner):
            for input in inputs[warmup:]:
                output = model(input)
                torch.sum(output).item()  # to make sure everything is computed


if __name__ == "__main__":
    # main_tests('sparse+qkv', False)
    # profile.print_times()
    # main_tests('sparse+lowrank', False)
    # profile.print_times()
    # main_tests('sparse+noop', False)
    # profile.print_times()
    # main_tests('sparse+perm', False)
    # profile.print_times()
    # test_all_sparse(15)
    main_tests("rewritten", False)
    profile.print_times()
    main_tests("dense", False)
    profile.print_times()
    test_all_sparse(11)
    # test_all_sparse(15)
    # main_tests('sparse', False)
    # bert.print_times()
