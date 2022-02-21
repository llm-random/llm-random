import bert
import torch
import time

import misc
import profile


def test_basic(self):
    batch, seql, dm, heads, dff = 3, 12, 32, 4, 64

    vocab_size, max_length = 107, 33
    output_size = 3
    n_blocks = 2

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm),
        bert.TokenEmbedding(vocab_size, dm)
    )

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        (lambda: bert.BatchSplitFF([], dm, dff, 4, 4, 4)),
        (lambda: bert.Attention(dm, heads)),
    )

    head = bert.PredictionHead(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)

    input = torch.randint(0, vocab_size, (batch, seql))

    output = model(input)

    self.assertShape(output, (batch, seql, output_size))


CUDA = torch.device("cuda")

USE_CUDA = True


class NoopEnter(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def main_tests(version, disable_inner=False):
    # multiplier = 32
    one_size = 16
    expertsets = one_size
    nexperts = one_size
    expertsize = one_size * 4
    dff = expertsets * nexperts * expertsize

    batch, seql, dm, heads = 1, 1024, 1024, 16
    vocab_size, max_length = 107, 1024
    output_size = 64
    n_blocks = 4
    samples = 20
    warmup = 10

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm),
        bert.TokenEmbedding(vocab_size, dm)
    )

    if version == 'sparse':
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (lambda: bert.BatchSplitFF([], dm, dff, expertsets, nexperts, expertsize)),
            (lambda: bert.Attention(dm, heads)),
        )
    elif version == 'dense':
        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (lambda: bert.FeedForward(dm, dff)),
            (lambda: bert.Attention(dm, heads)),
        )
    else:
        raise ValueError('Unrecognized type of FF: {}'.format(version))

    head = bert.PredictionHead(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)
    model.train()

    inputs = [torch.randint(0, vocab_size, (batch, seql))
              for s in range(samples+warmup)]
    if USE_CUDA:
        model.to(CUDA)
        inputs = [x.to(CUDA) for x in inputs]

    with torch.no_grad():
    # with NoopEnter():
        for input in inputs[:warmup]:
            output = model(input)
            torch.sum(output).item()  # to make sure everything is computed
        profile.reset_times()
        with profile.Timer(f'{version}', disable_inner=disable_inner):
            for input in inputs[warmup:]:
                output = model(input)
                torch.sum(output).item()  # to make sure everything is computed


if __name__ == "__main__":
    main_tests('sparse', False)
    profile.print_times()
    main_tests('dense')
    profile.print_times()
    # main_tests('sparse', False)
    # bert.print_times()
