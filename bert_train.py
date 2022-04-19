import matplotlib.pyplot as plt
import tensorflow
import torch
from tensorflow.keras.datasets import imdb
import torch.nn.functional as F
import random
import numpy as np
import bert

INDEX_FROM = 4
CUTOFF = 128
MASK_ID = 3
PAD_ID = 0
CLS_ID = 1
assert INDEX_FROM == 4

BATCH_SIZE = 64
MASK_PERCENT = 0.3


# VOCAB_SIZE = 98304
# VOCAB_SIZE = 16 * 1024
VOCAB_SIZE = 2048
NUM_WORDS = VOCAB_SIZE - INDEX_FROM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=NUM_WORDS,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=INDEX_FROM)


def get_sample(x_set, y_set):
    assert len(x_set) == len(y_set)
    indices = np.random.randint(0, len(x_set), size=BATCH_SIZE)
    x_batch = x_set[indices]
    y_batch = y_set[indices]
    return x_batch, y_batch


def get_padding(x_set, y_set):
    x_set = [x[:CUTOFF] for x in x_set]
    x_set = [x + [PAD_ID] * (CUTOFF - len(x)) for x in x_set]
    return np.array(x_set), np.array(y_set)


def get_mask_target(x_set, y_set):
    y_class_set = y_set
    y_token_set = x_set
    return x_set, y_class_set, y_token_set


def apply_mask(x_set, y_class_set, y_token_set):
    remove_mask = np.random.random(size=x_set.shape) < MASK_PERCENT
    real_index = x_set >= INDEX_FROM
    remove_mask = real_index & remove_mask
    x_set = np.where(remove_mask, MASK_ID, x_set)
    return x_set, y_class_set, y_token_set


def get_model():
    batch, seql, dm, heads, dff = BATCH_SIZE, CUTOFF, 256, 16, 1024
    vocab_size, max_length = VOCAB_SIZE, CUTOFF
    output_size = VOCAB_SIZE
    n_blocks = 6

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm),
        bert.TokenEmbedding(vocab_size, dm)
    )

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        # (lambda: bert.FeedForward(dm, dff)),
        (lambda: bert.BatchSplitFF([], dm, dff, 8, 8, 16)),
        (lambda: bert.Attention(dm, heads)),
    )

    head = bert.PredictionHead(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)

    input = torch.randint(0, vocab_size, (batch, seql))
    output = model(input)
    del output  # this is just a check

    return model


def get_batch(x_set, y_set):
    processing = [
        get_sample,
        get_padding,
        get_mask_target,
        apply_mask,
        convert_to_tensors,
        to_devices,
    ]

    arr = x_set, y_set
    for func in processing:
        arr = func(*arr)
    x_set, y_class_set, y_token_set = arr
    return x_set, y_class_set, y_token_set


def get_eval_batch(x_set, y_set):
    processing = [
        get_padding,
        get_mask_target,
        convert_to_tensors,
        to_devices,
    ]

    arr = x_set, y_set
    for func in processing:
        arr = func(*arr)
    x_set, y_class_set, y_token_set = arr
    return x_set, y_class_set, y_token_set


def convert_to_tensors(*arr):
    arr = [torch.from_numpy(x) for x in arr]
    return arr


def to_devices(*arr):
    arr = [x.to(DEVICE) for x in arr]
    return arr


def train_step(model, optimizer):
    model.train()
    x_set, y_class_set, y_token_set = get_batch(x_train, y_train)

    model_output = model(x_set)
    mask_loss = F.cross_entropy(
        model_output.reshape(-1, VOCAB_SIZE),
        y_token_set.reshape(-1).long())
    class_loss = F.binary_cross_entropy_with_logits(
        model_output[:, 0, MASK_ID], y_class_set.double())
    total_loss = mask_loss + class_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


def eval_step(model):
    model.eval()
    x_set, y_class_set, y_token_set = get_batch(x_test, y_test)

    with torch.no_grad():
        model_output = model(x_set)
        class_loss = F.binary_cross_entropy_with_logits(
            model_output[:, 0, MASK_ID], y_class_set.double())

        return class_loss.detach()


if __name__ == "__main__":
    model = get_model()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for step in range(1000000+1):
        train_step(model, optimizer)
        if step % 1000 == 0:
            eval_loss = eval_step(model)
            print(f'Eval loss:', eval_loss)
        print(f'Step {step}')
