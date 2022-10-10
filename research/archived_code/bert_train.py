import torch
from tensorflow.keras.datasets import imdb
import torch.nn.functional as F
import datetime
import numpy as np

import research.conditional.ffs
from lizrd.core import misc
from lizrd.core import bert
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
import time

INDEX_FROM = 4
CUTOFF = 128
MASK_ID = 3
PAD_ID = 0
CLS_ID = 1
assert INDEX_FROM == 4

BATCH_SIZE = 64
MASK_PERCENT = 0.3
LEARNING_RATE = 0.0001
MASK_RELATIVE_LR = 0.15

# VOCAB_SIZE = 98304
VOCAB_SIZE = 4 * 1024
# VOCAB_SIZE = 2048
NUM_WORDS = VOCAB_SIZE - INDEX_FROM

TASK = None  # ClearML task
WRITER = None  # Tensorboard writer

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

    TASK.connect_configuration(name='hiperparameters', configuration={
        'batch': batch, 'seql': seql, 'dm': dm, 'heads': heads, 'dff': dff,
        'vocab_size': vocab_size, 'max_length': max_length,
        'output_size': output_size,
        'n_blocks': n_blocks,
        'learning_rate': LEARNING_RATE,
        'mask_relative_lr': MASK_RELATIVE_LR,
    })

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm),
        bert.TokenEmbedding(vocab_size, dm)
    )

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        # (lambda: bert.FeedForward(dm, dff)),
        (lambda: research.conditional.ffs.BatchSplitFF([], dm, dff, 8, 8, 16)),
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


def train_step(model, optimizer, step=0):
    model.train()
    x_set, y_class_set, y_token_set = get_batch(x_train, y_train)

    model_output = model(x_set)
    mask_loss = F.cross_entropy(
        model_output.reshape(-1, VOCAB_SIZE),
        y_token_set.reshape(-1).long())
    scaled_mask_loss = mask_loss * MASK_RELATIVE_LR
    class_loss = F.binary_cross_entropy_with_logits(
        model_output[:, 0, MASK_ID], y_class_set.double())
    total_loss = scaled_mask_loss + class_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step and WRITER:
        WRITER.add_scalar('loss/train_total', total_loss.item(), step)
        WRITER.add_scalar('loss/train_mask', mask_loss.item(), step)
        WRITER.add_scalar('loss/train_scaled_mask', scaled_mask_loss.item(), step)
        WRITER.add_scalar('loss/train_class', class_loss.item(), step)


def eval_step(model, step=0, sample=10):
    model.eval()

    with torch.no_grad():
        class_loss = 0.0
        for sample_i in range(sample):
            x_set, y_class_set, y_token_set = get_batch(x_test, y_test)
            model_output = model(x_set)
            class_loss += F.binary_cross_entropy_with_logits(
                model_output[:, 0, MASK_ID], y_class_set.double()).detach()
        class_loss /= sample

        if step and WRITER:
            WRITER.add_scalar('loss/eval_class', class_loss.item(), step)

        return class_loss.detach()


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    modelpath = f'runs/logtest/{timestamp}'

    writer = SummaryWriter(log_dir=modelpath)
    task = Task.init(project_name='jaszczur/sparsity/tests',
                     task_name=f'logging test {timestamp}')
    TASK = task
    WRITER = writer

    misc.print_available_gpus()

    model = get_model()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    EVAL_STEP = 100
    last_eval_time = None
    for step in range(10000+1):
        start_train_time = time.time()
        train_step(model, optimizer, step)
        end_train_time = time.time()
        WRITER.add_scalar('time/train', end_train_time - start_train_time, step)
        if step % EVAL_STEP == 0:
            begin_eval_time = time.time()
            eval_loss = eval_step(model, step, sample=EVAL_STEP//2)
            print(f'Eval loss:', eval_loss)
            torch.save(model.state_dict(), f'{modelpath}/model.pt')
            end_eval_time = time.time()
            WRITER.add_scalar('time/eval', end_eval_time - begin_eval_time, step)
            if last_eval_time:
                eval_time = end_eval_time - begin_eval_time
                since_last_eval = end_eval_time - last_eval_time
                eval_time_percent = eval_time / since_last_eval
                print(f'Eval time percent: {eval_time_percent}')
                if WRITER:
                    WRITER.add_scalar('time_percent/eval_time', eval_time_percent, step)
            last_eval_time = end_eval_time
        print(f'Step {step}')
