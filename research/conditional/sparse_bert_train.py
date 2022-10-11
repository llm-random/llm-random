import sys

import torch
import torch.nn.functional as F
import datetime

import research.conditional.ffs
from lizrd.core import misc
from lizrd.core import bert
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
import time
from lizrd.datasets import wikibookdata
from lizrd.support import profile

MASK_PERCENT = 0.2
MASK_LOSS_WEIGHT = 1.0
CLASS_LOSS_WEIGHT = 1.0
LEARNING_RATE = 0.0001

# # BERT-Mini
# DM = 256
# DFF = DM * 4
# BLOCKS = 4
# HEADS = 4

VOCAB_SIZE = 30522  # BertTokenizer uses this many words

TASK = None  # ClearML task
WRITER = None  # Tensorboard writer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DENSE = True

profile.DISABLED = True

NEXPERTS = 32
SPARSITY = 8
EXPERTSIZE = 64

NAME = ''
TESTING = False

for arg in sys.argv[1:]:
    if arg == 'TESTING':
        TESTING = True
    elif arg.startswith('LEARNING_RATE='):
        LEARNING_RATE = float(arg[len('LEARNING_RATE='):])
        LR_FROM_ARG = True
    elif arg.startswith('NEXPERTS='):
        NEXPERTS = int(arg[len('NEXPERTS='):])
    elif arg.startswith('SPARSITY='):
        SPARSITY = int(arg[len('SPARSITY='):])
    elif arg.startswith('EXPERTSIZE='):
        EXPERTSIZE = int(arg[len('EXPERTSIZE='):])
    elif arg == 'DENSE':
        DENSE = True
    elif arg == 'SPARSE':
        DENSE = False
    elif arg.startswith('CLEARMLDIR='):
        CLEARMLDIR = arg[len('CLEARMLDIR='):]
    elif arg.startswith('NAME='):
        NAME = arg[len('NAME='):]
    else:
        raise ValueError('Unknown argument: {}'.format(arg))

# Custom Bert, based on Small BERT
if TESTING:
    CUTOFF = 32
    DM = 16
    DFF = DM * 4
    BLOCKS = 2
    HEADS = 2
    BATCH_SIZE = 2
    USE_CLEARML = False
else:
    CUTOFF = 128
    DM = 512
    DFF = DM * 4
    BLOCKS = 4
    HEADS = 8
    BATCH_SIZE = 32
    USE_CLEARML = True


def get_model(dense=False):
    batch, seql, dm, heads, dff = BATCH_SIZE, CUTOFF, DM, HEADS, DFF
    vocab_size, max_length = VOCAB_SIZE, CUTOFF
    output_size = VOCAB_SIZE
    n_blocks = BLOCKS

    if USE_CLEARML:
        TASK.connect_configuration(name='hiperparameters', configuration={
            'batch': batch, 'seql': seql, 'dm': dm, 'heads': heads, 'dff': dff,
            'vocab_size': vocab_size, 'max_length': max_length,
            'output_size': output_size,
            'n_blocks': n_blocks,
            'learning_rate': LEARNING_RATE,
            'mask_loss_weight': MASK_LOSS_WEIGHT,
            'class_loss_weight': CLASS_LOSS_WEIGHT,
        })

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm),
        bert.TokenEmbedding(vocab_size, dm)
    )

    if dense:
        ff_layer = (lambda: bert.FeedForward(dm, dff))
    else:
        ff_layer = (lambda: research.conditional.ffs.RewrittenSplitFF([], dm, dff,
                                                                      NEXPERTS, SPARSITY, EXPERTSIZE))

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        (lambda: bert.Attention(dm, heads)),
        ff_layer,
        ff_layer,
        ff_layer,
    )

    head = bert.PredictionHead(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)

    input = torch.randint(0, vocab_size, (batch, seql))
    output = model(input)
    del output  # this is just a check

    return model


def train_step(model, optimizer, pdataset, step=0):
    model.train()
    processed_batch = pdataset.get_batch(BATCH_SIZE)
    assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
    x_set = processed_batch.masked_tokens
    y_class_set = processed_batch.swapped
    y_token_set = processed_batch.tokens
    y_mask_set = processed_batch.mask_mask

    model_output = model(x_set)
    mask_loss = F.cross_entropy(
        model_output.reshape(-1, VOCAB_SIZE),
        y_token_set.reshape(-1).long(),
        reduction='none')
    mask_loss *= y_mask_set.reshape(-1)  # only check masked words
    mask_loss = mask_loss.mean() / MASK_PERCENT
    scaled_mask_loss = mask_loss * MASK_LOSS_WEIGHT
    # class_loss = F.binary_cross_entropy_with_logits(
    #     model_output[:, 0, MASK_ID], y_class_set.double())
    # total_loss = scaled_mask_loss + class_loss
    total_loss = scaled_mask_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step and WRITER:
        WRITER.add_scalar('loss/train_total', total_loss.item(), step)
        WRITER.add_scalar('loss/train_mask', mask_loss.item(), step)
        # WRITER.add_scalar('loss/train_scaled_mask', scaled_mask_loss.item(), step)
        # WRITER.add_scalar('loss/train_class', class_loss.item(), step)


def eval_step(model, pdataset, step=0, sample=10):
    model.eval()

    with torch.no_grad():
        total_mask_loss = 0.0
        for sample_i in range(sample):
            processed_batch = pdataset.get_batch(BATCH_SIZE)
            assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
            x_set = processed_batch.masked_tokens
            y_class_set = processed_batch.swapped
            y_token_set = processed_batch.tokens
            y_mask_set = processed_batch.mask_mask
            model_output = model(x_set)
            mask_loss = F.cross_entropy(
                model_output.reshape(-1, VOCAB_SIZE),
                y_token_set.reshape(-1).long(),
                reduction='none')
            mask_loss *= y_mask_set.reshape(-1)  # only check masked words
            mask_loss = mask_loss.mean() / MASK_PERCENT
            scaled_mask_loss = mask_loss * MASK_LOSS_WEIGHT
            total_mask_loss += scaled_mask_loss.item()
        total_mask_loss /= sample

        if step and WRITER:
            WRITER.add_scalar('loss/eval_mask', total_mask_loss, step)

        return total_mask_loss


def get_processed_dataset():
    raw_dataset = wikibookdata.WikiBookDataset()
    processor = wikibookdata.SentencePairProcessor(
        max_total_length=CUTOFF,
        device=DEVICE,
        mask_percent=MASK_PERCENT,
        swap_percent=0.0,
    )
    return wikibookdata.ProcessedDataset(raw_dataset, processor)


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    modelpath = f'runs/wikibooktest/{timestamp}'

    if NAME:
        nametext = NAME
    else:
        nametext = 'dense' if DENSE else 'sparse'
    writer = SummaryWriter(log_dir=modelpath)
    if USE_CLEARML:
        task = Task.init(project_name='jaszczur/sparsity/tests',
                         task_name=f'{nametext} {timestamp}')
        TASK = task
    WRITER = writer

    misc.print_available_gpus()

    pdataset = get_processed_dataset()

    model = get_model(DENSE)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    EVAL_STEP = 100
    last_eval_time = None
    for step in range(10000000+1):
        start_train_time = time.time()
        train_step(model, optimizer, pdataset, step)
        end_train_time = time.time()
        WRITER.add_scalar('step', step, step)
        WRITER.add_scalar('time/train', end_train_time - start_train_time, step)
        if step % EVAL_STEP == 0:
            begin_eval_time = time.time()
            eval_loss = eval_step(model, pdataset, step, sample=EVAL_STEP//2)
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
