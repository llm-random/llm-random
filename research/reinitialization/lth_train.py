import sys
import time
import copy

import torch
import torch.nn.functional as F
import datetime
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from lizrd.core import misc
from lizrd.core import bert
from lizrd.datasets import wikibookdata
from lizrd.support import profile
from research.reinitialization import linears
from research.reinitialization.pruner import Pruner, LTHPruner

MASK_PERCENT = 0.2
MASK_LOSS_WEIGHT = 1.0
CLASS_LOSS_WEIGHT = 1.0
LEARNING_RATE = 0.0001

# BERT-Mini
DM = 256
DFF = DM * 4
BLOCKS = 4
HEADS = 4
CUTOFF = 32
BATCH_SIZE = 32
USE_CLEARML = True

VOCAB_SIZE = 30522  # BertTokenizer uses this many words

TASK = None  # ClearML task
WRITER = None  # Tensorboard writer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

profile.DISABLED = True

NAME = ""
TESTING = False

for arg in sys.argv[1:]:
    if arg == "TESTING":
        TESTING = True
    elif arg.startswith("LEARNING_RATE="):
        LEARNING_RATE = float(arg[len("LEARNING_RATE=") :])
        LR_FROM_ARG = True
    elif arg.startswith("CLEARMLDIR="):
        CLEARMLDIR = arg[len("CLEARMLDIR=") :]
    elif arg.startswith("NAME="):
        NAME = arg[len("NAME=") :]
    else:
        raise ValueError("Unknown argument: {}".format(arg))


def get_model(pruner):
    batch, seql, dm, heads, dff = BATCH_SIZE, CUTOFF, DM, HEADS, DFF
    vocab_size, max_length = VOCAB_SIZE, CUTOFF
    output_size = VOCAB_SIZE
    n_blocks = BLOCKS

    if USE_CLEARML:
        TASK.connect_configuration(
            name="hiperparameters",
            configuration={
                "batch": batch,
                "seql": seql,
                "dm": dm,
                "heads": heads,
                "dff": dff,
                "vocab_size": vocab_size,
                "max_length": max_length,
                "output_size": output_size,
                "n_blocks": n_blocks,
                "learning_rate": LEARNING_RATE,
                "mask_loss_weight": MASK_LOSS_WEIGHT,
                "class_loss_weight": CLASS_LOSS_WEIGHT,
                # "pruner_prob": pruner.prob,
                # "pruner_n_steps": pruner.n_steps_prune,
            },
        )

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm), bert.TokenEmbedding(vocab_size, dm)
    )

    ff_layer = lambda: linears.StructMagnitudePruneFF(dm, dff, pruner)

    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        (lambda: bert.Attention(dm, heads)),
        ff_layer,
    )

    head = bert.PredictionHead(dm, output_size)

    model = bert.BERT(embedding_layer, encoder_tower, head)

    input = torch.randint(0, vocab_size, (batch, seql))
    output = model(input)
    del output  # this is just a check

    return model


def train_step(model, optimizer, pdataset, pruner, step=0):
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
        reduction="none",
    )
    mask_loss *= y_mask_set.reshape(-1)  # only check masked words
    mask_loss = mask_loss.mean() / MASK_PERCENT
    scaled_mask_loss = mask_loss * MASK_LOSS_WEIGHT
    total_loss = scaled_mask_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step and WRITER:
        WRITER.add_scalar("loss/train_total", total_loss.item(), step)
        WRITER.add_scalar("loss/train_mask", mask_loss.item(), step)


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
                reduction="none",
            )
            mask_loss *= y_mask_set.reshape(-1)  # only check masked words
            mask_loss = mask_loss.mean() / MASK_PERCENT
            scaled_mask_loss = mask_loss * MASK_LOSS_WEIGHT
            total_mask_loss += scaled_mask_loss.item()
        total_mask_loss /= sample

        if step and WRITER:
            WRITER.add_scalar("loss/eval_mask", total_mask_loss, step)

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

# 1. train for a given number epochs
# 2. select some parameters having low impact (e. g. Magnitude pruning)
# 3. record parameters in mask
# 4. retrain from scratch

params_left = 1.0
target_params_left = 0.1
params_to_prune_per_run = 0.1
rerun_epochs = 1000

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    modelpath = f"runs/wikibooktest/{timestamp}"
    nametext = NAME
    writer = SummaryWriter(log_dir=modelpath)
    if USE_CLEARML:
        task = Task.init(
            project_name="mpioro/reinit/lth", task_name=f"{nametext} {timestamp}"
        )
        TASK = task
    WRITER = writer
    misc.print_available_gpus()

    pruner = LTHPruner()
    model = get_model(pruner)
    pruner.register_model(model)

    last_eval_time = None
    full_step = 0

    model = get_model(pruner)
    while params_left > target_params_left:
        pdataset = get_processed_dataset()
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        memory = torch.cuda.mem_get_info(0)
        print(memory)
        WRITER.add_scalar("memory", scalar_value=full_step, global_step=full_step)
        WRITER.add_scalar("params_left", params_left, full_step)
        for run_step in range(rerun_epochs):
            train_step(model, optimizer, pdataset, pruner, full_step)
            full_step += 1
            WRITER.add_scalar("full_step", full_step, full_step)
        eval_loss = eval_step(model, pdataset, full_step, sample=100)
        pruner.prune(params_to_prune_per_run * params_left)
        params_left *= (1 - params_to_prune_per_run)
        pruner.reinitialize()
        torch.save(model.state_dict(), f"{modelpath}/model.pt")
