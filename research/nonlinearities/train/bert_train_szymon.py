import sys

import torch
import torch.nn.functional as F
import datetime

from lizrd.core import misc
from lizrd.core import bert
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
import time
from lizrd.datasets import wikibookdata
from lizrd.support import profile
from research.nonlinearities.core import research_bert

MASK_PERCENT = 0.2
MASK_LOSS_WEIGHT = 1.0
CLASS_LOSS_WEIGHT = 1.0
LEARNING_RATE = 0.00005
VOCAB_SIZE = 30522  # BertTokenizer uses this many words

TASK = None  # ClearML task
WRITER = None  # Tensorboard writer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

profile.DISABLED = True

NAME = ""
TESTING = False
FF_MODE = None
EXP_RATE_FROM_ARG, N_FF_HEADS_FROM_ARG, N_CHUNKS_FROM_ARG = False, False, False

for arg in sys.argv[1:]:
    if arg == "TESTING":
        TESTING = True
    elif arg.startswith("FF_MODE="):
        FF_MODE = str(arg[len("LEARNING_RATE=") :])
    elif arg.startswith("LEARNING_RATE="):
        LEARNING_RATE = float(arg[len("LEARNING_RATE=") :])
        LR_FROM_ARG = True
    elif arg.startswith("EXP_RATE="):
        EXP_RATE = int(arg[len("EXP_RATE=") :])
        EXP_RATE_FROM_ARG = True
    elif arg.startswith("N_FF_HEADS="):
        N_FF_HEADS = int(arg[len("N_FF_HEADS=") :])
        N_FF_HEADS_FROM_ARG = True
    elif arg.startswith("N_CHUNKS="):
        N_CHUNKS = int(arg[len("EXP_RATE=") :])
        N_CHUNKS_FROM_ARG = True
    elif arg.startswith("CLEARMLDIR="):
        CLEARMLDIR = arg[len("CLEARMLDIR=") :]
    elif arg.startswith("NAME="):
        NAME = arg[len("NAME=") :]
    elif arg.startswith("USE_CLEARML="):
        USE_CLEARML = bool(int(arg[len("USE_CLEARML=") :]))
    else:
        raise ValueError("Unknown argument: {}".format(arg))

# Custom Bert, based on Small BERT
FF_MODE = "bottleneck"
EXP_RATE = 8
if TESTING:
    CUTOFF = 32
    DM = 16
    DFF = DM * 4
    BLOCKS = 2
    HEADS = 2
    BATCH_SIZE = 2
    USE_CLEARML = False
    N_FF_HEADS = 2
else:
    CUTOFF = 128
    DM = 512
    DFF = DM * 4
    BLOCKS = 4
    HEADS = 8
    BATCH_SIZE = 32
    USE_CLEARML = True


FF_MODE_MAP = {
    "vanilla": (bert.FeedForward, (DM, DFF)),
    "bottleneck": (research_bert.FeedForwardBottleneck, (DM, EXP_RATE)),
    "multineck": (research_bert.FeedForwardMultineck, (DM, EXP_RATE, N_FF_HEADS)),
    "choppedneck": (research_bert.FeedForwardChoppedneck, (DM, N_CHUNKS)),
}
assert FF_MODE is not None, f"FF_MODE must be specified"
assert None not in FF_MODE_MAP[FF_MODE][1], f"FF args must be specified"


def get_model():
    batch, seql, dm, heads, dff, ff_heads, ff_mode = (
        BATCH_SIZE,
        CUTOFF,
        DM,
        HEADS,
        DFF,
        N_FF_HEADS,
        FF_MODE,
    )
    vocab_size, max_length = VOCAB_SIZE, CUTOFF
    output_size = VOCAB_SIZE
    n_blocks = BLOCKS

    if USE_CLEARML:
        TASK.set_parameters(
            **{
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
                "ff_num_heads": N_FF_HEADS,
                "ff_mode": FF_MODE,
            },
        )

    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm), bert.TokenEmbedding(vocab_size, dm)
    )

    ff_layer_type, ff_kwargs = FF_MODE_MAP[ff_mode]
    ff_layer = lambda: ff_layer_type(*ff_kwargs)

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
        reduction="none",
    )
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
        # WRITER.add_scalar("loss/train_total", total_loss.item(), step)
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


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    modelpath = f"runs/wikibooktest/{timestamp}"

    nametext = NAME
    writer = SummaryWriter(log_dir=modelpath)
    if USE_CLEARML:
        task = Task.init(
            project_name="nonlinearities/initial_tests",
            task_name=f"{nametext} {timestamp}",
        )
        TASK = task
    WRITER = writer

    misc.print_available_gpus()

    pdataset = get_processed_dataset()

    model = get_model()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"learning rate: {LEARNING_RATE}")
    EVAL_STEP = 100
    last_eval_time = None
    for step in range(10000000 + 1):
        start_train_time = time.time()
        train_step(model, optimizer, pdataset, step)
        end_train_time = time.time()
        WRITER.add_scalar("step", step, step)
        WRITER.add_scalar("time/train", end_train_time - start_train_time, step)
        if step % EVAL_STEP == 0:
            begin_eval_time = time.time()
            eval_loss = eval_step(model, pdataset, step, sample=EVAL_STEP // 2)
            print(f"Eval loss:", eval_loss)
            torch.save(model.state_dict(), f"{modelpath}/model.pt")
            end_eval_time = time.time()
            WRITER.add_scalar("time/eval", end_eval_time - begin_eval_time, step)
            if last_eval_time:
                eval_time = end_eval_time - begin_eval_time
                since_last_eval = end_eval_time - last_eval_time
                eval_time_percent = eval_time / since_last_eval
                print(f"Eval time percent: {eval_time_percent}")
                if WRITER:
                    WRITER.add_scalar("time_percent/eval_time", eval_time_percent, step)
            last_eval_time = end_eval_time
        print(f"Step {step}")
