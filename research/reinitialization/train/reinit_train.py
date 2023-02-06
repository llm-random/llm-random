import argparse
from typing import List, Optional

import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from lizrd.core import misc, bert
from research.reinitialization.core import linears
from research.reinitialization.core import linears_recycle
from research.reinitialization.core.pruner import Pruner
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,
    Trainer,
    RetrainTrainer,
)
from research.reinitialization.core.scheduler import DelayedConstScheduler
import secrets
import os

parser = argparse.ArgumentParser()

parser.add_argument("--testing_regular", action="store_true")
parser.add_argument("--testing_recycle", action="store_true")
parser.add_argument("--use_clearml", action="store_true")
parser.add_argument("--use_pruner", action="store_true")
parser.add_argument("--mixed_precision", action="store_true", default=True)

parser.add_argument("--pruner_prob", type=float, default=None)
parser.add_argument("--pruner_n_steps", type=int, default=None)
parser.add_argument("--project_name", type=str)

parser.add_argument("--name", type=str, default="")
parser.add_argument("--pruner_delay", type=int, default=0)
parser.add_argument("--pruner_n_steps_retrain", type=int, default=None)
parser.add_argument("--ff_layer", type=str, default="regular")
parser.add_argument("--trainer_type", type=str, default="regular")
parser.add_argument("--tags", nargs="*", type=str, default=None)
parser.add_argument("--ds_seed", type=int, default=42)
parser.add_argument("--eval_ds_seed", type=int, default=1984)
parser.add_argument("--retrain_ds_seed", type=int, default=1998)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--cutoff", type=int, default=128)
parser.add_argument("--dm", type=int, default=256)
parser.add_argument("--dff", type=int, default=1024)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--dhead", type=int, default=32)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--learning_rate", type=float, default=8e-4)
parser.add_argument("--mask_loss_weight", type=float, default=1.0)
parser.add_argument("--class_loss_weight", type=float, default=1.0)
parser.add_argument("--mask_percent", type=float, default=0.15)
parser.add_argument("--n_steps", type=int, default=100_001)
parser.add_argument("--n_steps_eval", type=int, default=100)
parser.add_argument("--immunity", type=int, default=10)
parser.add_argument("--reinit_dist", type=str, default="init")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--n_log_plots_steps", type=int, default=None)
parser.add_argument("--n_log_steps", type=int, default=100)
parser.add_argument("--retrain_warmup_steps", type=int, default=None)
parser.add_argument("--log_neuron_diff", action="store_true")
parser.add_argument("--log_neuron_diff_steps", type=int, default=1000)
parser.add_argument("--log_neuron_diff_sample_size", type=int, default=1)
parser.add_argument("--log_neuron_diff_n_samples", type=int, default=100)
parser.add_argument("--neuron_diff_ds_seed", type=int, default=511)
parser.add_argument("--neuron_diff_batches", type=int, default=10)
parser.add_argument("--testing_diff", action="store_true")

args = parser.parse_args()

# useful predefined configs for debugging locally
if args.testing_diff:
    args.project_name = f"{os.getenv('USER')}/testing"
    args.name="jk"
    args.log_neuron_diff = True
    args.log_neuron_diff_steps = 10
    args.log_neuron_diff_sample_size = 1
    args.log_neuron_diff_n_samples = 10
    args.ff_layer = "retrain_recycle"
    args.batch_size = 4
    args.cutoff = 32
    args.mixed_precision = True
    args.tags = ["testing_neuron_diff"]
    args.use_clearml = True
    args.use_pruner = True
    args.pruner_n_steps = 2000
    args.pruner_prob = 0.
    args.pruner_delay = 6000
    args.pruner_n_steps_retrain = 0
    args.n_log_plots_steps = 20
    args.trainer_type = "retrain"
    args.n_steps = 100
    args.n_log_steps = 5
elif args.testing_regular:
    args.project_name = f"{os.getenv('USER')}/testing"
    args.ff_layer = "regular"
    args.cutoff = 32
    args.dm = 2
    args.dff = 4
    args.n_blocks = 2
    args.heads = 2
    args.tags = ["testing_regular"]
    args.n_steps = 100
    args.use_pruner = False
    args.batch_size = 4
elif args.testing_recycle:
    args.project_name = f"{os.getenv('USER')}/testing"
    args.use_clearml = True
    args.ff_layer = "retrain_recycle"
    args.cutoff = 32
    args.n_steps = 100
    args.use_clearml = True
    args.tags = ["testing_recycle"]
    args.use_pruner = True
    args.pruner_n_steps = 10
    args.pruner_prob = 0.1
    args.pruner_delay = 6
    args.pruner_n_steps_retrain = 10
    args.trainer_type = "retrain"
    args.n_log_plots_steps = 40
    args.n_steps_eval = 10
    args.n_log_steps = 10
    args.batch_size = 8

# basic validation of args
if args.use_pruner and (args.pruner_n_steps is None or args.pruner_prob is None):
    raise ValueError(
        "use_pruner set but pruner_n_steps or pruner_prob or pruner_delay not set"
    )
if args.trainer_type == "retrain" and args.pruner_n_steps_retrain is None:
    raise ValueError("trainer_type is retrain but pruner_n_steps_retrain not set")
if args.trainer_type == "retrain" and not args.use_pruner:
    raise ValueError("trainer_type is retrain but use_pruner not set")
if not args.use_pruner and (
    args.pruner_n_steps is not None
    or args.pruner_prob is not None
    or args.pruner_delay > 0
):
    raise ValueError(
        "use_pruner not set but pruner_n_steps or pruner_prob or pruner_delay set"
    )

print("BEGINNING OF FILE")
print("cuda available:")
print(torch.cuda.is_available())

# constants
VOCAB_SIZE = 30522  # BertTokenizer uses this many words
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tags_to_name(tags: Optional[List[str]]) -> str:
    return "_".join(tags) if tags else ""


def make_concise_datetime() -> str:
    now = datetime.datetime.now()
    return str(now.year)[-2:] + "_" + now.strftime("%m-%d_%H:%M:%S")


timestamp = make_concise_datetime()
unique_timestamp = f"{timestamp}{secrets.token_urlsafe(1)}"

if args.use_clearml:
    task = Task.init(
        project_name=args.project_name,
        task_name=f"{args.name} {tags_to_name(args.tags)} {unique_timestamp}",
    )
    task.connect(vars(args))
    if args.tags:
        task.add_tags(args.tags)

modelpath = f"runs/wikibooktest/{unique_timestamp}"
writer = SummaryWriter(log_dir=modelpath)

# set pruner if needed
if args.use_pruner and args.pruner_n_steps:
    pruner = Pruner()
    scheduler = DelayedConstScheduler(
        pruner,
        args.pruner_n_steps,
        args.pruner_prob,
        args.pruner_delay,
        args.pruner_n_steps_retrain,
    )
else:
    pruner = None
    scheduler = None

# set ff layer
if args.ff_layer == "regular":
    ff_layer_fun = lambda: bert.FeedForward(args.dm, args.dff)
elif args.ff_layer == "unstruct_prune":
    ff_layer_fun = lambda: linears.UnstructPruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "struct_prune":
    ff_layer_fun = lambda: linears.StructPruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "unstruct_magnitude_prune":
    ff_layer_fun = lambda: linears.UnstructMagnitudePruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "struct_magnitude_prune":
    ff_layer_fun = lambda: linears.StructMagnitudePruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "unstruct_magnitude_recycle":
    ff_layer_fun = lambda: linears_recycle.UnstructMagnitudeRecycleFF(
        args.dm, args.dff, pruner
    )
elif args.ff_layer == "struct_magnitude_recycle":
    ff_layer_fun = lambda: linears_recycle.StructMagnitudeRecycleFF(
        args.dm, args.dff, pruner
    )
elif args.ff_layer == "retrain_recycle":
    ff_layer_fun = lambda: linears_recycle.RetrainRecycleFF(args.dm, args.dff, pruner)
elif args.ff_layer == "struct_magnitude_recycle_with_immunity":
    ff_layer_fun = lambda: linears_recycle.StructMagnitudeRecycleImmunityFF(
        args.dm, args.dff, pruner, args.immunity, args.reinit_dist
    )
elif args.ff_layer == "masked_ff":
    ff_layer_fun = linears.MaskedFF

misc.print_available_gpus()
pdataset = get_processed_dataset(
    batch_size=args.batch_size,
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
    num_workers=args.num_workers,
    seed=args.ds_seed,
)
eval_pdataset = get_processed_dataset(
    batch_size=args.batch_size,
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
    num_workers=1,
    seed=args.eval_ds_seed,
)

model = get_model(
    max_length=args.cutoff,
    vocab_size=VOCAB_SIZE,
    ff_layer_fun=ff_layer_fun,
    dm=args.dm,
    n_blocks=args.n_blocks,
    device=DEVICE,
    attention_layer_fun=lambda: bert.Attention(args.dm, args.heads, dhead=args.dhead),
)

# set optimizer
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

# dataset for neuron diff
if args.log_neuron_diff:
    pdataset_neuron_diff = get_processed_dataset(
        batch_size=args.batch_size,
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=DEVICE,
        num_workers=args.num_workers,
        seed=args.neuron_diff_ds_seed,
    )
else:
    pdataset_neuron_diff = None

if args.trainer_type == "retrain":
    pdataset_retrain = get_processed_dataset(
        batch_size=args.batch_size,
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=DEVICE,
        num_workers=args.num_workers,
        seed=args.retrain_ds_seed,
    )
    trainer = RetrainTrainer(
        model=model,
        optimizer=optimizer,
        pdataset=pdataset,
        pdataset_eval=eval_pdataset,
        batch_size=args.batch_size,
        vocab_size=VOCAB_SIZE,
        mask_percent=args.mask_percent,
        mask_loss_weight=args.mask_loss_weight,
        modelpath=modelpath,
        pruner=pruner,
        writer=writer,
        scheduler=scheduler,
        mixed_precision=args.mixed_precision,
        n_log_plots_steps=args.n_log_plots_steps,
        n_log_steps=args.n_log_steps,
        pdataset_retrain=pdataset_retrain,
        retrain_warmup_steps=args.retrain_warmup_steps,
        neuron_diff_dataset=pdataset_neuron_diff,
        neuron_diff_steps=args.log_neuron_diff_steps,
        neuron_diff_sample_size=args.log_neuron_diff_sample_size,
        neuron_diff_n_samples=args.log_neuron_diff_n_samples,
        neuron_diff_n_batches=args.neuron_diff_batches,
    )
else:
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        pdataset=pdataset,
        pdataset_eval=eval_pdataset,
        batch_size=args.batch_size,
        vocab_size=VOCAB_SIZE,
        mask_percent=args.mask_percent,
        mask_loss_weight=args.mask_loss_weight,
        modelpath=modelpath,
        pruner=pruner,
        writer=writer,
        scheduler=scheduler,
        mixed_precision=args.mixed_precision,
        n_log_steps=args.n_log_steps,
        neuron_diff_dataset=pdataset_neuron_diff,
        neuron_diff_steps=args.neuron_diff_steps,
        neuron_diff_sample_size=args.neuron_diff_sample_size,
        neuron_diff_n_samples=args.neuron_diff_n_samples,
    )

trainer.train(args.n_steps, args.n_steps_eval)
