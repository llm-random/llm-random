import argparse
import datetime
import os
import secrets
import time

import neptune.new as neptune
import torch
from clearml import Task
from torch.utils.tensorboard import SummaryWriter

from lizrd.core import misc
from lizrd.support.logging import NeptuneLogger, ClearMLLogger
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,
)
from research.nonlinearities.core import misc_logging
from research.nonlinearities.core.trainers import NonlinearityTrainer
from research.nonlinearities.train.utils import (
    get_ff_layer,
    get_attention_layer,
    make_concise_datetime,
)


def tags_to_name(tags) -> str:
    return "_".join(tags) if tags else ""


parser = argparse.ArgumentParser()

# core hyperparameters, fixed for all experiments; needs a good reason to change

parser.add_argument("--use_clearml", type=bool, default=False)
parser.add_argument("--use_neptune", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--cutoff", type=int, default=128)
parser.add_argument("--dmodel", type=int, default=256)
parser.add_argument("--dff", type=int, default=1024)
parser.add_argument("--n_att_heads", type=int, default=4)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--mixed_precision", type=bool, default=False)
parser.add_argument("--log_distributions", type=bool, default=False)
parser.add_argument("--logging_frequency", type=int, default=1000)
parser.add_argument("--mask_loss_weight", type=float, default=1.0)
parser.add_argument("--mask_percent", type=float, default=0.15)
parser.add_argument("--n_steps", type=int, default=100_001)
parser.add_argument("--seed", type=int, default=42)

# parameters usually changed for experiments

parser.add_argument("--ff_mode", type=str, default="vanilla")
parser.add_argument("--project_name", type=str, default="nonlinearities/initial_tests")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--tags", nargs="*", type=str, default=None)

# experimental/legacy parameters

parser.add_argument("--exp_rate", type=int, default=4)
parser.add_argument("--n_ff_heads", type=int, default=8)
parser.add_argument("--d_ff_head", type=int, default=256)
parser.add_argument("--n_chunks", type=int, default=4)
parser.add_argument("--multineck_mode", type=str, default="none")
parser.add_argument("--inception_head_sizes", nargs="*", type=float)
parser.add_argument("--dbottle", type=int, default=-1)
parser.add_argument("--n_bias_copies", type=int, default=-1)
parser.add_argument("--attention_mode", type=str, default="vanilla")
parser.add_argument("--attention_thinning_coeff", type=float, default=1.0)
parser.add_argument("--n_steps_eval", type=int, default=100)
parser.add_argument("--class_loss_weight", type=float, default=1.0)
parser.add_argument("--save_model_checkpoints", type=bool, default=False)
parser.add_argument("--deterministic", type=bool, default=True)


args = parser.parse_args()

VOCAB_SIZE = 30522  # BertTokenizer uses this many words
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
modelpath = f"runs/wikibooktest/{timestamp}"
writer = SummaryWriter(log_dir=modelpath)

ff_layer_fun = get_ff_layer(args)
attention_layer_fun = get_attention_layer(args)

misc.print_available_gpus()

train_dataloader = get_processed_dataset(
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    seed=args.seed,
)

model = get_model(
    max_length=args.cutoff,
    vocab_size=VOCAB_SIZE,
    ff_layer_fun=ff_layer_fun,
    attention_layer_fun=attention_layer_fun,
    dm=args.dmodel,
    n_blocks=args.n_blocks,
    device=DEVICE,
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
trainer = NonlinearityTrainer(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    mask_loss_weight=args.mask_loss_weight,
    modelpath=modelpath,
    writer=writer,
    mixed_precision=args.mixed_precision,
    save_model_checkpoints=args.save_model_checkpoints,
    distribution_logging=args.log_distributions,
    logging_frequency=args.logging_frequency,
)

dummy_ff_layer = ff_layer_fun()
single_ff_layer_parameter_count = misc_logging.get_parameter_count(dummy_ff_layer)
model_parameter_count = misc_logging.get_parameter_count(model)
del dummy_ff_layer

parameter_counts = {
    "single_ff_layer_parameter_count": single_ff_layer_parameter_count,
    "model_parameter_count": model_parameter_count,
}


if args.use_clearml:
    task = Task.init(
        project_name=args.project_name,
        task_name=f"{args.name}_{tags_to_name(args.tags)}",
    )
    task.connect(vars(args))
    task.connect(parameter_counts, "parameter_counts")
    if args.tags:
        task.add_tags(args.tags)
    if not args.deterministic:
        task.set_random_seed(int(time.time()))
    if args.seed:
        task.set_random_seed(args.seed)
    logger = ClearMLLogger(task)

timestamp = make_concise_datetime()
unique_timestamp = f"{timestamp}{secrets.token_urlsafe(1)}"

if args.use_neptune:
    run = neptune.init_run(
        project="pmtest/llm-efficiency",
        tags=args.tags,
        name=f"{args.name} {tags_to_name(args.tags)} {unique_timestamp}",
    )
    run["args"] = vars(args)
    run["working_directory"] = os.getcwd()
    run["git_branch"] = os.getcwd().split("/")[-1]
    logger = NeptuneLogger(run)

modelpath = f"models/{unique_timestamp}"
os.makedirs(modelpath, exist_ok=True)
trainer.train(args.n_steps)
