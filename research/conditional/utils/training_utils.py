import gc

import torch
from lizrd.datasets.wikibookdata import get_processed_dataset

from lizrd.train.train_utils import get_model
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.model_utils import get_ff_layer, get_attention_layer


def find_optimal_grad_accumulation(args, vocab_size, device):
    """
    Find the optimal number of gradient accumulation steps for a given model.
    NO SUPPORT FOR DISTRIBUTED TRAINING.
    """
    n_grad_acc_steps = 1
    model_fits_in_memory = False
    trainer, model, optimizer = None, None, None
    while True:
        try:
            print(f"Trying {n_grad_acc_steps} grad steps...")
            trainer, model, optimizer = get_trainer(
                args, vocab_size, device, n_grad_acc_steps
            )
            model_fits_in_memory = True
            trainer.train(10)
            break
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise Exception(f"Unknown error: {e}")
            n_grad_acc_steps += 1
        finally:
            if not model_fits_in_memory:
                raise Exception(
                    "Model does not fit in memory. No accumulation of gradients is possible."
                )
            torch.cuda.empty_cache()
            del trainer, model, optimizer
            gc.collect()
    print(f"Found optimal value of gradient accumulation: {n_grad_acc_steps}.")
    return n_grad_acc_steps


def get_trainer(args, vocab_size, device, grad_acc_steps):
    train_dataloader = get_processed_dataset(
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.data_seed,
        model_type=args.model_type,
        data_distributed=False,
    )
    ff_layer_fun = get_ff_layer(args)
    attention_layer_fun = get_attention_layer(args)

    model = get_model(
        max_length=args.cutoff,
        vocab_size=vocab_size,
        ff_layer_fun=ff_layer_fun,
        attention_layer_fun=attention_layer_fun,
        dm=args.dmodel,
        n_blocks=args.n_blocks,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = ConditionalTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        vocab_size=vocab_size,
        mask_percent=args.mask_percent,
        mixed_precision=args.mixed_precision,
        logger=None,
        hack_name=args.hack_name,
        model_type=args.model_type,
        logging_interval_loss=args.logging_interval_loss,
        logging_interval_light=args.logging_interval_light,
        logging_interval_heavy=args.logging_interval_heavy,
        n_gpus=args.n_gpus,
        loss_checkpoint_chungs=args.loss_checkpoint_chungs,
        gradient_accumulation_steps=grad_acc_steps,
    )
    return trainer, model, optimizer
