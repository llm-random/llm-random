from typing import Callable, Optional

from attr import define
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from lizrd.core import bert
from lizrd.datasets import wikibookdata
from research.reinitialization.core.scheduler import BaseScheduler
from research.reinitialization.core.pruner import BasePruner


def get_model(
    max_length: int,
    vocab_size: int,
    ff_layer_fun: Callable[[], torch.nn.Module],
    attention_layer_fun: Callable[[], torch.nn.Module],
    dm: int,
    n_blocks: int,
    device: torch.device,
):
    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm), bert.TokenEmbedding(vocab_size, dm)
    )
    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        attention_layer_fun,
        ff_layer_fun,
    )
    head = bert.PredictionHead(dm, vocab_size)
    model = bert.BERT(embedding_layer, encoder_tower, head)

    # sanity check to make sure it works
    input = torch.randint(0, vocab_size, (16, 10))
    model(input)
    del input

    return model.to(device)


def get_processed_dataset(
    batch_size: int,
    max_total_length: int,
    mask_percent: float,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> wikibookdata.ProcessedDatasetWrapper:
    raw_dataset = wikibookdata.WikiBookDataset()
    processor = wikibookdata.SentenceProcessor(
        max_total_length=max_total_length,
        mask_percent=mask_percent,
    )
    dataset = wikibookdata.ProcessedDataset(raw_dataset, processor)
    return wikibookdata.ProcessedDatasetWrapper(
        pdataset=dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )


@define
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    pdataset: wikibookdata.ProcessedDatasetWrapper
    pdataset_eval: wikibookdata.ProcessedDatasetWrapper
    batch_size: int
    vocab_size: int
    mask_percent: float
    mask_loss_weight: float
    modelpath: str
    pruner: BasePruner
    scheduler: Optional[BaseScheduler] = None
    writer: Optional[SummaryWriter] = None
    mixed_precision: bool = False
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    log_n_steps: int = None
    eval_batches: int = 10

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

    def _pruning_step(self, step):
        if self.scheduler.is_time_to_prune(step):
            self.pruner.prune(self.scheduler.prob)

    def _train_step(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        self.model.train()
        processed_batch = self.pdataset.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask

        with torch.autocast(
            device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
        ):
            model_output = self.model(x_set)
            mask_loss = F.cross_entropy(
                model_output.reshape(-1, self.vocab_size),
                y_token_set.reshape(-1).long(),
                reduction="none",
            )
            mask_loss *= y_mask_set.reshape(-1)  # only check masked words
            mask_loss = mask_loss.mean() / self.mask_percent
            scaled_mask_loss = mask_loss * self.mask_loss_weight
            total_loss = scaled_mask_loss

        self.optimize(total_loss, optimizer)

        return total_loss.item(), mask_loss.item()

    def _log_train_stats(self, total_loss: float, mask_loss: float, step: int):
        self.writer.add_scalar("loss/train_total", total_loss, step)
        self.writer.add_scalar("loss/train_mask", mask_loss, step)

    def _eval_step(
        self,
        step,
        sample,
    ):
        self.model.eval()

        with torch.no_grad():
            total_mask_loss = 0.0
            for _ in range(sample):
                processed_batch = self.pdataset_eval.get_batch()
                assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
                x_set = processed_batch.masked_tokens
                y_token_set = processed_batch.tokens
                y_mask_set = processed_batch.mask_mask
                model_output = self.model(x_set)
                mask_loss = F.cross_entropy(
                    model_output.reshape(-1, self.vocab_size),
                    y_token_set.reshape(-1).long(),
                    reduction="none",
                )
                mask_loss *= y_mask_set.reshape(-1)  # only check masked words
                mask_loss = mask_loss.mean() / self.mask_percent
                scaled_mask_loss = mask_loss * self.mask_loss_weight
                total_mask_loss += scaled_mask_loss.item()
            total_mask_loss /= sample

            if self.writer:
                self.writer.add_scalar("loss/eval_mask", total_mask_loss, step)
            print(f"Eval loss:", total_mask_loss)
            torch.save(self.model.state_dict(), f"{self.modelpath}/model.pt")

    def train(self, n_steps: int, n_steps_eval: int):
        for step in range(n_steps):
            self._pruning_step(step)
            total_loss, mask_loss = self._train_step(self.optimizer)
            self._log_train_stats(total_loss, mask_loss, step)
            self.writer.add_scalar("step", step, step)
            if step % n_steps_eval == 0:
                self._eval_step(step, self.eval_batches)
            if self.log_n_steps and step % self.log_n_steps == 0:
                self.scheduler.pruner.log(step)
            print(f"Step {step}")


class SetLRTemporarily:
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def __enter__(self):
        self.original_lrs = []
        for param_group in self.optimizer.param_groups:
            self.original_lrs.append(param_group["lr"])
            param_group["lr"] = self.lr

    def __exit__(self, *args):
        for param_group, lr in zip(self.optimizer.param_groups, self.original_lrs):
            param_group["lr"] = lr


@define
class RetrainTrainer(Trainer):
    retrain_count: int = 0
    statistics_reset_steps: int = 0

    def _log_train_stats(self, total_loss: float, mask_loss: float, step: int):
        self.writer.add_scalar("loss/train_total", total_loss, step)
        self.writer.add_scalar("loss/train_mask", mask_loss, step)
        self.writer.add_scalar(
            "full_loss/train_total", total_loss, step + self.retrain_count
        )
        self.writer.add_scalar(
            "full_loss/train_mask", mask_loss, step + self.retrain_count
        )

    def _log_retrain_stats(self, total_loss: float, mask_loss: float, step: int):
        self.writer.add_scalar(
            "full_loss/train_total", total_loss, step + self.retrain_count
        )
        self.writer.add_scalar(
            "full_loss/train_mask", mask_loss, step + self.retrain_count
        )

    def _pruning_step(self, step):
        if self.scheduler.is_time_to_prune(step):
            self._retrain(step)

    def _retrain(self, step):
        self.pruner.prepare_new(self.scheduler.prob)

        # freeze model
        self.model.requires_grad_(False)

        # unfreeze new
        self.pruner.pre_retrain()

        # create retrain optimizer (without old stats)
        retrain_optim = torch.optim.Adam(
            self.model.parameters(),
            self.optimizer.param_groups[0]["lr"],
            self.optimizer.param_groups[0]["betas"],
            self.optimizer.param_groups[0]["eps"],
            self.optimizer.param_groups[0]["weight_decay"],
        )

        # gather statistics for new optimizer
        with SetLRTemporarily(retrain_optim, 0.0):
            for _ in range(self.statistics_reset_steps):
                self._train_step(retrain_optim)

        # retrain_optim.param_groups[0]["lr"] = 0.0
        # for _ in range(self.statistics_reset_steps):
        #     self._train_step(retrain_optim)
        retrain_optim.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"]

        # retrain
        for _ in range(self.scheduler.n_steps_retrain):
            self.retrain_count += 1
            total_loss, mask_loss = self._train_step(retrain_optim)
            self._log_retrain_stats(total_loss, mask_loss, step)

        # gather statistics for original optimizer
        # original_lr = self.optimizer.param_groups[0]["lr"]
        # self.optimizer.param_groups[0]["lr"] = 0.0
        self.pruner.apply_new_weights()

        with SetLRTemporarily(self.optimizer, 0.0):
            for _ in range(self.statistics_reset_steps):
                self._train_step(self.optimizer)

        # unfreeze model
        self.model.requires_grad_(True)
        self.pruner.post_retrain()
