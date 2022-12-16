import copy
from typing import Callable, Optional

from attr import define
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from lizrd.core import bert
from lizrd.datasets import wikibookdata
from lizrd.core.misc import are_state_dicts_the_same


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
    writer: Optional[SummaryWriter] = None
    mixed_precision: bool = False
    scaler: Optional[torch.cuda.amp.GradScaler] = None

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _train_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        pdataset: wikibookdata.ProcessedDatasetWrapper,
        step=0,
    ):
        model.train()
        processed_batch = pdataset.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask

        with torch.autocast(
            device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
        ):
            model_output = model(x_set)
            mask_loss = F.cross_entropy(
                model_output.reshape(-1, self.vocab_size),
                y_token_set.reshape(-1).long(),
                reduction="none",
            )
            mask_loss *= y_mask_set.reshape(-1)  # only check masked words
            mask_loss = mask_loss.mean() / self.mask_percent
            scaled_mask_loss = mask_loss * self.mask_loss_weight
            total_loss = scaled_mask_loss

        self.optimize(total_loss)

        if step and self.writer:
            self.writer.add_scalar("loss/train_total", total_loss.item(), step)
            self.writer.add_scalar("loss/train_mask", mask_loss.item(), step)

    def _eval_step(
        self,
        model: torch.nn.Module,
        pdataset: wikibookdata.ProcessedDatasetWrapper,
        step: int = 0,
        sample: int = 10,
    ):
        model.eval()

        with torch.no_grad():
            total_mask_loss = 0.0
            for _ in range(sample):
                processed_batch = pdataset.get_batch()
                assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
                x_set = processed_batch.masked_tokens
                y_token_set = processed_batch.tokens
                y_mask_set = processed_batch.mask_mask
                model_output = model(x_set)
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

            if step and self.writer:
                self.writer.add_scalar("loss/eval_mask", total_mask_loss, step)

            return total_mask_loss

    def train(self, n_steps: int, n_steps_eval: int):
        for step in range(n_steps):
            self._train_step(
                self.model, self.optimizer, self.pdataset, self.scheduler, step
            )
            self.writer.add_scalar("step", step, step)
            if step % n_steps_eval == 0:
                eval_loss = self._eval_step(self.model, self.pdataset_eval, step)
                print(f"Eval loss:", eval_loss)
                torch.save(self.model.state_dict(), f"{self.modelpath}/model.pt")
            print(f"Step {step}")
