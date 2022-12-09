from typing import Optional

from attr import define
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from lizrd.datasets import wikibookdata


@define
class NonlinearityTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    pdataset: wikibookdata.ProcessedDataset
    batch_size: int
    vocab_size: int
    mask_percent: float
    mask_loss_weight: float
    modelpath: str
    save_model_checkpoints: str
    mixed_precision: bool = False
    writer: Optional[SummaryWriter] = None
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
        pdataset: wikibookdata.ProcessedDataset,
        step=0,
    ):

        model.train()
        processed_batch = pdataset.get_batch(self.batch_size)
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
        pdataset: wikibookdata.ProcessedDataset,
        step: int = 0,
        sample: int = 10,
    ):
        model.eval()

        with torch.no_grad():
            total_mask_loss = 0.0
            for _ in range(sample):
                processed_batch = pdataset.get_batch(self.batch_size)
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
            self._train_step(self.model, self.optimizer, self.pdataset, step)
            self.writer.add_scalar("step", step, step)
            if step % n_steps_eval == 0:
                eval_loss = self._eval_step(
                    self.model, self.pdataset, step, sample=n_steps_eval // 2
                )
                print(f"Eval loss:", eval_loss)
                if self.save_model_checkpoints:
                    torch.save(self.model.state_dict(), f"{self.modelpath}/model.pt")
            print(f"Step {step}")
