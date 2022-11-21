import copy
from typing import Callable, Optional

from attr import define
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from lizrd.core import bert
from lizrd.datasets import wikibookdata
from research.reinitialization.core.scheduler import BaseScheduler
from lizrd.core import misc
from research.reinitialization.core.pruner import BasePruner
from lizrd.core.misc import are_state_dicts_the_same


def get_model(
    max_length: int,
    vocab_size: int,
    ff_layer_fun: Callable[[], torch.nn.Module],
    dm: int,
    n_blocks: int,
    heads: int,
    device: torch.device,
):
    embedding_layer = bert.EmbeddingLayer(
        bert.PositionalEmbedding(max_length, dm), bert.TokenEmbedding(vocab_size, dm)
    )
    encoder_tower = bert.EncoderTower(
        n_blocks,
        dm,
        (lambda: bert.Attention(dm, heads)),
        ff_layer_fun,
    )
    head = bert.PredictionHead(dm, vocab_size)
    model = bert.BERT(embedding_layer, encoder_tower, head)

    # sanity check to make sure it works
    input = torch.randint(0, vocab_size, (16, 10))
    model(input)

    return model.to(device)


def get_processed_dataset(
    max_total_length: int, mask_percent: float, device: torch.device
):
    raw_dataset = wikibookdata.WikiBookDataset()
    processor = wikibookdata.SentencePairProcessor(
        max_total_length=max_total_length,
        device=device,
        mask_percent=mask_percent,
        swap_percent=0.0,
    )
    return wikibookdata.ProcessedDataset(raw_dataset, processor)


@define
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    pdataset: wikibookdata.ProcessedDataset
    batch_size: int
    vocab_size: int
    mask_percent: float
    mask_loss_weight: float
    modelpath: str
    scheduler: Optional[BaseScheduler] = None
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
        pdataset: wikibookdata.ProcessedDataset,
        scheduler: Optional[BaseScheduler],
        step=0,
    ):
        if scheduler:
            scheduler.step()
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
            self._train_step(
                self.model, self.optimizer, self.pdataset, self.scheduler, step
            )
            self.writer.add_scalar("step", step, step)
            if step % n_steps_eval == 0:
                eval_loss = self._eval_step(
                    self.model, self.pdataset, step, sample=n_steps_eval // 2
                )
                print(f"Eval loss:", eval_loss)
                torch.save(self.model.state_dict(), f"{self.modelpath}/model.pt")
            print(f"Step {step}")


@define
class LTHTrainer:
    model: torch.nn.Module
    optimizer_creator: Callable[[torch.nn.Module], torch.optim.Optimizer]
    pdataset_creator: Callable[[], wikibookdata.ProcessedDataset]
    pruner: BasePruner
    batch_size: int
    vocab_size: int
    mask_percent: float
    mask_loss_weight: float
    modelpath: str
    n_steps_per_run: int
    n_steps_eval: int
    writer: SummaryWriter
    pruning_rate: float
    target_params: float
    initial_model_path: Optional[str] = None

    def _save_model_params(self):
        self.initial_model_path = f"{self.modelpath}/init.pt"
        print(f'Saving initial model to "{self.initial_model_path}"')
        torch.save(self.model.state_dict(), self.initial_model_path)

    def _save_checkpoint(self, step):
        model_path = f"{self.modelpath}/{step}.pt"
        print(f'Saving checkpoint@{step} to "{model_path}"')
        torch.save(self.model.state_dict(), model_path)

    def _reinitialize_model(self):
        """Reinitialize the model to its original state without losing track of masks."""
        with torch.no_grad():
            masks = copy.deepcopy([layer.mask for layer in self.pruner.layers])
            model_state_dict = torch.load(self.initial_model_path)
            assert not are_state_dicts_the_same(
                self.model.state_dict(), model_state_dict
            )
            self.model.load_state_dict(model_state_dict)
            assert are_state_dicts_the_same(self.model.state_dict(), model_state_dict)
            for layer, mask in zip(self.pruner.layers, masks):
                layer.mask = mask
            assert not are_state_dicts_the_same(
                self.model.state_dict(), model_state_dict
            )

    def _log_masks_percentage(self, step):
        zeros = 0
        total = 0
        for layer in self.pruner.layers:
            zeros += torch.sum(layer.mask == 0).item()
            total += layer.mask.numel()
        self.writer.add_scalar("mask_percentage", zeros / total, step)

    def _train_step(
        self,
        optimizer: torch.optim.Optimizer,
        pdataset: wikibookdata.ProcessedDataset,
        step=0,
    ):
        self.model.train()
        processed_batch = pdataset.get_batch(self.batch_size)
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
        total_loss = scaled_mask_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step and self.writer:
            self.writer.add_scalar("loss/train_total", total_loss.item(), step)
            self.writer.add_scalar("loss/train_mask", mask_loss.item(), step)

    def _eval_step(
        self,
        pdataset: wikibookdata.ProcessedDataset,
        step: int = 0,
        sample: int = 10,
    ):
        self.model.eval()

        with torch.no_grad():
            total_mask_loss = 0.0
            for _ in range(sample):
                processed_batch = pdataset.get_batch(self.batch_size)
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

            self.writer.add_scalar("loss/eval_mask", total_mask_loss, step)
            print(f"Eval loss:", total_mask_loss)
            return total_mask_loss

    def train(self):
        self._save_model_params()
        parameters_left = 1.0
        total_step = 0
        while True:
            optimizer = self.optimizer_creator(self.model)
            pdataset = self.pdataset_creator()
            self.writer.add_scalar("parameters_left", parameters_left, total_step)
            for step in range(self.n_steps_per_run):
                self._train_step(optimizer, pdataset, total_step)
                if step % self.n_steps_eval == 0:
                    self._eval_step(
                        pdataset, step=total_step, sample=self.n_steps_eval // 2
                    )
                self.writer.add_scalar("total_step", total_step, total_step)
                print(f"Run step {step}; Total step {total_step}")
                total_step += 1
            # just in case parameters left is not exact
            self._save_checkpoint(total_step)
            self._log_masks_percentage(total_step)
            self.pruner.step(parameters_left * self.pruning_rate)
            if parameters_left < self.target_params:
                break
            parameters_left *= 1 - self.pruning_rate
            self._reinitialize_model()
