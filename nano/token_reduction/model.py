import math
import os
import re
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from attr import define
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting
import torch.distributed as dist
from dataclasses import dataclass
from model import (
    Common,
    EmbeddingLayer,
    Linear,
    PositionalEmbedding,
    TokenEmbedding,
    Trainer,
    create_batch_fingerprint,
    TowerConfig,
    TransformerTower,
    BlockConfig,
    TransformerBlock,
    PredictionHead,
    RMSNorm,
)

logger = logging.getLogger(__name__)


def split_indexes(result_seq_len, n_tokens_to_reduce):
    available_range = result_seq_len + n_tokens_to_reduce
    permutation = torch.randperm(available_range)

    return permutation[:result_seq_len].sort().values, permutation[result_seq_len:]


def batched_split_indexes(batch_size, seq_len, result_seq_len, n_tokens_to_reduce):
    split_tuples = [
        split_indexes(result_seq_len, n_tokens_to_reduce) for _ in range(batch_size)
    ]
    tuple_of_splits = list(zip(*split_tuples))
    stacked_split = [torch.stack(t) for t in tuple_of_splits]

    return stacked_split


def flatten_with_indices_adjustment(indices, seq_len):
    """
    This function flattens the indices and adjusts indices to match indices in flattened tensor.
    """
    batch_size = indices.shape[0]
    indices = indices + torch.arange(batch_size).unsqueeze(1) * seq_len
    return indices.flatten()


def batch_index_select(inputs, indexes):
    return inputs[torch.arange(inputs.size(0)).unsqueeze(-1), indexes]


class CommonDroppingConfig(Common):
    dropped_tokens: int


class TokenDroppingEmbedding(torch.nn.Module):
    def __init__(self, normal_embedding):
        super().__init__()
        self.normal_embedding = normal_embedding

    def forward(self, x, indexes):
        x = self.normal_embedding(x)
        if self.training:
            x = batch_index_select(x, indexes)
        return x


def create_token_dropping_function(config, common: CommonDroppingConfig):
    normal_embedding = EmbeddingLayer(
        *[
            TokenEmbedding(
                common.vocab_size,
                common.dmodel,
                init_type=common.init_type,
                init_scale=common.init_scale,
            ),
            PositionalEmbedding(
                common.sequence_length + common.dropped_tokens,
                common.dmodel,
                init_type=common.init_type,
                init_scale=common.init_scale,
            ),
        ]
    )
    return TokenDroppingEmbedding(normal_embedding)


@dataclass
class MTPConfig:
    n_mtp: int
    mtp_block_config: BlockConfig


class LLM_MTP(nn.Module):
    def __init__(
        self,
        embedding,
        common: Common,
        tower_config: TowerConfig,
        mtp_config: MTPConfig,
    ):
        super().__init__()
        self.n_mtp = mtp_config.n_mtp

        self.embedding_layer = embedding

        tower_config.n_blocks -= 1  # MTP heads are de facto last encoder layer.
        self.encoder = TransformerTower(
            common=common,
            tower_config=tower_config,
        )

        self.mtp_modules = nn.ModuleList(
            [
                TransformerBlock(common, mtp_config.mtp_block_config)
                for _ in range(mtp_config.n_mtp + 1)
            ]
        )

        self.head = PredictionHead(
            common.dmodel,
            common.vocab_size,
            init_type=common.init_type,
            init_scale=common.init_scale,
            use_layer_norm=common.head_norm,
        )

        self._add_metric_log_names()

    def _add_metric_log_names(self):
        def _get_metric_log_name(name: str):
            meaningful_regex = ["block_\\d+", "attention", "feedforward", "residual"]
            module_names = name.split(".")
            meaningful_names = [
                module_name
                for module_name in module_names
                if any(re.search(pattern, module_name) for pattern in meaningful_regex)
            ]
            return "/".join(meaningful_names)

        for name, model in self.named_modules():
            model.log_name = _get_metric_log_name(name)

    def forward(self, *args, **kwargs):
        embedding_out = self.embedding_layer(*args, **kwargs)
        encoder_out = self.encoder(embedding_out)
        logits_list = []
        if self.training:
            for i in range(self.n_mtp + 1):
                mtp_module_out = self.mtp_modules[i](
                    encoder_out,
                )
                mtp_logits = self.head(mtp_module_out)
                logits_list.append(mtp_logits)
        else:
            mtp_module_out = self.mtp_modules[0](
                encoder_out,
            )
            mtp_logits = self.head(mtp_module_out)
            logits_list.append(mtp_logits)
        return logits_list


class DeepSeekMTPHead(nn.Module):
    def __init__(
        self,
        common: Common,
        block_config: BlockConfig,
    ):
        super().__init__()
        self.norm_embedding = RMSNorm(dmodel=common.dmodel)
        self.norm_tt = RMSNorm(dmodel=common.dmodel)
        self.proj = Linear(
            in_features=2 * common.dmodel,
            out_features=common.dmodel,
            bias=False,
            init_type=common.init_type,
            init_scale=common.init_scale,
        )
        self.mtp_block = TransformerBlock(
            common=common,
            block_config=block_config,
        )

    def forward(self, embedding_out, tt_out):
        embedding_out = self.norm_embedding(embedding_out)
        tt_out = self.norm_tt(tt_out)
        out = torch.concat((tt_out, embedding_out), dim=-1)
        out = self.proj(out)
        out = self.mtp_block(out)
        return out


class LLM_DeepSeekMTP(nn.Module):
    def __init__(
        self,
        embedding,
        common: Common,
        tower_config: TowerConfig,
        mtp_config: MTPConfig,
    ):
        super(LLM_DeepSeekMTP, self).__init__()
        self.n_mtp = mtp_config.n_mtp

        self.embedding_layer = embedding

        self.encoder = TransformerTower(
            common=common,
            tower_config=tower_config,
        )

        self.mtp_modules = nn.ModuleList(
            [
                DeepSeekMTPHead(common, mtp_config.mtp_block_config)
                for _ in range(mtp_config.n_mtp)
            ]
        )

        self.head = PredictionHead(
            common.dmodel,
            common.vocab_size,
            init_type=common.init_type,
            init_scale=common.init_scale,
            use_layer_norm=common.head_norm,
        )

        self._add_metric_log_names()

    def _add_metric_log_names(self):
        def _get_metric_log_name(name: str):
            meaningful_regex = ["block_\\d+", "attention", "feedforward", "residual"]
            module_names = name.split(".")
            meaningful_names = [
                module_name
                for module_name in module_names
                if any(re.search(pattern, module_name) for pattern in meaningful_regex)
            ]
            return "/".join(meaningful_names)

        for name, model in self.named_modules():
            model.log_name = _get_metric_log_name(name)

    def forward(self, *args, **kwargs):
        # args[0]: [bsz, seq_len + n_dropped + n_mtp, dmodel]
        embedding_out = self.embedding_layer(*args, **kwargs)
        # Embedding out shoud be: [batch_size, seq_len + n_mtp, dmodel]
        seq_len = embedding_out.shape[1] - self.n_mtp
        if self.training:
            transformer_tower_input = embedding_out[:, :seq_len]
        else:
            transformer_tower_input = embedding_out
        transformer_tower_out = self.encoder(transformer_tower_input)
        main_head_logits = self.head(transformer_tower_out)
        logits_list = [main_head_logits]
        if self.training:
            for i in range(self.n_mtp):
                mtp_embedding_input = embedding_out[:, (i + 1) : ((i + 1) + seq_len)]
                transformer_tower_out = self.mtp_modules[i](
                    mtp_embedding_input,
                    transformer_tower_out,
                )
                mtp_logits = self.head(transformer_tower_out)
                logits_list.append(mtp_logits)
        return logits_list


def get_deepseek_embedding(common, n_mtp):
    return EmbeddingLayer(
        TokenEmbedding(
            common.vocab_size,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        PositionalEmbedding(
            common.sequence_length + n_mtp,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
    )


class TokenMergingEmbedding(torch.nn.Module):
    def __init__(self, normal_embedding, common: CommonDroppingConfig):
        super().__init__()
        self.normal_embedding = normal_embedding
        self.linear = Linear(
            common.dmodel,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        )

    def forward(self, x, keep_indexes=None, merge_indexes=None):
        x = self.normal_embedding(x)
        if self.training:
            merge_tokens = batch_index_select(x, merge_indexes)
            merge_tokens = self.linear(merge_tokens)

            # It can happend that if we pick for merge last token from sequence, we do not have next token to merge it with, so we add zero vector
            x = F.pad(x, (0, 0, 0, 1), value=0)

            x[
                torch.arange(merge_indexes.size(0)).unsqueeze(-1), merge_indexes + 1
            ] += merge_tokens

            x = batch_index_select(x, keep_indexes)
        return x


class TokenMergingEmbeddingBothTokens(torch.nn.Module):
    def __init__(self, normal_embedding, common: CommonDroppingConfig):
        super().__init__()
        self.normal_embedding = normal_embedding
        self.linear = Linear(
            common.dmodel * 2,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        )

    def forward(self, x, keep_indexes, merge_indexes):
        x = self.normal_embedding(x)
        if self.training:
            # It can happend that if we pick for merge last token from sequence, we do not have next token to merge it with, so we add zero vector
            x = F.pad(x, (0, 0, 0, 1), value=0)

            merge_tokens_a = batch_index_select(x, merge_indexes)
            merge_tokens_b = batch_index_select(x, merge_indexes + 1)
            merge_tokens = torch.cat((merge_tokens_a, merge_tokens_b), dim=-1)

            merge_tokens = self.linear(merge_tokens)
            x[torch.arange(merge_indexes.size(0)).unsqueeze(-1), merge_indexes + 1] = (
                merge_tokens
            )

            x = batch_index_select(x, keep_indexes)
        return x


def create_token_merging_function(config, common: CommonDroppingConfig):
    normal_embedding = EmbeddingLayer(
        *[
            TokenEmbedding(
                common.vocab_size,
                common.dmodel,
                init_type=common.init_type,
                init_scale=common.init_scale,
            ),
            PositionalEmbedding(
                common.sequence_length + common.dropped_tokens,
                common.dmodel,
                init_type=common.init_type,
                init_scale=common.init_scale,
            ),
        ]
    )
    return TokenMergingEmbedding(normal_embedding, common)


class DroppingTrainer(Trainer):
    def calculate_loss(self, batch):
        def _hack_for_python_garbage_collection(input_ids, target_ids, keep_indexes):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            predicted_ids = self.model(input_ids, keep_indexes)

            # Tensors should be on the same device for loss calculation #TODO check maybe it should be exactly self.device
            target_ids = target_ids.to(predicted_ids.device)

            mask_loss = F.cross_entropy(
                predicted_ids.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            loss = mask_loss.mean()
            return loss

        losses = []
        if self.model.training:
            input_data, (keep_indexes, _dropped) = batch

            for batch_chunk, keep_indexes_chunk in zip(
                input_data.chunk(self.gradient_accumulation_steps),
                keep_indexes.chunk(self.gradient_accumulation_steps),
            ):
                input_ids, target_ids = self._preprocess_input(batch_chunk)
                target_ids = batch_index_select(target_ids, keep_indexes_chunk)

                loss = _hack_for_python_garbage_collection(
                    input_ids, target_ids, keep_indexes_chunk
                )
                if self.model.training:
                    loss.backward()

                losses.append(loss.item())

                if self.model.training:
                    self._update_processed_tokens(input_ids)
        else:
            for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
                input_ids, target_ids = self._preprocess_input(batch_chunk)
                input_ids = input_ids.to(self.device)

                loss = _hack_for_python_garbage_collection(input_ids, target_ids, None)
                losses.append(loss.item())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_loss = torch.tensor(losses, device=loss.device).mean()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

        return avg_loss / float(os.environ["WORLD_SIZE"])


class MergingTrainer(Trainer):
    def calculate_loss(self, batch):
        def _hack_for_python_garbage_collection(
            input_ids, target_ids, keep_indexes, drop_indexes
        ):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            predicted_ids = self.model(input_ids, keep_indexes, drop_indexes)

            # Tensors should be on the same device for loss calculation #TODO check maybe it should be exactly self.device
            target_ids = target_ids.to(predicted_ids.device)

            mask_loss = F.cross_entropy(
                predicted_ids.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            loss = mask_loss.mean()
            return loss

        losses = []
        if self.model.training:
            input_data, (keep_indexes, drop_indexes) = batch
            for batch_chunk, keep_indexes_chunk, drop_indexes_chunk in zip(
                input_data.chunk(self.gradient_accumulation_steps),
                keep_indexes.chunk(self.gradient_accumulation_steps),
                drop_indexes.chunk(self.gradient_accumulation_steps),
            ):
                input_ids, target_ids = self._preprocess_input(batch_chunk)
                target_ids = batch_index_select(target_ids, keep_indexes_chunk)

                loss = _hack_for_python_garbage_collection(
                    input_ids, target_ids, keep_indexes_chunk, drop_indexes_chunk
                )
                if self.model.training:
                    loss.backward()

                losses.append(loss.item())

                if self.model.training:
                    self._update_processed_tokens(input_ids)
        else:
            for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
                input_ids, target_ids = self._preprocess_input(batch_chunk)
                input_ids = input_ids.to(self.device)

                loss = _hack_for_python_garbage_collection(
                    input_ids, target_ids, None, None
                )
                losses.append(loss.item())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_loss = torch.tensor(losses, device=loss.device).mean()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

        return avg_loss / float(os.environ["WORLD_SIZE"])


class TrainerMTP(Trainer):
    def _update_processed_tokens(self, batch, *_args):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def prepare_input_output(self, batch):
        if self.model.training:
            input_ids = [batch[:, : -(self.model.n_mtp + 1)].to(self.device)]
            mtp_target_ids = [
                batch[
                    :,
                    (i + 1) : (-self.model.n_mtp + i if i < self.model.n_mtp else None),
                ]
                for i in range(self.model.n_mtp + 1)
            ]
        else:
            input_ids = [batch[:, :-1]]
            mtp_target_ids = [batch[:, 1:]]
        return input_ids, mtp_target_ids

    def hack_for_python_garbage_collection(self, batch):
        """we want to have no reference to model output while backpropagating to allow torch to free memory,
        so we wrap loss calculation in a function"""

        (model_input, mtp_target_ids) = self.prepare_input_output(batch)
        if self.model.training:
            self._update_processed_tokens(*model_input)
        mtp_outputs = self.model(*model_input)

        mtp_losses = []
        for predicted_ids, target_ids in zip(mtp_outputs, mtp_target_ids):
            target_ids = target_ids.to(self.device)
            mask_loss = F.cross_entropy(
                predicted_ids.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            loss = mask_loss.mean() / self.gradient_accumulation_steps
            mtp_losses.append(loss)

        return mtp_losses

    def calculate_loss(self, batch):
        losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            mtp_losses = self.hack_for_python_garbage_collection(batch_chunk)

            if self.model.training:
                loss = torch.stack(mtp_losses).sum()
                loss.backward()
            losses.append(mtp_losses)

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_mtp_losses = torch.tensor(losses, device=mtp_losses[0].device).sum(dim=0)
        if dist.is_initialized():
            dist.all_reduce(avg_mtp_losses, op=dist.ReduceOp.SUM)

        return avg_mtp_losses / float(os.environ["WORLD_SIZE"])

    def eval(self):
        self.model.eval()
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        eval_fingerprint = []
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                batch_fingerprint = create_batch_fingerprint(batch)
                eval_fingerprint.extend(batch_fingerprint)
                batch = batch.to(self.device)
                mtp_losses = self.calculate_loss(batch).float()
                losses.append(mtp_losses)
                self.metric_logger.flush_accumulated_metrics(self.step)
            avg_loss = torch.stack(losses).mean(dim=0)
            self.metric_logger.log("steps/eval/loss", self.step, avg_loss[0].item())
            self.metric_logger.log(
                "tokens/eval/loss", self.processed_tokens, avg_loss[0].item()
            )

        if self._should_log_eval_input:
            self.metric_logger.log(
                f"steps/eval/batch", self.step, str(eval_fingerprint)
            )

    def log_metrics(self, mtp_losses, grad_norm):
        super().log_metrics(mtp_losses[0], grad_norm)
        for i, mtp_loss in enumerate(mtp_losses):
            self.metric_logger.log(
                f"steps/train/mtp_loss_{i}", self.step, mtp_loss.item()
            )
            self.metric_logger.log(
                f"tokens/train/mtp_loss_{i}", self.processed_tokens, mtp_loss.item()
            )

        # log average loss per 100 steps
        if self.step > 0:
            self.loss_interval_100 += mtp_losses[0].item()
            if self.step % 100 == 0:
                self.metric_logger.log(
                    "steps/train/loss_100", self.step, self.loss_interval_100 / 100.0
                )
                self.loss_interval_100 = 0.0


@define(slots=False)
class TrainerDeepSeekMTP(TrainerMTP):
    mtp_lambda: float

    def prepare_input_output(self, batch):
        if self.model.training:
            input_ids = [batch[:, :-1].to(self.device)]
            mtp_target_ids = [
                batch[
                    :,
                    (i + 1) : (-self.model.n_mtp + i if i < self.model.n_mtp else None),
                ]
                for i in range(self.model.n_mtp + 1)
            ]
        else:
            input_ids = [batch[:, :-1]]
            mtp_target_ids = [batch[:, 1:]]
        return input_ids, mtp_target_ids

    def calculate_loss(self, batch):
        losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            mtp_losses = self.hack_for_python_garbage_collection(batch_chunk)

            if self.model.training:
                if len(mtp_losses) > 1:
                    mtp_loss_weight = self.mtp_lambda / (len(mtp_losses) - 1)
                    loss_multiplier = torch.tensor(
                        [1.0] + [mtp_loss_weight] * (len(mtp_losses) - 1),
                        device=self.device,
                    )
                    mtp_losses_scaled = torch.stack(mtp_losses) * loss_multiplier
                    loss = mtp_losses_scaled.sum()
                    loss.backward()
                else:
                    mtp_losses[0].backward()

            losses.append(mtp_losses)

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_mtp_losses = torch.tensor(losses, device=mtp_losses[0].device).sum(dim=0)
        if dist.is_initialized():
            dist.all_reduce(avg_mtp_losses, op=dist.ReduceOp.SUM)

        return avg_mtp_losses / float(os.environ["WORLD_SIZE"])


def collate_reduction(batch, result_seq_len, n_dropped_tokens):
    batch = torch.tensor(batch)
    batch_size, seq_len = batch.shape
    return (
        batch,
        batched_split_indexes(batch_size, seq_len, result_seq_len, n_dropped_tokens),
    )


def get_dropping_standard_embedding(
    vocab_size, dmodel, init_type, init_scale, sequence_length, reduction_tokens
):
    return EmbeddingLayer(
        TokenEmbedding(
            vocab_size,
            dmodel,
            init_type,
            init_scale,
        ),
        PositionalEmbedding(
            sequence_length + reduction_tokens,
            dmodel,
            init_type,
            init_scale,
        ),
    )


class ReductionScheduler:
    def __init__(self, schedule_config, total_steps):
        self.schedule_config = schedule_config
        self.total_steps = total_steps
        self._validate_config()

    def _validate_config(self):
        if not self.schedule_config:
            raise ValueError("Schedule config cannot be empty")

        percentages = [phase.get("percentage", 1.0) for phase in self.schedule_config]

        if abs(sum(percentages) - 1.0) > 1e-6:
            raise ValueError("Phase percentages must sum to 1.0")
        if any(p < 0 or p > 1.0 for p in percentages):
            raise ValueError("Phase percentages must be between 0.0 and 1.0")

    def get_value(self, step):
        cumulative_percentage = 0.0

        for phase in self.schedule_config:
            percentage = phase.get("percentage", 1.0)
            start_percentage = cumulative_percentage
            end_percentage = cumulative_percentage + percentage
            cumulative_percentage = end_percentage

            # Convert percentages to step counts
            start_step = start_percentage * self.total_steps
            end_step = end_percentage * self.total_steps
            duration = end_step - start_step

            if step > end_step:
                continue
            elif step <= end_step:
                t = (step - start_step) / duration if duration > 0 else 0.0
                start_scale = phase.get("start_scale", 1.0)
                end_scale = phase.get("end_scale", 1.0)

                if phase["type"] == "constant":
                    return start_scale
                elif phase["type"] == "linear":
                    return start_scale + t * (end_scale - start_scale)
                elif phase["type"] == "cosine":
                    cosine = 0.5 * (1 + math.cos(math.pi * t))
                    return end_scale + (start_scale - end_scale) * cosine

        return self.schedule_config[-1].get("end_scale", 1.0)


@define(slots=False)
class TrainerMTPMerge(TrainerMTP):
    sequence_length: int
    n_reduced_tokens: int
    token_reducing_scheduler: ReductionScheduler = None

    def _get_n_tokens_to_reduce(self):
        if self.token_reducing_scheduler is not None:
            scaler = self.token_reducing_scheduler.get_value(self.step)
            return round(scaler * self.n_reduced_tokens)
        return self.n_reduced_tokens

    def prepare_input_output(self, batch):
        if self.model.training:
            input_ids = [batch[:, : -(self.model.n_mtp + 1)].to(self.device)]
            n_tokens_to_reduce = self._get_n_tokens_to_reduce()
            keep_pos_ids, reduce_pos_ids = batched_split_indexes(
                batch.shape[0], None, self.sequence_length, n_tokens_to_reduce
            )
            mtp_target_ids = [
                batch_index_select(batch, keep_pos_ids + 1 + i)
                for i in range(self.model.n_mtp + 1)
            ]
            input_ids.extend([keep_pos_ids, reduce_pos_ids])
        else:
            input_ids = [batch[:, :-1]]
            mtp_target_ids = [batch[:, 1:]]
        return input_ids, mtp_target_ids

    def log_metrics(self, loss, grad_norm):
        super().log_metrics(loss, grad_norm)
        self.metric_logger.log(
            "steps/train/n_reduced_tokens", self.step, self._get_n_tokens_to_reduce()
        )


@define(slots=False)
class TrainerDeepSeekMTPMerge(TrainerDeepSeekMTP):
    sequence_length: int
    n_reduced_tokens: int
    token_reducing_scheduler: ReductionScheduler = None

    def _get_n_tokens_to_reduce(self):
        if self.token_reducing_scheduler is not None:
            scaler = self.token_reducing_scheduler.get_value(self.step)
            return round(scaler * self.n_reduced_tokens)
        return self.n_reduced_tokens

    def prepare_input_output(self, batch):
        if self.model.training:
            input_ids = [batch[:, :-1].to(self.device)]
            n_tokens_to_reduce = self._get_n_tokens_to_reduce()

            keep_pos_ids, reduce_pos_ids = batched_split_indexes(
                batch.shape[0], None, self.sequence_length, n_tokens_to_reduce
            )
            mtp_target_ids = [
                batch_index_select(batch, keep_pos_ids + 1 + i)
                for i in range(self.model.n_mtp + 1)
            ]

            start_mtp_indexes = self.sequence_length + n_tokens_to_reduce
            end_mtp_indexes = start_mtp_indexes + self.model.n_mtp
            mtp_indexes = torch.tensor(
                range(start_mtp_indexes, end_mtp_indexes)
            ).repeat(len(keep_pos_ids), 1)

            keep_pos_ids = torch.cat((keep_pos_ids, mtp_indexes), dim=1)

            input_ids.extend([keep_pos_ids, reduce_pos_ids])
        else:
            input_ids = [batch[:, :-1]]
            mtp_target_ids = [batch[:, 1:]]

        return input_ids, mtp_target_ids

    def log_metrics(self, loss, grad_norm):
        super().log_metrics(loss, grad_norm)
        self.metric_logger.log(
            "steps/train/n_reduced_tokens", self.step, self._get_n_tokens_to_reduce()
        )


@define(slots=False)
class TrainerMerge(Trainer):
    sequence_length: int
    n_reduced_tokens: int
    token_reducing_scheduler: ReductionScheduler = None

    def _update_processed_tokens(self, batch, *_args):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def _get_n_tokens_to_reduce(self):
        if self.token_reducing_scheduler is not None:
            scaler = self.token_reducing_scheduler.get_value(self.step)
            return round(scaler * self.n_reduced_tokens)
        return self.n_reduced_tokens

    def prepare_input_output(self, batch):
        if self.model.training:
            input_ids = [batch[:, :-1].to(self.device)]
            n_tokens_to_reduce = self._get_n_tokens_to_reduce()

            keep_pos_ids, reduce_pos_ids = batched_split_indexes(
                batch.shape[0], None, self.sequence_length, n_tokens_to_reduce
            )
            target_ids = batch_index_select(batch, keep_pos_ids + 1)
            input_ids.extend([keep_pos_ids, reduce_pos_ids])
        else:
            input_ids = [batch[:, :-1]]
            target_ids = batch[:, 1:]

        return input_ids, target_ids

    def log_metrics(self, loss, grad_norm):
        super().log_metrics(loss, grad_norm)
        self.metric_logger.log(
            "steps/train/n_reduced_tokens", self.step, self._get_n_tokens_to_reduce()
        )

    def hack_for_python_garbage_collection(self, batch):
        """we want to have no reference to model output while backpropagating to allow torch to free memory,
        so we wrap loss calculation in a function"""

        (model_input, target_ids) = self.prepare_input_output(batch)
        if self.model.training:
            self._update_processed_tokens(*model_input)
        predicted_ids = self.model(*model_input)

        target_ids = target_ids.to(self.device)
        mask_loss = F.cross_entropy(
            predicted_ids.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        loss = mask_loss.mean()

        return loss

    def calculate_loss(self, batch):
        losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            loss = self.hack_for_python_garbage_collection(batch_chunk)
            if self.model.training:
                loss.backward()
            losses.append(loss)

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_mtp_losses = torch.tensor(losses, device=self.device).mean(dim=0)
        if dist.is_initialized():
            dist.all_reduce(avg_mtp_losses, op=dist.ReduceOp.SUM)

        return avg_mtp_losses / float(os.environ["WORLD_SIZE"])


