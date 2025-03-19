from functools import partial
import os
import re
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting
from torchdata.stateful_dataloader import StatefulDataLoader
import torch.distributed as dist
from dataclasses import dataclass
from model import (
    C4Dataset,
    Common,
    EmbeddingLayer,
    Linear,
    PositionalEmbedding,
    TokenEmbedding,
    Trainer,
    get_dataloader,
    TowerConfig,
    TransformerTower,
    BlockConfig,
    TransformerBlock,
    PredictionHead,
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
        super(LLM_MTP, self).__init__()

        self.embedding_layer = embedding

        self.encoder = TransformerTower(
            common=common,
            tower_config=tower_config,
        )

        self.mtp_modules = nn.ModuleList(
            [TransformerBlock(common, mtp_config.mtp_block_config) for _ in range(mtp_config.n_mtp)]
        )

        self.head = PredictionHead(
            common.dmodel,
            common.vocab_size,
            init_type=common.init_type,
            init_scale=common.init_scale,
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
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        return x


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

    def forward(self, x, keep_indexes, merge_indexes):
        x = self.normal_embedding(x)
        if self.training:
            merge_tokens = batch_index_select(x, merge_indexes)
            merge_tokens = self.linear(merge_tokens)

            # It can happend that if we pick for merge last token from sequence, we do not have next token to merge it with, so we add zero vector
            batch_size, _, dmodel = x.shape
            additional = torch.zeros(batch_size, 1, dmodel, device=x.device)
            x = torch.cat([x, additional], dim=1)

            x[
                torch.arange(merge_indexes.size(0)).unsqueeze(-1), merge_indexes + 1
            ] += merge_tokens

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

class MTPTrainer(Trainer):
    n_mtp: int

    def _preprocess_input_mtp(self, batch):  # TODO test it
        input_ids = batch[:, :-self.n_mtp].contiguous()
        target_ids = batch[:, 1:].contiguous()

        return input_ids, target_ids

    def calculate_loss(self, batch):

        def _hack_for_python_garbage_collection(input_ids, target_ids):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            encoder_embeddings = self.model(input_ids)
            encoder_embeddings_detatched = encoder_embeddings.detach()

            # Tensors should be on the same device for loss calculation #TODO check
            target_ids = target_ids.to(encoder_embeddings.device)

            for i in range(self.n_mtp):
                mtp_module_output = self.model.mtp_modules[i](encoder_embeddings_detatched)
                predicted_ids = self.model.head(mtp_module_output)
                mtp_loss = F.cross_entropy(
                    predicted_ids.flatten(0, -2),
                    target_ids[:, i:i + 1 - self.n_mtp].reshape(-1).long(),
                    reduction="none",
                )
                mtp_loss.backward()
            loss = mask_loss.mean() / self.gradient_accumulation_steps
            return loss

        losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            input_ids, target_ids = self._preprocess_input_mtp(batch_chunk)
            input_ids = input_ids.to(self.device)
            if self.model.training:
                self._update_processed_tokens(input_ids)

            loss = _hack_for_python_garbage_collection(input_ids, target_ids)
            if self.model.training:
                loss.backward()
            losses.append(loss.item())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_loss = torch.tensor(losses, device=loss.device).sum()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

        return avg_loss / float(os.environ["WORLD_SIZE"])

def collate_reduction(result_seq_len, n_dropped_tokens, batch):
    batch = torch.tensor(batch)
    batch_size, seq_len = batch.shape
    return (
        batch,
        batched_split_indexes(batch_size, seq_len, result_seq_len, n_dropped_tokens),
    )


def get_dropping_dataloader(
    dataloader_config: dict,
    batch_size_per_device: int,
    sequence_length: int,
    dropped_tokens: int,
    seed: int,
    dataset_split: str,
):
    if dataloader_config.dataset == "c4":
        path = (
            dataloader_config.training_dataset_path
            if dataset_split == "train"
            else dataloader_config.eval_dataset_path
        )
        dataset = C4Dataset(
            sequence_length=sequence_length + dropped_tokens + 1,
            path=path,
            seed=seed,
            use_new_sampling_method=dataloader_config.use_new_sampling_method,
            shuffle=dataloader_config.shuffle,
            world_size_independent=dataloader_config.world_size_independent,
        )
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=batch_size_per_device,
            collate_fn=partial(collate_reduction, sequence_length, dropped_tokens),
            pin_memory=True,
            num_workers=dataloader_config.num_workers,
        )
    else:
        raise ValueError(f"Unsupported model type: '{dataloader_config.dataset}'")

    return dataloader


def get_reduction_dataloaders(
    dataloader_config,
    sequence_length,
    seed,
    dropped_tokens,
):
    world_size = int(os.environ["WORLD_SIZE"])
    batch_size_per_device = dataloader_config.total_batch_size // world_size
    logger.debug(f"Batch size per device: {batch_size_per_device}")
    logger.debug(f"Total: {dataloader_config.total_batch_size}")

    train_dataloader = get_dropping_dataloader(
        dataloader_config=dataloader_config,
        batch_size_per_device=batch_size_per_device,
        sequence_length=sequence_length,
        seed=seed,
        dropped_tokens=dropped_tokens,
        dataset_split="train",
    )
    eval_dataloader = get_dataloader(
        dataloader_config=dataloader_config,
        batch_size_per_device=batch_size_per_device,
        sequence_length=sequence_length,
        seed=seed,
        dataset_split="validation",
    )
    return train_dataloader, eval_dataloader


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
