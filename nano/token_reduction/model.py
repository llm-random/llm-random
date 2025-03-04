from functools import partial
import os
import torch
import os
import torch
import torch.nn.functional as F
import logging
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdata.stateful_dataloader import StatefulDataLoader
import torch.distributed as dist
from model import (
    C4Dataset,
    Common,
    EmbeddingLayer,
    Linear,
    NeptuneLogger,
    PositionalEmbedding,
    TokenEmbedding,
    Trainer,
    cleanup,
    distributed_setup,
    get_composition_file_path,
    get_metric_logger,
    get_scheduler,
    load_checkpoint,
    load_training_state,
    setup_enviroment,
    wrap_model,
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
        x = batch_index_select(x, indexes)
        return x


def create_token_dropping_function(_config, common: CommonDroppingConfig):
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
        merge_tokens = batch_index_select(x, merge_indexes)
        merge_tokens = self.linear(merge_tokens)

        # It can happend that if we pick for merge last token from sequence, we do not have next token to merge it with, so we add zero vector
        batch_size, _, dmodel = x.shape
        x = torch.cat([x, torch.zeros(batch_size, 1, dmodel)], dim=1)

        x[
            torch.arange(merge_indexes.size(0)).unsqueeze(-1), merge_indexes + 1
        ] += merge_tokens

        x = batch_index_select(x, keep_indexes)
        return xq


def create_token_merging_function(_config, common: CommonDroppingConfig):
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


def collate_reduction(result_seq_len, n_dropped_tokens, batch):
    batch = torch.tensor(batch)
    batch_size, seq_len = batch.shape
    return (
        batch,
        batched_split_indexes(batch_size, seq_len, result_seq_len, n_dropped_tokens),
    )


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

        input_data, (keep_indexes, _dropped) = batch
        losses = []

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

        input_data, (keep_indexes, drop_indexes) = batch
        losses = []

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

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_loss = torch.tensor(losses, device=loss.device).mean()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

        return avg_loss / float(os.environ["WORLD_SIZE"])


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


def run(cfg, hydra_config):
    instantiate(cfg.training, _convert_="all")  # Works as check
    setup_enviroment()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed_setup(world_size)
    training_state = load_training_state(cfg.checkpoint_config)
    metric_logger = get_metric_logger(
        metric_logger_config=instantiate(cfg.metric_logger_config, _convert_="all"),
        neptune_run_id=training_state["run_id"],
    )

    if isinstance(metric_logger, NeptuneLogger):
        config_path = get_composition_file_path(hydra_config)
        metric_logger.run["hydra_config"] = hydra_config
        metric_logger.run["job_config"] = cfg
        metric_logger.run["config_composition_file"].upload(config_path)
        metric_logger.run["sys/tags"].add(
            OmegaConf.to_object(hydra_config.overrides.task)
        )

    torch.manual_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model, _convert_="all").to(device)
    if world_size > 1:
        if torch.cuda.is_available():
            model = wrap_model(model, cfg.distributed.fsdp)
        else:
            logger.info("FSDP is not supported with CPU. Running DDP instead")
            model = DDP(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = get_scheduler(optimizer, cfg.training)

    train_dataloader = get_dropping_dataloader(
        dataloader_config=cfg.training.dataloader,
        batch_size_per_device=cfg.training.dataloader.total_batch_size,
        sequence_length=cfg.model.common.sequence_length,
        dropped_tokens=cfg.model.common.dropped_tokens,
        seed=cfg.training.seed,
        dataset_split="train",
    )
    eval_dataloader = None

    load_checkpoint(
        cfg.checkpoint_config, model, optimizer, scheduler, train_dataloader
    )
    DroppingTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        training_state=training_state,
        n_steps=cfg.training.n_steps,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        metric_logger=metric_logger,
        eval_interval=cfg.training.evaluation.eval_interval,
        n_eval_steps=cfg.training.evaluation.n_eval_steps,
        gradient_clipping=cfg.training.gradient_clipping,
        checkpoint_config=cfg.checkpoint_config,
    ).train()

    cleanup()
