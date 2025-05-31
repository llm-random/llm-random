from collections import OrderedDict
import os
import re
import neptune
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Literal
from attr import define
import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from typing import Optional, List
from datasets import load_from_disk
from attr import dataclass
import itertools
import numpy as np
from abc import ABC, abstractmethod
import random
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase
from abc import ABC, abstractmethod
import torch.distributed as dist
from torch.nn.attention import SDPBackend
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting
import sys
from datasets import load_dataset
import logging
from neptune.integrations.python_logger import NeptuneHandler
from old_datasets import LLMBatch
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp
from datasets.distributed import split_dataset_by_node
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR


logger = logging.getLogger(__name__)


def check_env_vars():
    assert int(os.environ["RANK"]) < int(os.environ["WORLD_SIZE"])


def setup_enviroment():
    if "WORLD_SIZE" not in os.environ:
        logger.warning("WORLD_SIZE is not set, setting it to 1")
        os.environ["WORLD_SIZE"] = "1"

    if "RANK" not in os.environ:
        if "SLURM_PROCID" in os.environ:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
        else:
            logger.warning("RANK is not set, setting it to 0")
            os.environ["RANK"] = "0"

    if "LOCAL_RANK" not in os.environ:
        if "SLURM_LOCALID" in os.environ:
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        else:
            logger.warning("LOCAL_RANK is not set, setting it to 0")
            os.environ["LOCAL_RANK"] = "0"

    if "MASTER_ADDR" not in os.environ:
        default_master_addr = "localhost"
        logger.warning(f"MASTER_ADDR is not set, setting it to {default_master_addr}")
        os.environ["MASTER_ADDR"] = default_master_addr

    if "MASTER_PORT" not in os.environ:
        default_master_port = "12355"
        logger.warning(f"MASTER_PORT is not set, setting it to {default_master_port}")
        os.environ["MASTER_PORT"] = default_master_port

    check_env_vars()


def distributed_setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    else:
        logger.warning("CUDA is not available. Running on CPU and 'gloo' backend.")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def upload_config_file(metric_logger):
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    file_path = f"generated_configs/config_{slurm_array_task_id}.yaml"
    if slurm_array_task_id is not None and os.path.exists(file_path):
        metric_logger.run["yaml_config"].upload(
            f"generated_configs/config_{slurm_array_task_id}.yaml"
        )


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def run(cfg, metric_logger=None):
    instantiate(cfg.training, _convert_="all")  # Works as check
    setup_enviroment()

    if "distributed" in cfg and cfg.distributed is not None:
        distributed_setup()
    training_state = load_training_state(cfg.checkpoint_config)

    if metric_logger is None:
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
            neptune_run_id=training_state["run_id"],
        )

    if isinstance(metric_logger, NeptuneLogger):
        metric_logger.run["job_config"] = cfg
        upload_config_file(metric_logger)

    torch.manual_seed(cfg.trainer_factory.train_dataloader.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model, _convert_="all").to(device)

    # Residual layers needs metric_logger for logging update norms
    for _, module in model.named_modules():
        if isinstance(module, Residual):
            module.set_metric_logger(metric_logger)

    if "distributed" in cfg and cfg.distributed is not None:
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

    scheduler = instantiate(cfg.training.scheduler)(optimizer=optimizer)

    load_checkpoint(cfg.checkpoint_config, model, optimizer, scheduler)
    trainer_factory = instantiate(cfg.trainer_factory)
    trainer_factory(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        metric_logger=metric_logger,
    ).train()

    cleanup()


def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


def _process_document(document, encode_fn, eot_str):
    """
    Used for processing the dataset in the C4Dataset class.
    It should be lambda inside a map function, but apparently it is not possible to pickle it.
    """
    return {
        "tokens": encode_fn(
            document["text"] + eot_str,
            truncation=False,
            max_length=int(1e10),
        )
    }


class C4Dataset(IterableDataset):
    """
    world_size_independent - if True, we take the whole dataset and take every 'mod rank' element. If world_size == 1 it does not matter.
    shuffle - if True, we shuffle the dataset independently on each rank part (unless world_size_independent is True)


    world_size_independent should be 'True' for tests and eval
    """

    BUFFER_SIZE = 10000
    NUM_SHARDS = 64

    def __init__(
        self,
        sequence_length,
        path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        seed: Optional[int] = None,
        eot_str: str = "<|endoftext|>",
        use_new_sampling_method: bool = True,
        shuffle: bool = True,
        world_size_independent: bool = False,
    ):
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.rank = int(os.environ.get("RANK"))
        self.use_new_sampling_method = use_new_sampling_method
        self.world_size_independent = world_size_independent
        if tokenizer is None:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._load_dataset(path, split, seed, tokenizer, eot_str, shuffle)
        self.sequence_length = sequence_length
        self.seed = seed
        self.rng = random.Random(seed)

    def _load_dataset(self, path, split, seed, tokenizer, eot_str, shuffle: bool):
        if path is None:
            logger.debug(
                f"Loading 'allenai/c4' dataset from HuggingFace with split={split}"
            )
            hf_dataset = load_dataset(
                "allenai/c4",
                "en",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
        else:
            logger.debug(f"Loading dataset from path '{path}'")
            hf_dataset = load_from_disk(path)
            hf_dataset = hf_dataset.to_iterable_dataset(num_shards=self.NUM_SHARDS)

        if not self.world_size_independent:
            hf_dataset = split_dataset_by_node(
                hf_dataset, rank=self.rank, world_size=self.world_size
            )

        if shuffle:
            hf_dataset = hf_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=seed)

        self.data_generator = hf_dataset.map(
            _process_document,
            fn_kwargs={"encode_fn": tokenizer.encode, "eot_str": eot_str},
        )

    def get_infinite_sampler(self):
        epoch = 0
        while True:
            self.data_generator.set_epoch(epoch)
            for next_sample in self.data_generator:
                yield next_sample
            epoch += 1

    def sample_packer(self):
        buffer: List[int] = []
        sampler = iter(self.get_infinite_sampler())
        if self.use_new_sampling_method:

            while True:
                sample = next(sampler)["tokens"]

                if len(buffer) == 0:
                    rand_num = self.rng.randint(0, len(sample) - 1)
                    sample = sample[rand_num:]

                buffer.extend(sample)

                if len(buffer) >= self.sequence_length:
                    yield buffer[: self.sequence_length]
                    buffer = []
        else:
            document_lengths: List[int] = []
            while True:
                tokens = next(sampler)["tokens"]
                buffer.extend(tokens)

                document_lengths.append(len(tokens))
                if (
                    sum(document_lengths) - max(document_lengths)
                ) > self.sequence_length:
                    sample_start = self.rng.randint(0, len(buffer) - 1)
                    sample_end = sample_start + self.sequence_length
                    input_ids = list(take_circular(buffer, sample_start, sample_end))
                    yield input_ids
                    buffer, document_lengths = [], []

    def __iter__(self):
        self.rng.seed(self.seed)
        if self.world_size_independent:
            return itertools.islice(
                self.sample_packer(), self.rank, None, self.world_size
            )
        else:
            return self.sample_packer()


def collate_wrapper(examples):
    return torch.from_numpy(np.array(examples))


def get_dataloader(
    dataset_type: str,
    dataset_path: str,
    dataset_split: str,
    total_batch_size: int,
    sequence_length: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    use_new_sampling_method: bool,
    world_size_independent: bool,
    collate_fn: Callable = collate_wrapper,
):
    world_size = int(os.environ["WORLD_SIZE"])
    batch_size_per_device = total_batch_size // world_size
    logger.debug(f"Batch size per device: {batch_size_per_device}")
    logger.debug(f"Total: {total_batch_size}")
    if dataset_type == "c4":
        dataset = C4Dataset(
            sequence_length=sequence_length + 1,
            split=dataset_split,
            path=dataset_path,
            seed=seed,
            use_new_sampling_method=use_new_sampling_method,
            shuffle=shuffle,
            world_size_independent=world_size_independent,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_device,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unsupported dataset type: '{dataset_type}'")

    return dataloader


@dataclass
class AttentionConfig:
    mode: str
    n_heads: int


@dataclass
class FeedForwardConfig:
    mode: str


@dataclass
class BlockConfig:
    attention: AttentionConfig
    feedforward: FeedForwardConfig
    residual_mode: str
    norm_class_mode: str


@dataclass
class TowerConfig:
    mode: str
    n_blocks: int
    block_config: BlockConfig


@dataclass
class EmbeddingConfig:
    mode: str


class Common(BaseModel):
    model_type: str
    sequence_length: int
    dmodel: int
    dff: int
    init_type: str
    init_scale: float
    vocab_size: int
    head_norm: bool


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cosine", "trapezoidal"]
    warmup_steps: NonNegativeInt


class CosineSchedulerConfig(SchedulerConfig):
    n_steps: PositiveInt
    final_lr_fraction: NonNegativeFloat


class TrapezoidalSchedulerConfig(SchedulerConfig):
    constant_steps: NonNegativeInt
    decay_steps: NonNegativeInt


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: PositiveFloat
    weight_decay: PositiveFloat
    scheduler: object
    gradient_accumulation_steps: PositiveInt
    n_steps: PositiveInt
    gradient_clipping: PositiveFloat
    evaluation: dict


class MetricLoggerConfig(BaseModel):
    type: Optional[str]
    project_name: Optional[str]
    name: Optional[str]
    tags: Optional[List[str]]
    heavy_metrics_calculation_interval: Optional[int]
    new_neptune_job: Optional[bool] = None


class RMSNorm(nn.Module):
    def __init__(self, dmodel, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.g = nn.Parameter(torch.ones(dmodel))
        self.b = nn.Parameter(torch.zeros(dmodel))

    def forward(self, x):
        norm = torch.mean(x**2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.g + self.b


class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer
        self.metric_logger = None

    def set_metric_logger(self, metric_logger):
        self.metric_logger = metric_logger

    def forward(self, x):
        out = self.layer(x)
        if self.metric_logger is not None:
            self.metric_logger.accumulate_metrics(
                layer_name=f"{self.log_name}",
                transform_fn=Residual.intermediate_norms,
                calculate_fn=Residual.calculate_metrics,
                metrics={
                    "residual_stream": x,
                    "updates": out,
                },
            )
        return out + x

    @staticmethod
    def intermediate_norms(residual_stream: torch.Tensor, updates: torch.Tensor):

        with torch.no_grad():
            update_norms = torch.norm(updates, dim=-1)
            residual_norms = torch.norm(residual_stream, dim=-1)

            return {
                "update_norms_list": update_norms,
                "residual_norms_list": residual_norms,
            }

    @staticmethod
    def calculate_metrics(
        update_norms_list: torch.Tensor, residual_norms_list: torch.Tensor
    ):
        update_norms_concat = torch.cat(update_norms_list)
        residual_norms_concat = torch.cat(residual_norms_list)

        if dist.is_initialized():
            world_size = int(os.environ["WORLD_SIZE"])
            gpu_batch_size, seq_len = residual_norms_concat.shape
            update_norms = torch.empty(
                world_size * gpu_batch_size,
                seq_len,
                device=update_norms_concat.device,
                dtype=update_norms_concat.dtype,
            )
            dist.all_gather_into_tensor(update_norms, update_norms_concat)

            residual_norms = torch.empty(
                world_size * gpu_batch_size,
                seq_len,
                device=residual_norms_concat.device,
                dtype=residual_norms_concat.dtype,
            )
            dist.all_gather_into_tensor(residual_norms, residual_norms_concat)
        else:
            update_norms = update_norms_concat
            residual_norms = residual_norms_concat

        with torch.no_grad():
            update_norms_std, update_norms_mean = torch.std_mean(update_norms)
            residual_norms_std, residual_norms_mean = torch.std_mean(residual_norms)

            update_to_residual_ratio = update_norms / residual_norms
            ratio_std, ratio_mean = torch.std_mean(update_to_residual_ratio)

            return {
                "update_norms/mean": update_norms_mean.item(),
                "update_norms/std": update_norms_std.item(),
                "residual_norms/mean": residual_norms_mean.item(),
                "residual_norms/std": residual_norms_std.item(),
                "update_to_residual_ratio/mean": ratio_mean.item(),
                "update_to_residual_ratio/std": ratio_std.item(),
            }


def PreNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return Residual(
        nn.Sequential(
            OrderedDict(
                [
                    ("pre_norm", norm_class(dmodel)),
                    (f"{name}", layer),
                ]
            )
        )
    )


def PostNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return nn.Sequential(
        OrderedDict(
            [
                (f"{name}", Residual(layer)),
                ("post_norm", norm_class(dmodel)),
            ]
        )
    )


def TokenEmbedding(
    vocab_size,
    embedding_dim,
    init_type: str,
    init_scale: float,
):
    weight = get_init_weight(
        shape=(vocab_size, embedding_dim),
        fan_in=1,
        init_type=init_type,
        scale=init_scale,
    )
    return nn.Embedding(vocab_size, embedding_dim, _weight=weight)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: str,
        init_scale: float,
    ):
        super(PositionalEmbedding, self).__init__()
        self.layer = nn.Embedding(max_length, embedding_dim)
        default_weight = self.layer.weight.data
        self.layer.weight.data = get_init_weight(
            shape=default_weight.shape,
            fan_in=1,
            init_type=init_type,
            scale=init_scale,
            dtype=default_weight.dtype,
        )
        # TODO(jaszczur): add initialization as positional encoding

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.layer(positions)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(
        self,
        common,
        block_config,
    ):
        super(TransformerBlock, self).__init__()
        residual_fn = get_residual_function(
            block_config.residual_mode, common.dmodel, block_config.norm_class_mode
        )

        attention_function = get_attention_function(common, block_config.attention)

        ff_layer = get_ff_layer_function(
            common,
            block_config.feedforward.mode,
        )

        residual_layers = [
            (
                "residual_attention",
                residual_fn(layer=attention_function(), name="attention"),
            ),
            (
                "residual_feedforward",
                residual_fn(layer=ff_layer(), name="feedforward"),
            ),
        ]
        self.block = nn.Sequential(OrderedDict(residual_layers))

    def forward(self, x):
        return self.block(x)


class TransformerTower(nn.Module):
    def __init__(
        self,
        common: Common,
        tower_config: TowerConfig,
    ):
        super().__init__()
        blocks = [
            (
                f"block_{i}",
                TransformerBlock(
                    common,
                    tower_config.block_config,
                ),
            )
            for i in range(tower_config.n_blocks)
        ]
        self.blocks = nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        return self.blocks(x)


class Aggregate(nn.Module):
    def __init__(self, function, *layers):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        result = None
        for layer in self.layers:
            if result is None:
                result = layer(x)
            else:
                result = self.function(result, layer(x))
        return result


class Aggregate(nn.Module):
    def __init__(
        self,
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *layers: nn.Module,
    ):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)
        assert len(self.layers) > 0, "Aggregate must have at least one layer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.layers[0](x)
        for layer in self.layers[1:]:
            result = self.function(result, layer(x))
        return result


class Linear(nn.Linear):
    def __init__(self, *args, init_type, init_scale, **kwargs):
        if "bias" not in kwargs:
            kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.weight.data = get_init_weight(
            shape=self.weight.shape,
            fan_in=self.in_features,
            init_type=init_type,
            scale=init_scale,
            dtype=self.weight.dtype,
        )


class EmbeddingLayer(Aggregate):
    def __init__(self, *layers):
        super(EmbeddingLayer, self).__init__((lambda x, y: x + y), *layers)


class PredictionHead(nn.Module):
    def __init__(
        self, embedding_dim, output_size, init_type, init_scale, use_layer_norm: bool
    ):
        super(PredictionHead, self).__init__()

        layers = OrderedDict()
        if use_layer_norm:
            layers["head_norm"] = nn.LayerNorm(embedding_dim)
        layers["head"] = Linear(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )

        self.unembedding = nn.Sequential(layers)

    def forward(self, x):
        return self.unembedding(x)


class LLM(nn.Module):
    def __init__(
        self,
        embedding,
        common: Common,
        tower_config: TowerConfig,
    ):
        super(LLM, self).__init__()

        self.embedding_layer = embedding

        self.encoder = TransformerTower(
            common=common,
            tower_config=tower_config,
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
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x


def FeedForward(
    dmodel,
    dff,
    init_type: str,
    init_scale: float,
):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "logging_ff_pre_relu",
                    Linear(
                        dmodel,
                        dff,
                        bias=True,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "logging_ff_post_relu",
                    Linear(
                        dff,
                        dmodel,
                        bias=True,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
            ]
        )
    )


def attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
):
    # https://github.com/pytorch/pytorch/blob/ce503c1b40207dab770c28cbd4568cd9e105277b/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L556
    with torch.nn.attention.sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            is_causal=causal,
        )


class AttentionMechanism(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ):
        return attention_mechanism(
            query=query,
            key=key,
            value=value,
            causal=causal,
        )


class Attention(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
    ):
        super(Attention, self).__init__()

        self.heads = heads
        self.causal = causal

        self.input_projection = Linear(
            dmodel,
            3 * dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            dmodel,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.attention_mechanism = AttentionMechanism()

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        q = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


##### Loggers #######
class MetricLogger(ABC):
    def __init__(self, config=None):
        self.heavy_metrics_calculation_interval = (
            1 if config is None else config.heavy_metrics_calculation_interval
        )
        self.accumulators = {}
        self.step = 0

    @property
    def _should_log_heavy_metrics(self) -> bool:
        return (
            self.step is not None
            and self.heavy_metrics_calculation_interval > 0
            and (self.step) % self.heavy_metrics_calculation_interval == 0
        )

    @abstractmethod
    def log(self, name, step, value):
        pass

    def accumulate_metrics(self, layer_name, calculate_fn, metrics, transform_fn=None):
        if self._should_log_heavy_metrics:
            if transform_fn is not None:
                metrics = transform_fn(**metrics)

            if layer_name not in self.accumulators:
                self.accumulators[layer_name] = MetricAccumulator(calculate_fn, metrics)
            else:
                self.accumulators[layer_name].append(metrics)

    def flush_accumulated_metrics(self, step):
        if self._should_log_heavy_metrics:
            for name, accumulator in self.accumulators.items():
                for metric, result in accumulator.calculate().items():
                    self.log(f"steps/{name}/{metric}", step, result)
                accumulator.reset()

    def set_step(self, step):
        self.step = step


class MetricAccumulator:
    def __init__(self, calculate_fn, metrics):
        self.acc_dict = {}
        self.calculate_fn = calculate_fn
        self.append(metrics)

    def append(self, metrics):
        for key, value in metrics.items():
            if key in self.acc_dict:
                self.acc_dict[key].append(value)
            else:
                self.acc_dict[key] = [value]

    def calculate(self):
        return self.calculate_fn(**self.acc_dict)

    def reset(self):
        for key in self.acc_dict:
            self.acc_dict[key] = []


class DummyLogger(MetricLogger):
    def log(self, _name, _step, _value):
        pass


class NeptuneLogger(MetricLogger):
    def __init__(self, run, rank, config=None):
        super().__init__(config)
        self.run = run
        self.rank = rank

    def log(self, name, step, value):
        if self.rank == 0:
            self.run[name].append(value=value, step=step)


class StdoutLogger(MetricLogger):
    def __init__(self, config=None):
        super().__init__(config)
        self.rank = os.environ.get("RANK", 0)
        logger.info("Logging to stdout.")

    def log(self, name, step, value):
        logger.info(f"[device:{self.rank}] on step:{step} -> {name}: {value}")


class RecorderLogger(MetricLogger):
    def __init__(self, config=None):
        super().__init__(config)
        self.data = {}

    def log(self, name, step, value):
        if name not in self.data:
            self.data[name] = []
        self.data[name].append((value, step))

    def clear(self):
        self.data = {}


def get_composition_file_path(hydra_config):
    """
    Get the path to the file passed to main
    """
    config_name = hydra_config.job.config_name
    config_path = [
        path["path"]
        for path in hydra_config.runtime.config_sources
        if path["schema"] == "file"
    ][0]

    return f"{config_path}/{config_name}.yaml"


def get_metric_logger(
    metric_logger_config: Optional[MetricLoggerConfig] = None,
    neptune_run_id: Optional[str] = None,
):
    _metric_logger = None
    if metric_logger_config.type == "neptune":
        neptune_run_id = (
            None if metric_logger_config.new_neptune_job else neptune_run_id
        )
        rank = int(os.environ["RANK"])
        if int(os.environ["WORLD_SIZE"]) > 1:

            # As suggested here: https://docs.neptune.ai/tutorials/running_distributed_training/#tracking-a-multi-node-ddp-job
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

            if rank == 0:
                neptune_logger = neptune.init_run(
                    project=metric_logger_config.project_name,
                    with_id=neptune_run_id,
                    monitoring_namespace=f"monitoring/gpu_{rank}",
                    name=metric_logger_config.name,
                    tags=metric_logger_config.tags,
                )
                if neptune_run_id is None:
                    neptune_run_id = neptune_logger["sys/id"].fetch()
                    broadcast_message(rank, neptune_run_id)
                _metric_logger = NeptuneLogger(
                    neptune_logger, rank, metric_logger_config
                )
            else:
                if neptune_run_id is None:
                    neptune_run_id = broadcast_message(rank)
                neptune_logger = neptune.init_run(
                    project=metric_logger_config.project_name,
                    with_id=neptune_run_id,
                    monitoring_namespace=f"monitoring/gpu_{rank}",
                    name=metric_logger_config.name,
                    tags=metric_logger_config.tags,
                )
                _metric_logger = NeptuneLogger(
                    neptune_logger, rank, metric_logger_config
                )

        else:
            neptune_logger = neptune.init_run(
                project=metric_logger_config.project_name,
                name=metric_logger_config.name,
                tags=metric_logger_config.tags,
                with_id=neptune_run_id,
            )
            _metric_logger = NeptuneLogger(neptune_logger, rank, metric_logger_config)

        npt_handler = NeptuneHandler(run=_metric_logger.run)
        logger.addHandler(npt_handler)

    elif metric_logger_config.type == "stdout":
        _metric_logger = StdoutLogger(metric_logger_config)
    elif metric_logger_config.type == "record":
        _metric_logger = RecorderLogger(metric_logger_config)
    elif metric_logger_config.type == "dummy":
        _metric_logger = DummyLogger(metric_logger_config)
    elif metric_logger_config.type == None:
        raise RuntimeError("Metric logger is not initialized yet.")
    else:
        raise ValueError(f"Unknown logger type: { metric_logger_config.type}")
    return _metric_logger


##### End Loggers #######


class TrapezoidalLR(SequentialLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        constant_steps,
        decay_steps,
    ):
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps

        # Define individual schedulers
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        constant_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=constant_steps,
        )

        linear_decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=decay_steps,
        )

        schedulers = [warmup_scheduler, constant_scheduler, linear_decay_scheduler]
        milestones = [warmup_steps, warmup_steps + constant_steps]

        super(TrapezoidalLR, self).__init__(
            optimizer, schedulers=schedulers, milestones=milestones
        )

    def load_state_dict(self, loaded_state):

        if loaded_state["last_epoch"] < self.last_epoch:
            raise RuntimeError(
                "Loaded scheduler checkpoint should have more steps than current one."
            )

        """
        It is a workaround for the problem with loading state_dict of different schedulers.
        The problem is that the state_dict of SequentialLR is not obvious. State of particular scheduler changes after each step. 
        But when it is milestone step, both schedulers are updated. It is easier to just load state and then step to the last_epoch.
        """
        while loaded_state["last_epoch"] > self.last_epoch:
            self.step()


def create_batch_fingerprint(batch):
    def prefix_suffix_only(array, prefix=3, suffix=3):
        prefix_part = array[:prefix]
        suffix_part = array[-suffix:]
        result = prefix_part + suffix_part
        return result

    first_row = prefix_suffix_only(batch[0]).numpy().tolist()
    middle_row = prefix_suffix_only(batch[len(batch) // 2]).numpy().tolist()
    last_row = prefix_suffix_only(batch[-1]).numpy().tolist()

    return first_row + middle_row + last_row


@define(slots=False)
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_accumulation_steps: int
    training_state: dict
    n_steps: int
    train_dataloader: IterableDataset
    eval_dataloader: IterableDataset
    metric_logger: MetricLogger
    eval_interval: int
    n_eval_steps: int
    gradient_clipping: Optional[float]
    checkpoint_config: Optional[dict]

    def __attrs_post_init__(self):
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.model.parameters()).device
        self.loss_interval_100 = 0.0
        self.eval_iterator = iter(self.eval_dataloader)

        if self.start_step > 0:
            n_skip_eval_batches = (
                (self.start_step - 1) // self.eval_interval * self.n_eval_steps
            )
            logger.debug(f"Skipping {n_skip_eval_batches} eval batches")
            for _ in range(n_skip_eval_batches):
                next(self.eval_iterator)

    @property
    def _should_evaluate(self) -> bool:
        return (
            self.eval_interval > 0
            and self.step % self.eval_interval == 0
            and self.step != 0
        )

    @property
    def _should_log_eval_input(self) -> bool:
        return self.step % (self.eval_interval * 100) == 0

    @property
    def _should_save_checkpoint(self) -> bool:
        return (
            self.checkpoint_config.interval > 0
            and (self.step) % self.checkpoint_config.interval == 0
            and self.step != 0
            and self.checkpoint_config.save_path is not None
        )

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.model.train()
            loss = self.calculate_loss(batch)

            grad_norm = self.clip_gradient()

            self.log_metrics(loss, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()

    def _preprocess_input(self, batch):  # TODO test it
        input_ids = batch[:, :-1].contiguous()
        target_ids = batch[:, 1:].contiguous()

        return input_ids, target_ids

    def calculate_loss(self, batch):
        def _hack_for_python_garbage_collection(input_ids, target_ids):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            predicted_ids = self.model(input_ids)

            # Tensors should be on the same device for loss calculation #TODO check
            target_ids = target_ids.to(predicted_ids.device)

            mask_loss = F.cross_entropy(
                predicted_ids.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            loss = mask_loss.mean() / self.gradient_accumulation_steps
            return loss

        losses = []
        if type(batch) is LLMBatch:
            for input_ids, target_ids in zip(
                batch.input_ids.chunk(self.gradient_accumulation_steps),
                batch.target_ids.chunk(self.gradient_accumulation_steps),
            ):
                input_ids = input_ids.to(self.device)
                predicted_ids = self.model(input_ids)

                # Tensors should be on the same device for loss calculation #TODO check
                target_ids = target_ids.to(predicted_ids.device)

                mask_loss = F.cross_entropy(
                    predicted_ids.flatten(0, -2),
                    target_ids.reshape(-1).long(),
                    reduction="none",
                )
                loss = mask_loss.mean() / self.gradient_accumulation_steps

                if self.model.training:
                    loss.backward()
                losses.append(loss.item())
        else:
            for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
                input_ids, target_ids = self._preprocess_input(batch_chunk)
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
                loss = self.calculate_loss(batch)
                losses.append(loss.item())
                self.metric_logger.flush_accumulated_metrics(self.step)
            avg_loss = torch.tensor(losses).mean()
            self.metric_logger.log("steps/eval/loss", self.step, avg_loss.item())
            self.metric_logger.log(
                "tokens/eval/loss", self.processed_tokens, avg_loss.item()
            )

        if self._should_log_eval_input:
            self.metric_logger.log(
                f"steps/eval/batch", self.step, str(eval_fingerprint)
            )

    def clip_gradient(self):
        if self.gradient_clipping is not None:
            if isinstance(self.model, FSDP):
                return self.model.clip_grad_norm_(self.gradient_clipping)
            else:
                return torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )

    def _update_processed_tokens(self, batch):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def log_metrics(self, loss, grad_norm):
        self.metric_logger.log("step", self.step, self.step)
        self.metric_logger.log("steps/train/loss", self.step, loss.item())
        self.metric_logger.log(
            "steps/train/lr", self.step, (self.scheduler.get_last_lr()[0])
        )
        self.metric_logger.log("steps/train/grad_norm", self.step, grad_norm.item())
        self.metric_logger.log(
            "steps/train/processed_tokens", self.step, self.processed_tokens
        )

        self.metric_logger.log("tokens/train/loss", self.processed_tokens, loss.item())
        self.metric_logger.log(
            "tokens/lr", self.processed_tokens, (self.scheduler.get_last_lr()[0])
        )
        self.metric_logger.log(
            "tokens/train/grad_norm", self.processed_tokens, grad_norm.item()
        )
        self.metric_logger.flush_accumulated_metrics(self.step)

    def save_checkpoint(self):
        if isinstance(self.model, FSDP):
            # Sharded save
            checkpoint_folder = step_checkpoint_path(self.checkpoint_config, self.step)
            state_dict = {
                "app": TrainingState(self.model, self.optimizer, self.scheduler)
            }
            dcp.save(state_dict, checkpoint_id=checkpoint_folder)
            logger.info(f"Saved sharded model checkpoint in {checkpoint_folder}")
        else:
            # Non-sharded save
            if os.environ["RANK"] == "0":
                checkpoint_folder = step_checkpoint_path(
                    self.checkpoint_config, self.step
                )
                os.makedirs(checkpoint_folder, exist_ok=True)
                checkpoint_path = f"{checkpoint_folder}/{self.checkpoint_config.model_checkpoint_filename}"
                state_to_save = {
                    "model": (
                        self.model.module.state_dict()
                        if type(self.model) is DDP
                        else self.model.state_dict()
                    ),
                    "optim": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }
                torch.save(state_to_save, checkpoint_path)
                logger.info(
                    f"Saved non-sharded model checkpoint in '{checkpoint_path}'"
                )

        if os.environ["RANK"] == "0":
            save_training_state(
                checkpoint_config=self.checkpoint_config,
                step=self.step,
                processed_tokens=self.processed_tokens,
                metric_logger=self.metric_logger,
            )


def init_kaiming_uniform(shape, fan_in, scale, dtype=torch.float32):
    range_ = scale * (3 / fan_in) ** 0.5
    return torch.zeros(shape, dtype=dtype).uniform_(-range_, range_)


def init_truncated_normal(shape, fan_in, scale, dtype=torch.float32):
    std = (scale / fan_in) ** 0.5
    low = -2 * scale
    high = 2 * scale
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)


def init_truncated_normal_fixed(shape, fan_in, scale, dtype=torch.float32):
    std = scale * (1 / fan_in) ** 0.5
    low = -2 * std
    high = 2 * std
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)


def get_init_weight(shape, fan_in, init_type: str, scale, dtype=torch.float32):
    init_types = {
        "kaiming_uniform": init_kaiming_uniform,
        "truncated_normal": init_truncated_normal,
        "truncated_normal_fixed": init_truncated_normal_fixed,
    }

    if init_type not in init_types:
        raise ValueError(f"Unknown init_type: {init_type}")

    return init_types[init_type](shape=shape, fan_in=fan_in, scale=scale, dtype=dtype)


def get_norm_class_function(norm_class_mode: str):
    norm_classes = {
        "layer_norm": nn.LayerNorm,
        "rms_norm": RMSNorm,
    }

    if norm_class_mode not in norm_classes:
        raise NotImplementedError(
            f"Norm class {norm_class_mode} not implemented. Supported types are: {list(norm_classes.keys())}"
        )

    return norm_classes[norm_class_mode]


def get_residual_function(
    residual_mode: str, dmodel: int, norm_class_mode: str
) -> Callable[[], nn.Module]:
    norm_class = get_norm_class_function(norm_class_mode)
    residual_layers = {
        "pre_norm": lambda layer, name: PreNormBlock(
            dmodel, layer, name, norm_class=norm_class
        ),
        "post_norm": lambda: PostNormBlock(dmodel, norm_class=norm_class),
    }

    if residual_mode not in residual_layers:
        raise NotImplementedError(
            f"Unsupported residual_mode: {residual_mode}. Supported modes are: {list(residual_layers.keys())}"
        )

    return residual_layers[residual_mode]


def get_attention_function(
    common: Common,
    attention_config: AttentionConfig,
) -> Callable[[], nn.Module]:
    causal = common.model_type == "gpt"

    attention_functions = {
        "vanilla": lambda: Attention(
            dmodel=common.dmodel,
            heads=attention_config.n_heads,
            causal=causal,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        # Add other attention modes here
    }

    if attention_config.mode not in attention_functions:
        raise ValueError(
            f"Unsupported attention_mode: {attention_config.mode}. Supported modes are: {list(attention_functions.keys())}"
        )

    return attention_functions[attention_config.mode]


def get_ff_layer_function(
    common: Common,
    ff_mode: str,
) -> Callable[[], nn.Module]:

    ff_functions = {
        "vanilla": lambda: FeedForward(
            common.dmodel,
            common.dff,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        # Add other here
    }

    if ff_mode not in ff_functions:
        raise ValueError(
            f"Unsupported ff_mode: {ff_mode}. Supported modes are: {list(ff_functions.keys())}"
        )

    return ff_functions[ff_mode]


def get_vanilla_embedding(common):
    return EmbeddingLayer(
        TokenEmbedding(
            common.vocab_size,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
        PositionalEmbedding(
            common.sequence_length,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        ),
    )


def get_cosine_scheduler_with_warmup(optimizer, config: CosineSchedulerConfig):
    assert (
        len(optimizer.param_groups) == 1
    ), "Cosine scheduler only supports one param group"
    optimizer_lr = optimizer.param_groups[0][
        "lr"
    ]  # param_groups changes when applying scheduler
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )
    after_warmup_steps = config.n_steps - config.warmup_steps - 1
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0
    )  # TODO this is only because of a bug in llm-random
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=after_warmup_steps,
        eta_min=config.final_lr_fraction * optimizer_lr,
    )
    training_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps, config.warmup_steps + 1],
    )
    return training_scheduler


def get_classes_from_globals(names):
    return [globals().get(name) for name in names]


#### Distributed ####
def wrap_model(model, fsdp_config):

    classes_to_wrap = get_classes_from_globals(fsdp_config.modules_to_wrap)
    igonore_mixed_precision_classes = get_classes_from_globals(
        fsdp_config.mixed_precision.ignored_classes
    )
    mixed_precision_dtype = getattr(
        sys.modules["torch"], fsdp_config.mixed_precision.dtype
    )

    wrapped_model = FSDP(
        model,
        device_id=int(os.environ["RANK"]),
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            cast_forward_inputs=True,
            _module_classes_to_ignore=igonore_mixed_precision_classes,
        ),
        auto_wrap_policy=ModuleWrapPolicy(classes_to_wrap),
    )
    return wrapped_model


class TrainingState(Stateful):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])


def broadcast_message(rank, message=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Broadcast the length of the message
    if rank == 0:
        message_length_tensor = torch.tensor(
            len(message), dtype=torch.int32, device=device
        )
    else:
        message_length_tensor = torch.empty(1, dtype=torch.int32, device=device)

    dist.broadcast(message_length_tensor, src=0)

    # Prepare the tensor to hold the message of the correct length
    if rank == 0:
        message_tensor = torch.tensor(
            list(message.encode("utf-8")), dtype=torch.uint8, device=device
        )
    else:
        message_tensor = torch.empty(
            message_length_tensor.item(), dtype=torch.uint8, device=device
        )

    # Broadcast the message string data
    dist.broadcast(message_tensor, src=0)

    return message_tensor.cpu().numpy().tobytes().decode("utf-8")


def step_checkpoint_path(checkpoint_config, step):
    full_config_path = get_full_checkpoint_save_path(checkpoint_config.save_path)
    return f"{full_config_path}/step_{step}"


def save_training_state(
    checkpoint_config,
    step,
    processed_tokens,
    metric_logger=None,
):
    run_id = (
        metric_logger.run["sys/id"].fetch()
        if type(metric_logger) is NeptuneLogger
        else None
    )

    directory = step_checkpoint_path(checkpoint_config, step)
    torch.save(
        {"next_step": step + 1, "run_id": run_id, "processed_tokens": processed_tokens},
        f"{directory}/{checkpoint_config.training_state_filename}",
    )

    logger.info(
        f"Saved training state in '{checkpoint_config.save_path}/{checkpoint_config.training_state_filename}'"
    )


def get_full_checkpoint_save_path(save_path):
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    return (
        f"{save_path}/{slurm_array_task_id}"
        if slurm_array_task_id is not None
        else save_path
    )


def load_training_state(checkpoint_config):
    training_start_config = {"next_step": 0, "run_id": None, "processed_tokens": 0}

    checkpoint_folder = checkpoint_config.get("load_path", None)
    if checkpoint_folder is None:
        checkpoint_path = checkpoint_config.get("save_path", None)
        if checkpoint_path is None:
            logger.warning(
                "Checkpoint save path is not set. Starting training from scratch."
            )
            return training_start_config
        full_checkpoint_path = get_full_checkpoint_save_path(
            checkpoint_config.save_path
        )
        os.makedirs(full_checkpoint_path, exist_ok=True)
        checkpoint_folder = _find_latest_checkpoint(full_checkpoint_path)

    if checkpoint_folder is None:
        return training_start_config

    training_state_path = (
        f"{checkpoint_folder}/{checkpoint_config.training_state_filename}"
    )
    if os.path.isfile(training_state_path):
        return torch.load(training_state_path)
    else:
        logger.warning(
            f"Training state file '{training_state_path}' not found. "
            "Starting training from scratch."
        )

    return training_start_config


def _find_latest_checkpoint(path: str) -> str:
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if not files:
        logger.info(f"No checkpoints in '{path}'")
        return

    return max(files, key=os.path.getmtime)


def load_checkpoint(checkpoint_config, model, optimizer, scheduler):
    checkpoint_folder = checkpoint_config.get("load_path", None)
    if checkpoint_folder is None:
        checkpoint_path = checkpoint_config.get("save_path", None)
        if checkpoint_path is None:
            return
        full_checkpoint_path = get_full_checkpoint_save_path(
            checkpoint_config.save_path
        )
        checkpoint_folder = _find_latest_checkpoint(full_checkpoint_path)

    if checkpoint_folder is not None:
        if isinstance(model, FSDP):
            # Sharded load
            state_dict = {"app": TrainingState(model, optimizer, scheduler)}
            dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_folder)
            logger.debug(f"Loaded sharded checkpoint from '{checkpoint_folder}'")
        else:
            # Non-sharded load
            checkpoint_model = (
                f"{checkpoint_folder}/{checkpoint_config.model_checkpoint_filename}"
            )
            checkpoint = torch.load(checkpoint_model)
            if type(model) is DDP:
                logger.info(f"Loading DDP model from '{checkpoint_folder}'")
                model.module.load_state_dict(checkpoint["model"])
            else:
                logger.info(f"Loading non-DDP model from '{checkpoint_folder}'")
                model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optim"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"Loaded non-sharded sheduler from '{checkpoint_folder}'")
            logger.debug(f"Loaded non-sharded checkpoint from '{checkpoint_folder}'")
