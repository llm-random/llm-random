from collections import OrderedDict
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable
from attr import define
import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from typing import Optional, Literal, List, Iterator
from functools import partial
from datasets import load_dataset, load_from_disk
from attr import dataclass
import itertools
import numpy as np
from abc import ABC, abstractmethod
import random
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast


@dataclass
class LLMExample(object):
    input_ids: List[int]
    target_ids: List[int]
    should_calculate_loss: List[
        int
    ]  # e.g. in BERT loss is not calculated over non-masked tokens


class LLMBatch:
    def __init__(self, examples: List[LLMExample]):
        self.input_ids = self._make_tensor([example.input_ids for example in examples])
        self.target_ids = self._make_tensor(
            [example.target_ids for example in examples]
        )
        self.should_calculate_loss = self._make_tensor(
            [example.should_calculate_loss for example in examples]
        )
        self.N = len(examples)  # assuming all tensors have the first dimension as N
        self.set_n_chunks(n_chunks=1)

        assert self.input_ids.shape == self.target_ids.shape
        assert self.input_ids.shape == self.should_calculate_loss.shape

    def pin_memory(self):
        """Pin memory for faster transfer to GPU as described in https://pytorch.org/docs/stable/data.html#memory-pinning"""
        self.input_ids = self.input_ids.pin_memory()
        self.target_ids = self.target_ids.pin_memory()
        self.should_calculate_loss = self.should_calculate_loss.pin_memory()
        return self

    def set_n_chunks(self, n_chunks: int):
        assert (
            self.N % n_chunks == 0
        ), "batch_size must be divisible by n_chunks without remainder."

        self.chunk_size = self.N // n_chunks
        self.idx = 0

    def __iter__(self):
        return self

    @property
    def device(self) -> torch.device:
        assert (
            self.input_ids.device
            == self.target_ids.device
            == self.should_calculate_loss.device
        )
        return self.input_ids.device

    def to(self, device) -> "LLMBatch":
        self.input_ids = self.input_ids.to(device)
        self.target_ids = self.target_ids.to(device)
        self.should_calculate_loss = self.should_calculate_loss.to(device)
        return self

    def _make_tensor(self, list_of_token_lists: List[List[int]]) -> torch.Tensor:
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)

    def __next__(self):
        if self.idx < self.N:
            chunk_input_ids = self.input_ids[self.idx : self.idx + self.chunk_size]
            chunk_target_ids = self.target_ids[self.idx : self.idx + self.chunk_size]
            chunk_should_calculate_loss = self.should_calculate_loss[
                self.idx : self.idx + self.chunk_size
            ]
            self.idx += self.chunk_size
            return chunk_input_ids, chunk_target_ids, chunk_should_calculate_loss
        else:
            raise StopIteration


class AbstractDataset:
    def __init__(self, seed: Optional[int] = None):
        self.set_rng(seed)

    def set_rng(self, seed: Optional[int] = None):
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        self.np_rng = np_rng
        self.py_rng = py_rng

    @abstractmethod
    def get_document(self) -> str:
        raise NotImplementedError()


class C4Dataset(AbstractDataset):
    total_gpt2_tokens = 173_648_052_806  # number of tokens in the C4 dataset when using GPT2TokenizerFast

    def __init__(
        self,
        seed: Optional[int] = None,
        split: str = "train",
        use_dummy_dataset: bool = False,
        dataset_path: Optional[str] = None,
    ):
        super().__init__(seed=seed)
        assert split in ["train", "validation"]
        if dataset_path is not None:
            self.dataset = load_from_disk(dataset_path)
        elif use_dummy_dataset:
            if split != "train":
                raise NameError(
                    "Dummy dataset only supports train split for C4 dataset"
                )
            self.dataset = load_dataset("stas/c4-en-10k", split=split)
        else:
            self.dataset = load_dataset("c4", "en", split=split)

    def get_document(self) -> str:
        return self.dataset[self.py_rng.randint(0, len(self.dataset) - 1)]["text"]


def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


class AbstractTokenizer(ABC):
    VOCAB_SIZE: int
    sequence_separator_id: Optional[int]
    mask_id: Optional[int]
    eot_id: Optional[int]
    blanks_ids: Optional[List[int]]

    @abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError()


def disable_tokenizer_warnings(hf_tokenizer):
    # set model max length to high number to disable warnings
    # we handle sequence length ourselves
    hf_tokenizer.model_max_length = 100_000


class GPTTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        disable_tokenizer_warnings(self.tokenizer)
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)


class AbstractPacker(ABC, IterableDataset):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: Callable[[], AbstractDataset],
        tokenizer_maker: Callable[[], AbstractTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._tokenizer = None
        self._dataset = None
        self.dataset_maker = dataset_maker
        self.tokenizer_maker = tokenizer_maker
        self.sequence_length = sequence_length
        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.seed = seed

    def set_rng(self, seed: Optional[int] = None):
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        self.np_rng = np_rng
        self.py_rng = py_rng

        self.dataset.set_rng(seed)

    def __iter__(self) -> Iterator[LLMExample]:
        while True:
            yield self.get_sample()

    @abstractmethod
    def get_sample(self) -> LLMExample:
        raise NotImplementedError()

    @property
    def dataset(self) -> AbstractDataset:
        if self._dataset is None:
            self._dataset = self.dataset_maker()
            self._dataset.set_rng(self.seed)
        return self._dataset

    @property
    def tokenizer(self) -> AbstractTokenizer:
        if self._tokenizer is None:
            self._tokenizer = self.tokenizer_maker()
        return self._tokenizer


class GPTPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )

    def get_sample(self) -> LLMExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        assert eot_id is not None

        buffer: List[int] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document()
            tokens = self.tokenizer.text_to_ids(document)
            buffer.extend(tokens + [eot_id])

            document_lengths.append(len(tokens) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(buffer) - 1)
        sample_end = sample_start + self.sequence_length

        input_ids = list(take_circular(buffer, sample_start, sample_end))
        target_ids = list(take_circular(buffer, sample_start + 1, sample_end + 1))
        calculate_loss = [1] * len(target_ids)

        return LLMExample(input_ids, target_ids, calculate_loss)


class DataloaderWrapper:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.generator = iter(dataloader)
        self.device = device

    def get_batch(self) -> LLMBatch:
        return next(self.generator).to(self.device)


def worker_init_fn(seed, worker_id):
    worker_info = torch.utils.data.get_worker_info()
    packer: AbstractPacker = (
        worker_info.dataset
    )  # the dataset copy in this worker process
    packer.set_rng(seed + worker_id)


def get_processed_dataset(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    model_type: Literal["bert", "gpt"] = "bert",
    dataset_type: Literal["wikibook", "c4"] = "wikibook",
    use_dummy_dataset: bool = False,
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
):
    if dataset_type == "c4":
        dataset = partial(
            C4Dataset,
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
            dataset_path=dataset_path,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if model_type == "gpt":
        packer = GPTPacker(
            sequence_length=sequence_length,
            dataset_maker=dataset,
            tokenizer_maker=GPTTokenizer,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dataloader = DataLoader(
        packer,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=LLMBatch,
        worker_init_fn=partial(worker_init_fn, seed),
        shuffle=False,
        pin_memory=True,
    )

    return DataloaderWrapper(dataloader, device)


@dataclass
class AttentionConfig:
    mode: str
    dhead: int
    n_heads: int
    flash: bool


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
class Common:
    model_type: str
    sequence_length: int
    dmodel: int
    dff: int
    init_type: str
    init_scale: float


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

    def forward(self, x):
        out = self.layer(x)
        return out + x


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


class PredictionHead(Linear):
    def __init__(self, embedding_dim, output_size, init_type, init_scale):
        super(PredictionHead, self).__init__(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )


class LLM(nn.Module):

    def __init__(self, common: Common, tower_config: TowerConfig):
        super(LLM, self).__init__()

        embedding_components = [
            TokenEmbedding(
                GPTTokenizer.VOCAB_SIZE,
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
        ]
        self.embedding_layer = EmbeddingLayer(*embedding_components)

        self.encoder = TransformerTower(
            common=common,
            tower_config=tower_config,
        )

        self.head = PredictionHead(
            common.dmodel,
            GPTTokenizer.VOCAB_SIZE,
            init_type=common.init_type,
            init_scale=common.init_scale,
        )

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
    dhead: int,
    causal: bool,
    flash: bool,
):
    if flash:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                is_causal=causal,
            )
    else:
        # implementation without flash assumes other dim order
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", query, key)
        a = a * (1 / dhead**0.5)
        if causal:
            a.masked_fill_(
                torch.tril(torch.ones_like(a)) == 0, float("-inf")
            )  # mask out future tokens
        a = torch.softmax(a, dim=-1)
        output = torch.einsum("... h l L, ... L h d -> ... l h d", a, value)
        output = output.transpose(1, 2)

    return output


class AttentionMechanism(nn.Module):
    def __init__(self, use_flash_attention: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dhead: int,
        causal: bool,
        *args,
        **kwargs,
    ):
        return attention_mechanism(
            query=query,
            key=key,
            value=value,
            dhead=dhead,
            causal=causal,
            flash=self.use_flash_attention,
        )


class Attention(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


@define(slots=False)
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_accumulation_steps: int
    start_step: int
    n_steps: int
    train_dataloader: DataloaderWrapper

    def train(self):
        for _ in range(self.start_step, self.n_steps):
            self.model.train()
            processed_batch = self.train_dataloader.get_batch()
            processed_batch.set_n_chunks(self.gradient_accumulation_steps)

            losses = []
            for input_tokens, gt_tokens, mask in processed_batch:
                model_output = self.model(input_tokens)

                # Tensors should be on the same device for loss calculation
                gt_tokens = gt_tokens.to(model_output.device)
                mask = mask.to(model_output.device)

                mask_loss = F.cross_entropy(
                    model_output.flatten(0, -2),
                    gt_tokens.reshape(-1).long(),
                    reduction="none",
                )
                mask_loss = mask_loss[mask.reshape(-1) == 1]
                loss = mask_loss.mean() / self.gradient_accumulation_steps
                loss.backward()
                losses.append(loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            print(self.scheduler.get_last_lr())  # TODO
            print(np.sum(losses))  # TODO


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
            dhead=attention_config.dhead,
            flash=attention_config.flash,
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


def get_scheduler(optimizer, training_config):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=training_config.warmup_steps,
    )
    after_warmup_steps = training_config.n_steps - training_config.warmup_steps - 1
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=after_warmup_steps, eta_min=0.1 * training_config.learning_rate
    )
    training_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant_scheduler, cosine_scheduler],
        milestones=[training_config.warmup_steps, training_config.warmup_steps + 1],
    )
    return training_scheduler
