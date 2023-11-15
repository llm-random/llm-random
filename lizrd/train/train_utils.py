from typing import Callable, Optional
import torch
from lizrd.core import llm


def get_model(
    max_length: int,
    vocab_size: int,
    ff_layer_fun: Callable[[], torch.nn.Module],
    attention_layer_fun: Callable[[], torch.nn.Module],
    dm: int,
    n_blocks: int,
    device: torch.device,
    init_type,
    init_scale,
    gradient_checkpointing: bool = False,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_layer = llm.EmbeddingLayer(
        llm.PositionalEmbedding(
            max_length, dm, init_type=init_type, init_scale=init_scale
        ).to(first_gpu),
        llm.TokenEmbedding(
            vocab_size, dm, init_type=init_type, init_scale=init_scale
        ).to(first_gpu),
    )

    layer_dict = {"attention": attention_layer_fun, "feedforward": ff_layer_fun}
    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = llm.TransformerTower(
        n_blocks,
        dm,
        layer_dict,
        gradient_checkpointing,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn,
    )

    head = llm.PredictionHead(
        dm, vocab_size, init_type=init_type, init_scale=init_scale
    ).to(last_gpu)

    model = llm.LLM(embedding_layer, encoder_tower, head)

    return model
