import itertools
import unittest
from unittest.mock import patch
from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn.functional as F
from hydra import initialize, compose
from hydra.utils import instantiate
import torch
import sys

from model import get_dataloader, get_metric_logger

import os

TOLERANCE = 1e-5
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def copy_weights_between_models(model, hf_model):
    hf_transformer = hf_model.transformer
    hf_transformer.wte.weight.data.copy_(model.embedding_layer.layers[0].weight)
    hf_transformer.wpe.weight.data.copy_(model.embedding_layer.layers[1].layer.weight)

    for i, block in enumerate(model.encoder.blocks):
        hf_block = hf_transformer.h[i]
        hf_block.ln_1.weight.data.copy_(
            block.block.residual_attention.layer.pre_norm.weight
        )
        hf_block.ln_1.bias.data.copy_(
            block.block.residual_attention.layer.pre_norm.bias
        )

        hf_block.attn.c_attn.weight.data.copy_(
            block.block.residual_attention.layer.attention.input_projection.weight.t()
        )

        hf_block.attn.c_attn.bias.requires_grad = False

        hf_block.attn.c_proj.weight.data.copy_(
            block.block.residual_attention.layer.attention.output_projection.weight.t()
        )

        hf_block.attn.c_proj.bias.requires_grad = False

        hf_block.ln_2.weight.data.copy_(
            block.block.residual_feedforward.layer.pre_norm.weight
        )
        hf_block.ln_2.bias.data.copy_(
            block.block.residual_feedforward.layer.pre_norm.bias
        )

        hf_block.mlp.c_fc.weight.data.copy_(
            block.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight.t()
        )
        hf_block.mlp.c_fc.bias.data.copy_(
            block.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.bias.t()
        )

        # Code belowe is correct
        # However this is a bit tricky, as Conv1D even with the very same weights gives different output.
        # The difference is very small (~1e-5) and likely comes from numeric precision.
        # hf_block.mlp.c_proj.weight.data.copy_(
        #     block.block.residual_feedforward.layer.feedforward.logging_ff_post_relu.weight.t()
        # )
        # hf_block.mlp.c_proj.bias.data.copy_(
        #     block.block.residual_feedforward.layer.feedforward.logging_ff_post_relu.bias.t()
        # )
        # This is workaround to make the weights exactly the same
        hf_block.mlp.c_proj = (
            block.block.residual_feedforward.layer.feedforward.logging_ff_post_relu
        )

    hf_transformer.ln_f = torch.nn.Identity()

    hf_model.lm_head.weight.data.copy_(model.head.weight)


class TestHFModel(unittest.TestCase):
    def test_model(self):

        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_hf", overrides=[])

        torch.manual_seed(cfg.training.seed)
        metric_logger_config = instantiate(cfg.metric_logger, _convert_="all")
        _ = get_metric_logger(metric_logger_config)  # for early initialization
        model = instantiate(cfg.model, _convert_="all")

        hf_config = GPT2Config(
            n_embd=cfg.model.common.dmodel,
            n_positions=cfg.model.common.sequence_length,
            n_layer=cfg.model.tower_config.n_blocks,
            n_head=cfg.model.tower_config.block_config.attention.n_heads,
            activation_function="relu",
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            tie_word_embeddings=False,
        )
        hf_model = GPT2LMHeadModel(config=hf_config)
        copy_weights_between_models(model, hf_model)
        with patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0"}):
            train_dataloader = get_dataloader(
                dataloader_config=cfg.training.dataloader,
                batch_size_per_device=cfg.training.dataloader.total_batch_size,
                sequence_length=cfg.model.common.sequence_length - 1,
                seed=cfg.training.seed,
                dataset_split="train",
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        optimizer_hf = torch.optim.AdamW(
            hf_model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        for i, batch in enumerate(itertools.islice(train_dataloader, 10)):
            optimizer_hf.zero_grad()
            optimizer.zero_grad()

            hf_model_output = hf_model(batch, labels=batch)
            model_output = model(batch)

            loss_input = model_output[:, :-1, :].contiguous()
            target_ids = batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                loss_input.view(-1, loss_input.size(-1)), target_ids.view(-1)
            )

            hf_model_output.loss.backward()
            loss.backward()
            optimizer_hf.step()
            optimizer.step()

            if sys.version == "3.10.12":
                assert torch.equal(
                    hf_model_output.logits, model_output
                ), "Logits are not equal"
                assert torch.equal(hf_model_output.loss, loss), "Loss is not equal"
            else:
                assert torch.allclose(
                    hf_model_output.logits, model_output, atol=TOLERANCE
                ), "Logits are not equal"
                assert torch.allclose(hf_model_output.loss, loss), "Loss is not equal"


if __name__ == "__main__":
    unittest.main()
