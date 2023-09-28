from functools import partial
import torch
from typing import Optional

from lizrd.text.data import LLMBatch
from torch.utils.checkpoint import checkpoint

import torch.nn.functional as F

from research.blanks.utils import (
    get_first_blanks_in_series,
    iterate_through_nth_blanks_masks,
)


def make_loss_function(loss_checkpoint_chungs: int, n_blanks: int = 0):
    if loss_checkpoint_chungs == 0:
        return partial(calculate_llm_loss, n_blanks=n_blanks)
    else:
        if n_blanks > 0:
            raise NotImplementedError(
                "Loss checkpointing not supported for blanks (yet)"
            )
        return partial(chungized_llm_loss, n_chungs=loss_checkpoint_chungs)


def chungized_llm_loss(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    vocab_size: int,
    n_chungs: int,
):
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

    def make_custom_forward():
        def custom_forward(*inputs):
            output = model.head(inputs[0])
            with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float16):
                gt = inputs[1]
                mask = inputs[2]
                gt = gt.to(output.device)
                loss = F.cross_entropy(
                    output.reshape(-1, vocab_size),
                    gt.reshape(-1).long(),
                    reduction="none",
                )

                correct_tokens = gt.long() == output.argmax(dim=-1)
                correct_tokens = correct_tokens.long().reshape(-1) * mask.reshape(-1)
                correct_tokens = correct_tokens.sum()

                total_tokens = mask.sum()

            return loss[mask.reshape(-1) == 1], correct_tokens, total_tokens

        return custom_forward

    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=torch.float16
    ):
        encoder_output = model.encoder(input_tokens)
        chunged_inputs = torch.chunk(encoder_output, n_chungs, dim=0)
        chunged_non_masked_inputs = torch.chunk(gt_tokens, n_chungs, dim=0)
        chunged_non_masked_masks = torch.chunk(mask, n_chungs, dim=0)

        num_tokens = 0
        total_loss = 0
        total_correct_tokens = 0
        total_masked_tokens = 0
        for chunged_input, chunged_gt, chunged_mask in zip(
            chunged_inputs, chunged_non_masked_inputs, chunged_non_masked_masks
        ):
            (
                partial_loss_output,
                partial_correct_tokens,
                partial_masked_tokens,
            ) = checkpoint(
                make_custom_forward(), chunged_input, chunged_gt, chunged_mask
            )
            num_tokens += partial_loss_output.shape[0]
            total_loss += partial_loss_output.sum()
            total_correct_tokens += partial_correct_tokens
            total_masked_tokens += partial_masked_tokens

        aux_info = {
            "correct_tokens": total_correct_tokens,
            "total_masked_tokens": total_masked_tokens,
            "losses": {},
        }

        return total_loss / num_tokens, aux_info


def calculate_llm_loss(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    vocab_size: int,
    n_blanks: int = 0,
    blank_id: Optional[int] = 50257,
):
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=torch.float16
    ):
        model_output = model(input_tokens)

    # move the gt tokens and mask to the same device as the model output - they should be on the same device for loss calculation
    gt_tokens = gt_tokens.to(model_output.device)
    mask = mask.to(model_output.device)

    mask_loss = F.cross_entropy(
        model_output.reshape(-1, vocab_size),
        gt_tokens.reshape(-1).long(),
        reduction="none",
    )
    mask_loss = mask_loss[mask.reshape(-1) == 1]
    loss = mask_loss.mean()

    blanks_losses = {}
    if n_blanks > 0:
        assert blank_id is not None
        with torch.no_grad():
            is_blank = input_tokens.eq(blank_id)
            total_blanks_count = is_blank.sum()
            if total_blanks_count > 0:
                blank_start = get_first_blanks_in_series(is_blank)

                for n, nth_blank_mask in enumerate(
                    iterate_through_nth_blanks_masks(
                        blank_start, n_blanks, include_preblank=True
                    )
                ):
                    nth_blanks_count = nth_blank_mask.sum()
                    assert nth_blanks_count * n_blanks == total_blanks_count
                    if n == 0:
                        assert not input_tokens[nth_blank_mask == 1].eq(blank_id).any()
                    else:
                        assert input_tokens[nth_blank_mask == 1].eq(blank_id).all()
                    nth_blank_loss = (nth_blank_mask.reshape(-1) * mask_loss).sum()
                    blanks_losses[f"blank_{n}_loss"] = (
                        nth_blank_loss / nth_blanks_count if nth_blanks_count > 0 else 0
                    )

    correct_tokens = gt_tokens.long() == model_output.argmax(dim=-1)
    correct_tokens = correct_tokens.long().reshape(-1) * mask.reshape(-1)
    correct_tokens = correct_tokens.sum()
    total_masked_tokens = mask.sum()

    aux_info = {
        "correct_tokens": correct_tokens,
        "total_masked_tokens": total_masked_tokens,
        "losses": {},
        "blanks_losses": blanks_losses,
    }

    return loss, aux_info
