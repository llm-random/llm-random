import torch
import torch.nn.functional as F


def calculate_gpt_loss(batch, model, mixed_precision, vocab_size):
    input = batch.tokens
    target = batch.target_tokens
    non_padded_mask = batch.non_padded_mask

    if mixed_precision:
        with torch.autocast(
            device_type="cuda", enabled=mixed_precision, dtype=torch.float16
        ):
            model_output = model(input)
    else:
        model_output = model(input)

    lm_loss = F.cross_entropy(
        model_output.reshape(-1, vocab_size),
        target.reshape(-1).long(),
        reduction="none",
    )
    lm_loss *= non_padded_mask.reshape(-1)
    loss = lm_loss.mean()
    return loss


def calculate_bert_loss(batch, model, mixed_precision, vocab_size, mask_percent):
    x_set = batch.masked_tokens
    y_token_set = batch.tokens
    y_mask_set = batch.mask_mask

    if mixed_precision:
        with torch.autocast(
            device_type="cuda", enabled=mixed_precision, dtype=torch.float16
        ):
            model_output = model(x_set)
    else:
        model_output = model(x_set)

    mask_loss = F.cross_entropy(
        model_output.reshape(-1, vocab_size),
        y_token_set.reshape(-1).long(),
        reduction="none",
    )
    mask_loss *= y_mask_set.reshape(-1)
    loss = mask_loss.mean() / mask_percent
    return loss
