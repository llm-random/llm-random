import torch


def decode_single_example(
    model: torch.nn.Module,
    max_sequence_length: int,
    input_tokens_ids: torch.Tensor,
    end_token_id: int,
) -> torch.Tensor:
    output_tokens_ids = torch.nn.functional.pad(
        input_tokens_ids, (0, max_sequence_length - len(input_tokens_ids))
    )
    output_length = len(input_tokens_ids)
    model.eval()
    with torch.no_grad():
        while True:
            predictions = model(output_tokens_ids)
            next_token_id = torch.argmax(predictions, dim=-1)[output_length - 1].item()
            output_tokens_ids[output_length] = next_token_id
            output_length += 1
            if output_length == max_sequence_length or next_token_id == end_token_id:
                break
    return output_tokens_ids[:output_length]
