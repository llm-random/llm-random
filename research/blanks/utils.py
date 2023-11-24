from typing import List, Union
import torch
import torch.nn.functional as F
import numpy as np


def shift_left(x: torch.Tensor):
    """Shift the elements of the tensor to the left by one.

    Args:
        x: Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, hidden_size).

    Returns:
        Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, hidden_size).
    """

    return torch.cat([x[:, 1:], x[:, :1] * 0], dim=1)


def shift_right(x: torch.Tensor):
    """Shift the elements of the tensor to the right by one.

    Args:
        x: Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, hidden_size).

    Returns:
        Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, hidden_size).
    """

    return torch.cat([x[:, :1] * 0, x[:, :-1]], dim=1)


def get_first_blanks_in_series(is_blank: torch.Tensor):
    blank_start = (
        (
            F.conv1d(
                is_blank[:, None, :].float(),
                torch.tensor([-1.0, 1.0, 0.0], device=is_blank.device).reshape(
                    1, 1, -1
                ),
                padding="same",
            )
            == 1
        )
        .float()
        .squeeze_(1)
    )

    return blank_start


def get_last_blanks_in_series(is_blank: torch.Tensor):
    blank_end = (
        (
            F.conv1d(
                is_blank[:, None, :].float(),
                torch.tensor([0.0, 1.0, -1.0], device=is_blank.device).reshape(
                    1, 1, -1
                ),
                padding="same",
            )
            == 1
        )
        .float()
        .squeeze_(1)
    )

    return blank_end


def get_preblanks(is_blank: torch.Tensor):
    first_blanks = get_first_blanks_in_series(is_blank)
    preblanks = shift_left(first_blanks)

    return preblanks


def iterate_through_nth_blanks_masks(
    blank_start: torch.Tensor, n_blanks: int, include_preblank: bool
):
    working_copy = blank_start.clone()
    if include_preblank:
        working_copy = shift_left(working_copy)
        yield working_copy
        working_copy = shift_right(working_copy)

    for _ in range(n_blanks):
        yield working_copy
        working_copy = shift_right(working_copy)


def insert_blanks_input(
    input_sequence: List[int],
    blank_ids: List[int],
    blank_insertion_point: int,
    n_blanks: int,
) -> List[int]:
    assert len(blank_ids) == n_blanks

    return (
        input_sequence[:blank_insertion_point]
        + blank_ids
        + input_sequence[blank_insertion_point:]
    )[: len(input_sequence)]


def insert_blanks_target(
    target_sequence: List[int], blank_insertion_point: int, n_blanks: int
) -> List[int]:
    return (
        target_sequence[:blank_insertion_point]
        + [target_sequence[blank_insertion_point - 1]] * n_blanks
        + target_sequence[blank_insertion_point:]
    )[: len(target_sequence)]


def get_last_point_to_fit_blanks(sequence_length: int, n_blanks: int) -> int:
    return sequence_length - n_blanks


def can_fit_blanks(
    sequence_length: int, blank_insertion_point: int, n_blanks: int
) -> bool:
    return blank_insertion_point <= get_last_point_to_fit_blanks(
        sequence_length, n_blanks
    )


def make_blanks_attention_mask(seq_len, blanks_insertion_point, n_blanks) -> np.ndarray:
    """Generates attention mask input with blanks.

    Generates causal mask, where no tokens can attend to blanks except those that are in the same block and after the blank.

    Args:
        seq_len (int): length of the sequence
        blanks_insertion_point (int): where the blanks are inserted
        n_blanks (int): number of blanks in one block in the sequence

    Returns:
        torch.Tensor: which pairs take part in attention
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.bool))
    blanks_insertion_end = blanks_insertion_point + n_blanks
    mask[blanks_insertion_end:, blanks_insertion_point:blanks_insertion_end] = False
    return mask


def make_blanks_fixed_positions(
    x: torch.Tensor, blank_tokens_ids: torch.Tensor, n_blanks_block: int
) -> torch.Tensor:
    """Generates positions in a sequence taking blanks into account.
    Assumes that every sequence of blanks is exactly n_blanks long.

    Args:
        x (torch.Tensor): Input tokens of shape (batch_size, seq_len).
        blank_token_id (int): id of the blank token.
        n_blanks (int): number of blanks in one block in the sequence.

    Returns:
        torch.Tensor: positions of tokens in the sequence taking blanks into account.
    """
    positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
    positions = positions.repeat(x.shape[0], 1)
    is_blank = get_is_blank(x, blank_tokens_ids)
    n_blanks_up_to = is_blank.cumsum(dim=1)
    positions = positions - n_blanks_up_to
    n_blanks = is_blank.sum()
    if n_blanks % n_blanks_block != 0:
        raise ValueError(
            f"""Number of blanks in the sequence must be divisible by {n_blanks_block}. 
            It probably means that not all blocks of blanks are of the same length."""
        )
    positions[is_blank] += torch.arange(1, n_blanks_block + 1, device=x.device).repeat(
        n_blanks // n_blanks_block
    )
    return positions


def get_is_blank(
    x: torch.Tensor, blank_tokens_ids: Union[List[int], torch.Tensor]
) -> torch.Tensor:
    if isinstance(blank_tokens_ids, list):
        blank_tokens_ids = torch.tensor(blank_tokens_ids, device=x.device)

    return torch.isin(x, blank_tokens_ids)
