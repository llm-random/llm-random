from typing import List
import torch
import torch.nn.functional as F


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
    input_sequence: List[int], blank_id: int, blank_insertion_point: int, n_blanks: int
) -> List[int]:
    return (
        input_sequence[:blank_insertion_point]
        + [blank_id] * n_blanks
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


def make_blanks_fixed_positions(x: torch.Tensor, blank_token_id: int):
    # stack overflow https://stackoverflow.com/questions/18196811/cumsum-reset-at-nan
    final_positions = torch.zeros_like(x)
    for i, example in enumerate(x):
        positions = torch.arange(0, len(example), device=example.device)
        is_blank = example.eq(blank_token_id)
        n_blanks_up_to = is_blank.cumsum(0)
        positions = positions - n_blanks_up_to  # adjecent blanks have the same position
        is_not_blank = ~is_blank
        num_adjacent_blanks = torch.diff(
            n_blanks_up_to[is_not_blank],
            prepend=torch.zeros(1, device=example.device),
        ).long()  # for each non-blank token, how many blanks are immediately before it
        is_blank = is_blank.long()
        is_blank[
            is_not_blank
        ] = -num_adjacent_blanks  # now it looks like [0, 1, 1, -2, 0, 1, 1, 1, -3, ..]
        fixup = is_blank.cumsum(0)  # [0, 1, 2, 0, 0, 1, 2, 3, 0, ..]
        final_positions[i, :] = positions + fixup
    return final_positions
