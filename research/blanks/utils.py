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
