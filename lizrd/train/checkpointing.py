from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

from torch.utils.checkpoint import checkpoint
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

_IS_IN_FIRST_FORWARD: ContextVar[bool] = ContextVar(
    "is_in_first_forward", default=False
)
_IS_IN_SECOND_FORWARD: ContextVar[bool] = ContextVar(
    "is_in_second_forward", default=False
)


def is_in_first_forward() -> bool:
    return _IS_IN_FIRST_FORWARD.get()


def is_in_second_forward() -> bool:
    return _IS_IN_SECOND_FORWARD.get()


@contextmanager
def first_forward_manager():
    token = _IS_IN_FIRST_FORWARD.set(True)
    try:
        yield
    finally:
        _IS_IN_FIRST_FORWARD.reset(token)


@contextmanager
def second_forward_manager():
    token = _IS_IN_SECOND_FORWARD.set(True)
    try:
        yield
    finally:
        _IS_IN_SECOND_FORWARD.reset(token)


def make_checkpoint_wrapper_function():
    return partial(
        checkpoint_wrapper,
        checkpoint_fn=partial(
            checkpoint,
            context_fn=lambda: (first_forward_manager(), second_forward_manager()),
            use_reentrant=False,
        ),
    )
