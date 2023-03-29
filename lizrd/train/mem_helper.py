from typing import Callable


def make_justonce_scheduler(start: int, times: int) -> Callable[[int], bool]:
    def scheduler(epoch):
        if start <= epoch < start + times:
            return True
        return False

    return scheduler


def make_constant_scheduler(start: int, gap: int) -> Callable[[int], bool]:
    def scheduler(epoch):
        if start <= epoch and (epoch - start) % gap == 0:
            return True
        return False

    return scheduler


def make_backoff_scheduler(
    start: int, exponent: int, initial_gap: int = 1
) -> Callable[[int], bool]:
    assert exponent > 1

    def scheduler(epoch):
        if epoch < start:
            return False

        if epoch == start:
            return True

        previous_gaps = 0
        gap = initial_gap
        while epoch > start + previous_gaps + gap:
            previous_gaps += gap
            gap *= exponent

        if epoch == start + previous_gaps + gap:
            return True

        return False

    return scheduler


def create_trainer_scheduler(
    scheduler_type: str,
    **kwargs,
) -> Callable[[int], bool]:
    if scheduler_type == "justonce":
        return make_justonce_scheduler(**kwargs)
    elif scheduler_type == "constant":
        return make_constant_scheduler(**kwargs)
    elif scheduler_type == "backoff":
        return make_backoff_scheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def interpret_trainer_scheduler_string(scheduler_string: str) -> dict:
    scheduler_type, *args = scheduler_string.split(":")
    kwargs = {}
    for arg in args:
        k, v = arg.split("=")
        try:
            kwargs[k] = int(v)
        except ValueError:
            kwargs[k] = v
    return {"scheduler_type": scheduler_type, "kwargs": kwargs}
