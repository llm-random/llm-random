from dataclasses import dataclass


@dataclass
class BatchSizeRampupConfig:
    """
    Configuration for ramping up the batch size during training.

    Attributes:
        transition_points (list[float]): A list of token counts (in billions) where the batch size
            transitions to the next value.
        batch_sizes (list[float]): A list of batch sizes corresponding to each transition point.

    Example:
        Given `transition_points = [0.5, 1.0]` and `batch_sizes = [128, 256]`:
            - The batch size will be 128 until 0.5 billion tokens are processed.
            - Then it will change to 256 until 1.0 billion tokens are processed.
            - After 1.0 billion tokens, the batch size reaches its target value.
    """

    transition_points: list[float]
    batch_sizes: list[float]

    def __post_init__(self):
        assert len(self.transition_points) == len(self.batch_sizes)
