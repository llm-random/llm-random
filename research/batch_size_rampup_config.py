from dataclasses import dataclass


@dataclass
class BatchSizeRampupConfig:
    """
    Configuration for ramping up the batch size during training.

    :param transition_points: Steps at which the batch size transitions to the next value. (list[int])
    :param batch_sizes: Batch sizes corresponding to each transition point. (list[int])
    :example: Given `transition_points = [100, 300]` and `batch_sizes = [128, 256]`:
              - The batch size will be 128 until the 100th step.
              - Then it will change to 256 until the 300th step.
              - After 300 steps, the batch size reaches its target value.
    """

    def __init__(
        self,
        transition_points,
        batch_sizes,
    ):
        self.transition_points: list[int] = transition_points
        self.batch_sizes: list[int] = batch_sizes

    def __post_init__(self):
        assert len(self.transition_points) == len(self.batch_sizes)
