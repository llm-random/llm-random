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

    def __init__(
        self, transition_points, batch_sizes, transition_points_in="B", seq_len=None
    ):
        transition_points = self.convert_to_bilions_of_tokens(
            transition_points=transition_points,
            units=transition_points_in,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
        )
        self.transition_points: list[float] = transition_points
        self.batch_sizes: list[float] = batch_sizes

    def __post_init__(self):
        assert len(self.transition_points) == len(self.batch_sizes)

    def convert_to_bilions_of_tokens(
        self, transition_points, units, batch_sizes, seq_len
    ):
        if units == "B":
            return transition_points
        elif units == "M":
            return [tp * 1e-3 for tp in transition_points]
        elif units == "steps":
            transition_points_in_tokens = []
            steps_prev = tokens_prev = 0
            tokens_per_step_list = [seq_len * batch_size for batch_size in batch_sizes]
            for point, tokens_per_step in zip(transition_points, tokens_per_step_list):
                steps_needed = point - steps_prev
                tokens_needed = steps_needed * tokens_per_step
                tokens_current = tokens_prev + tokens_needed
                transition_points_in_tokens.append(tokens_current * 1e-9)
                tokens_prev = tokens_current
                steps_prev = point
            return transition_points_in_tokens
