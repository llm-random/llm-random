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
        self,
        transition_points,
        batch_sizes,
        units="tokens",
        seq_len=None,
    ):
        transition_points = self.convert_transition_points_to_steps(
            transition_points=transition_points,
            units=units,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
        )
        self.transition_points: list[int] = transition_points
        self.batch_sizes: list[int] = batch_sizes

    def __post_init__(self):
        assert len(self.transition_points) == len(self.batch_sizes)

    def convert_transition_points_to_steps(
        self, transition_points, units, batch_sizes, seq_len
    ):
        if units == "steps":
            return transition_points
        elif units == "tokens":
            transition_points_in_steps = []
            transition_points = [p * 1e9 for p in transition_points]
            tokens_per_step_list = [batch_size * seq_len for batch_size in batch_sizes]
            steps_prev = tokens_prev = 0

            for point, tokens_per_step in zip(transition_points, tokens_per_step_list):
                tokens_to_transition = point - tokens_prev
                steps_to_transition = tokens_to_transition // tokens_per_step
                point_in_steps = steps_prev + steps_to_transition

                transition_points_in_steps.append(int(point_in_steps))

                tokens_prev = point
                steps_prev = point_in_steps
            return transition_points_in_steps
