from lizrd.support.misc import (
    convert_steps_to_tokens,
    convert_tokens_to_steps,
    convert_transition_points_in_tokens_to_steps,
    get_batch_size,
)
from research.batch_size_rampup_config import BatchSizeRampupConfig
from lizrd.support.test_utils import GeneralTestCase


class CalculateCurrentBatchSizeTest(GeneralTestCase):
    def test_single_transition(self):
        # Test with a single transition point
        transition_points = [1]  # in billions
        batch_sizes = [100]
        target_batch_size = 1000
        seq_len = 100

        transition_points = convert_transition_points_in_tokens_to_steps(
            transition_points_in_tokens=transition_points,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
        )

        config = BatchSizeRampupConfig(
            transition_points=transition_points,
            batch_sizes=batch_sizes,
        )

        test_cases = [
            (0, 100),
            (0.5e9, 100),
            (1.01e9, 1000),
            (1.5e9, 1000),
            (2e9, 1000),
        ]

        for processed_tokens, expected_batch_size in test_cases:
            step = convert_tokens_to_steps(
                tokens=processed_tokens,
                seq_len=seq_len,
                target_batch_size=target_batch_size,
                transition_points=transition_points,
                batch_sizes=batch_sizes,
            )
            actual_batch_size = get_batch_size(
                step=step,
                target_batch_size=target_batch_size,
                transition_points=config.transition_points,
                batch_sizes=config.batch_sizes,
            )
            self.assertEqual(actual_batch_size, expected_batch_size)

    def test_multiple_transitions(self):
        # Test with multiple transition points
        transition_points = [0.5, 1, 2]  # in billions
        batch_sizes = [16, 64, 128]
        target_batch_size = 512
        seq_len = 512

        transition_points = convert_transition_points_in_tokens_to_steps(
            transition_points_in_tokens=transition_points,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
        )

        config = BatchSizeRampupConfig(
            transition_points=transition_points,
            batch_sizes=batch_sizes,
        )

        test_cases = [
            (0, 16),
            (0.3e9, 16),
            (0.8e9, 64),
            (1.5e9, 128),
            (1.9e9, 128),
            (2.5e9, 512),
            (3e9, 512),
            (50e9, 512),
        ]

        for processed_tokens, expected_batch_size in test_cases:
            step = convert_tokens_to_steps(
                tokens=processed_tokens,
                seq_len=seq_len,
                target_batch_size=target_batch_size,
                transition_points=transition_points,
                batch_sizes=batch_sizes,
            )
            actual_batch_size = get_batch_size(
                step=step,
                target_batch_size=target_batch_size,
                transition_points=config.transition_points,
                batch_sizes=config.batch_sizes,
            )
            self.assertEqual(actual_batch_size, expected_batch_size)

class ConvertTransitionPointsInTokensToStepsTest
    def test_single_transition(self):
        # Test with a single transition point
        transition_points = [0.001, 0.01, 0.1, 1.0]  # in billions
        batch_sizes = [1e2, 1e3, 1e4, 1e5]
        seq_len = 1e3

        actual_transition_points = convert_transition_points_in_tokens_to_steps(
            transition_points_in_tokens=transition_points,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
        )

        # for example we need 10 steps with bs 1e4 and seq_len 1e3 to get 1e8 tokens, but one 10th of the tokens was completed before we switched batch size
        expected_transition_points = [10, 19, 28, 37]
        self.assertEqual(actual_transition_points, expected_transition_points)


class CalculateNProcessedTokensTest(GeneralTestCase):
    def basic_test(self):
        step = 10
        seq_len = 512
        target_batch_size = 512
        n_gpus = 8

        expected_tokens = step * n_gpus * target_batch_size * seq_len

        actual_tokens = convert_steps_to_tokens(
            step,
            seq_len,
            target_batch_size,
        )

        actual_step = convert_tokens_to_steps(
            expected_tokens,
            seq_len,
            target_batch_size,
        )

        self.assertEqual(actual_tokens, expected_tokens)
        self.assertEqual(actual_step, step)

    def test_with_rampup(self):
        seq_len = 512
        target_batch_size = 512
        transition_points_in_tokens = [0.065536000, 0.327680000]
        batch_sizes = [128, 256]

        transition_points = convert_transition_points_in_tokens_to_steps(
            transition_points_in_tokens=transition_points_in_tokens,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
        )

        config = BatchSizeRampupConfig(
            transition_points=transition_points,
            batch_sizes=batch_sizes,
        )

        steps = [1000, 1500, 2000, 3000]
        expected_token_counts = [65536000, 131072000, 196608000, 327680000]

        for step, expected in zip(steps, expected_token_counts):
            actual_tokens = convert_steps_to_tokens(
                step,
                seq_len,
                target_batch_size,
                config.transition_points,
                config.batch_sizes,
            )

            actual_step = convert_tokens_to_steps(
                expected,
                seq_len,
                target_batch_size,
                config.transition_points,
                config.batch_sizes,
            )

            self.assertEqual(actual_tokens, expected)
            self.assertEqual(actual_step, step)
