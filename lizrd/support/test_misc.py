from lizrd.support.misc import (
    calculate_current_batch_size_from_rampup,
    convert_steps_to_tokens,
    convert_tokens_to_steps,
)
from research.batch_size_rampup_config import BatchSizeRampupConfig
from lizrd.support.test_utils import GeneralTestCase


class CalculateCurrentBatchSizeTest(GeneralTestCase):
    def test_single_transition(self):
        # Test with a single transition point
        transition_points = [1]  # in billions
        batch_sizes = [16]
        target_batch_size = 64

        test_cases = [
            (0, 16),
            (0.5e9, 16),
            (1.01e9, 64),
            (1.5e9, 64),
            (2e9, 64),
        ]

        for processed_tokens, expected_batch_size in test_cases:
            actual_batch_size = calculate_current_batch_size_from_rampup(
                processed_tokens,
                transition_points,
                batch_sizes,
                target_batch_size,
            )
            self.assertEqual(actual_batch_size, expected_batch_size)

    def test_multiple_transitions(self):
        # Test with multiple transition points
        transition_points = [0.5, 1, 2]  # in billions
        batch_sizes = [16, 64, 128]
        target_batch_size = 512

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
            actual_batch_size = calculate_current_batch_size_from_rampup(
                processed_tokens,
                transition_points,
                batch_sizes,
                target_batch_size,
            )
            self.assertEqual(actual_batch_size, expected_batch_size)


class CalculateNProcessedTokensTest(GeneralTestCase):
    def basic_test(self):
        step = 10
        seq_len = 512
        target_batch_size = 512
        n_gpus = 8
        rampup_config = None

        expected_tokens = step * n_gpus * target_batch_size * seq_len

        actual_tokens = convert_steps_to_tokens(
            step,
            seq_len,
            target_batch_size,
            rampup_config,
        )

        actual_step = convert_tokens_to_steps(
            expected_tokens,
            seq_len,
            target_batch_size,
            rampup_config,
        )

        self.assertEqual(actual_tokens, expected_tokens)
        self.assertEqual(actual_step, step)

    def test_with_rampup(self):
        seq_len = 512
        target_batch_size = 512
        n_gpus = 2
        config = BatchSizeRampupConfig(
            [0.065536000, 0.327680000],
            [128, 256],
            units="tokens",
            seq_len=seq_len,
        )

        steps = [1000, 1500, 2000, 3000]
        expected_token_counts = [65536000, 131072000, 196608000, 327680000]

        for step, expected in zip(steps, expected_token_counts):
            actual_tokens = convert_steps_to_tokens(
                step, seq_len, target_batch_size, config
            )

            actual_step = convert_tokens_to_steps(
                expected, seq_len, target_batch_size, config
            )

            print(f"actual_step: {actual_step}\tactual_tokens: {actual_tokens}")
            print(f"config: {config.transition_points}\tconfig: {config.batch_sizes}")

            self.assertEqual(actual_tokens, expected)
            self.assertEqual(actual_step, step)
