from lizrd.support.misc import (
    calculate_current_batch_size_from_rampup,
    calculate_n_processed_tokens,
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
        target_batch_size_per_gpu = 64
        n_gpus = 8
        rampup_config = None

        expected_tokens = step * n_gpus * target_batch_size_per_gpu * seq_len

        actual_tokens = calculate_n_processed_tokens(
            step,
            seq_len,
            target_batch_size_per_gpu,
            n_gpus,
            rampup_config,
        )

        self.assertEqual(actual_tokens, expected_tokens)

    def test_with_rampup(self):
        seq_len = 512
        target_batch_size_per_gpu = 256
        n_gpus = 2
        config = BatchSizeRampupConfig([0.065536000, 0.327680000], [128, 256])

        steps = [1000, 1500, 2000, 3000]
        expected_token_counts = [65536000, 131072000, 196608000, 327680000]

        for step, expected in zip(steps, expected_token_counts):
            actual_tokens = calculate_n_processed_tokens(
                step, seq_len, target_batch_size_per_gpu, n_gpus, config
            )

            print(f"actual: {actual_tokens}, expected: {expected}")

            self.assertEqual(actual_tokens, expected)
