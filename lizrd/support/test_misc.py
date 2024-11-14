from lizrd.support.misc import (
    convert_steps_to_tokens,
    convert_tokens_to_steps,
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

        config = BatchSizeRampupConfig(
            transition_points=transition_points,
            batch_sizes=batch_sizes,
            target_batch_size=target_batch_size,
            units="tokens",
            seq_len=seq_len,
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
                n_tokens=processed_tokens,
                seq_len=seq_len,
                rampup_config=config,
                target_batch_size=config.target_batch_size,
            )
            actual_batch_size = config.get_batch_size(step)
            self.assertEqual(actual_batch_size, expected_batch_size)

    def test_multiple_transitions(self):
        # Test with multiple transition points
        transition_points = [0.5, 1, 2]  # in billions
        batch_sizes = [16, 64, 128]
        target_batch_size = 512
        seq_len = 512

        config = BatchSizeRampupConfig(
            transition_points=transition_points,
            batch_sizes=batch_sizes,
            target_batch_size=target_batch_size,
            units="tokens",
            seq_len=seq_len,
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
                n_tokens=processed_tokens,
                seq_len=seq_len,
                rampup_config=config,
                target_batch_size=config.target_batch_size,
            )
            actual_batch_size = config.get_batch_size(step)
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
            target_batch_size=target_batch_size,
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

            self.assertEqual(actual_tokens, expected)
            self.assertEqual(actual_step, step)
