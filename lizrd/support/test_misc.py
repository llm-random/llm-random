from lizrd.support.misc import calculate_current_batch_size_from_rampup
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
