from lizrd.support.misc import calculate_current_bsz_from_rampup
from lizrd.support.test_utils import GeneralTestCase


class CalculateCurrentBszTest(GeneralTestCase):
    def test_single_transition(self):
        # Test with a single transition point
        transition_points = [1]  # in billions
        batch_sizes = [16, 32]
        target_batch_size = 64

        test_cases = [
            (0, 16),
            (0.5e9, 16),
            (1e9, 32),
            (1.5e9, 32),
            (2e9, 64),
        ]

        for processed_tokens, expected_batch_size in test_cases:
            actual_batch_size = calculate_current_bsz_from_rampup(
                processed_tokens,
                transition_points,
                batch_sizes,
                target_batch_size,
            )
            self.assertEqual(actual_batch_size, expected_batch_size)

    def test_multiple_transitions(self):
        # Test with multiple transition points
        transition_points = [1, 2, 3]  # in billions
        batch_sizes = [16, 32, 48, 64]
        target_batch_size = 80

        test_cases = [
            (0, 16),
            (0.5e9, 16),
            (1e9, 32),
            (1.5e9, 32),
            (2e9, 48),
            (2.5e9, 48),
            (3e9, 64),
            (3.5e9, 80),
            (4e9, 80),
        ]

        for processed_tokens, expected_batch_size in test_cases:
            actual_batch_size = calculate_current_bsz_from_rampup(
                processed_tokens,
                transition_points,
                batch_sizes,
                target_batch_size,
            )
            self.assertEqual(actual_batch_size, expected_batch_size)

    def test_exact_transition_points(self):
        # Test with processed tokens exactly at transition points
        transition_points = [1, 2]  # in billions
        batch_sizes = [16, 32, 48]
        target_batch_size = 64

        test_cases = [
            (1e9, 32),
            (2e9, 48),
            (3e9, 64),
        ]

        for processed_tokens, expected_batch_size in test_cases:
            actual_batch_size = calculate_current_bsz_from_rampup(
                processed_tokens,
                transition_points,
                batch_sizes,
                target_batch_size,
            )
            self.assertEqual(actual_batch_size, expected_batch_size)
