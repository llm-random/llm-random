from research.token_reduction.scheduler import (
    TokenReductionScheduler,
)
import unittest
class TestScheduler(unittest.TestCase):

    def test_linear_increase(self):
        scheduler = TokenReductionScheduler(ranges=[(1, 11, 0, 100)])
        expected_result = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        result = []
        for step in range(1, len(expected_result) + 1):
            scheduler.set_step(step)
            result.append(scheduler.value)
            scheduler.step()
        self.assertEqual(result, expected_result)

    def test_linear_decrease(self):
        scheduler = TokenReductionScheduler(ranges=[(1, 11, 100, 0)])
        expected_result = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
        result = []
        scheduler.set_step(1)
        for _ in range(1, len(expected_result) + 1):
            result.append(scheduler.value)
            scheduler.step()
        self.assertEqual(result, expected_result)

    def test_constant_value(self):
        scheduler = TokenReductionScheduler(ranges=[(1, 10, 50)])
        expected_result = [50] * 10
        result = []
        for step in range(1, len(expected_result) + 1):
            scheduler.set_step(step)
            result.append(scheduler.value)
            scheduler.step()
        self.assertEqual(result, expected_result)

    def test_mixed_ranges(self):
        scheduler = TokenReductionScheduler(
            ranges=[
                (1, 11, 0, 100),  # Linear from 0 to 100
                (12, 20, 100),  # Constant at 100
                (21, 30, 80, 20),  # Linear from 80 to 20
            ]
        )
        expected_result = (
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Linear
            + [100] * 9  # Constant
            + [80, 73, 67, 60, 53, 47, 40, 33, 27, 20]  # Linear
        )
        result = []
        for step in range(1, len(expected_result) + 1):
            scheduler.set_step(step)
            result.append(scheduler.value)
            scheduler.step()
        self.assertEqual(result, expected_result)

    def test_step_increment(self):
        scheduler = TokenReductionScheduler(ranges=[(1, 10, 5, 100)])
        step_inside = 5
        scheduler.set_step(step_inside)
        scheduler.step()
        self.assertEqual(scheduler.current_step, step_inside + 1)

    def test_max_step_constant(self):
        scheduler = TokenReductionScheduler(ranges=[(1, 10, 5)])
        scheduler.set_step(10)
        target_ver_max = 10
        for _ in range(target_ver_max):
            scheduler.step()
            self.assertEqual(scheduler.value, 5)

    def test_same_start_end(self):
        const_val = 4
        scheduler = TokenReductionScheduler(ranges=[(5, 30, const_val, const_val)])
        scheduler.set_step(2)
        for _ in range(36):
            scheduler.step()
            self.assertEqual(scheduler.value, const_val)

    def test_non_continuous_ranges(self):
        with self.assertRaises(ValueError):
            TokenReductionScheduler(ranges=[(1, 10, 0, 100), (12, 20, 100)])

    def test_overlapping_ranges(self):
        with self.assertRaises(ValueError):
            TokenReductionScheduler(ranges=[(1, 10, 0, 100), (9, 20, 100)])
