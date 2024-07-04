from research.token_reduction.scheduler import (
    TokenReductionScheduler,
)
import unittest


class TestScheduler(unittest.TestCase):
    def test_linear_increase(self):
        scheduler = TokenReductionScheduler(
            total_steps=4, seq_len=10, schedule_str="100_lin_1-5"
        )
        expected_result = [0, 10, 20, 30, 40]
        result = []
        for step in range(len(expected_result)):
            scheduler.set_step(step)
            result.append(scheduler.value)
        self.assertEqual(result, expected_result)

    def test_linear_decrease(self):
        scheduler = TokenReductionScheduler(
            total_steps=8, seq_len=10, schedule_str="100_lin_5-1"
        )
        expected_result = [40, 35, 30, 25, 20, 15, 10, 5, 0]
        result = []
        for step in range(len(expected_result)):
            scheduler.set_step(step)
            result.append(scheduler.value)
        self.assertEqual(result, expected_result)

    def test_constant_value(self):
        scheduler = TokenReductionScheduler(
            total_steps=10, seq_len=25, schedule_str="100_const_3"
        )
        expected_result = [50] * 10
        result = []
        for step in range(len(expected_result)):
            scheduler.set_step(step)
            result.append(scheduler.value)
        self.assertEqual(result, expected_result)

    def test_below_sum_percentages(self):
        with self.assertRaises(ValueError):
            TokenReductionScheduler(10, 20, "80_lin_5-1;10_const_3")

    def test_above_sum_percentages(self):
        with self.assertRaises(ValueError):
            TokenReductionScheduler(10, 20, "80_lin_5-1;30_const_3")


# TODO(anyone) Add more tests, especially for cosine schedule
