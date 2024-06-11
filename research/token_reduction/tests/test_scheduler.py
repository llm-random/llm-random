from research.token_reduction.scheduler import LinearTokenReductionScheduler
import unittest


class TestScheduler(unittest.TestCase):

    def test_increase(self):
        scheduler = LinearTokenReductionScheduler(
            start=0, stop=100, steps=10
        )
        expected_result = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100]
        result = []
        for _ in range(len(expected_result)):
            result.append(scheduler.value)
            scheduler.step()
        self.assertEqual(result, expected_result)

    def test_decrease(self):
        scheduler = LinearTokenReductionScheduler(start=100, stop=0, steps=10)
        expected_result = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 0]
        result = []
        for _ in range(len(expected_result)):
            result.append(scheduler.value)
            scheduler.step()
        self.assertEqual(result, expected_result)

    def test_value_within_range(self):
        scheduler = LinearTokenReductionScheduler(start=0, stop=100, steps=10)
        for _ in range(scheduler.steps + 1):
            scheduler.step()
            value = scheduler.value
            self.assertGreaterEqual(value, scheduler.start)
            self.assertLessEqual(value, scheduler.stop)

    def test_step_increment(self):
        scheduler = LinearTokenReductionScheduler(start=5, stop=100, steps=10)
        step_inside = 5
        scheduler.set_step(step_inside)
        scheduler.step()
        self.assertEqual(scheduler.current_step, step_inside + 1)

    def test_max_step(self):
        max_step = 10
        stop_value = 100
        scheduler = LinearTokenReductionScheduler(
            start=5, stop=stop_value, steps=max_step
        )
        scheduler.set_step(max_step)
        for _ in range(10):
            scheduler.step()
            self.assertEqual(scheduler.value, stop_value)
