import unittest
from model import (
    RecorderLogger,
)


def dummy_calculation_function(**kwargs):
    # Example function that computes sum of each item
    results = {}
    for key, value in kwargs.items():
        if len(value) != 0:
            results[key + "_calculation"] = sum(value)
    return results


class TestRecorderLogger(unittest.TestCase):
    def test_accumulate_and_flush(self):
        # recorder_cfg = MetricLoggerConfig(type="recorder")
        metric_logger = RecorderLogger()
        # accumulate_metrics
        metric_logger.accumulate_metrics(
            layer_name="residual_block_1",
            calculate_fn=dummy_calculation_function,
            metrics={
                "t1": 1,
                "f1": 3.0,
            },
        )
        metric_logger.accumulate_metrics(
            layer_name="residual_block_1",
            calculate_fn=dummy_calculation_function,
            metrics={
                "t1": 4,
                "f1": 2.0,
            },
        )

        metric_logger.accumulate_metrics(
            layer_name="residual_block_2",
            calculate_fn=dummy_calculation_function,
            metrics={"t2": 2.0},
        )

        # flush_accumulated_metrics flushes everything to logger.data
        metric_logger.flush_accumulated_metrics(step=5)

        # Check the data
        # The logs will be stored under "steps/residual_block_1/t1_calculation", etc.
        self.assertIn(
            (5, 5), metric_logger.data["steps/residual_block_1/t1_calculation"]
        )
        self.assertIn(
            (5.0, 5), metric_logger.data["steps/residual_block_1/f1_calculation"]
        )

        # For residual_block_2:
        self.assertIn(
            ((2.0, 5)), metric_logger.data["steps/residual_block_2/t2_calculation"]
        )

        metric_logger.flush_accumulated_metrics(step=6)

        self.assertNotIn(
            ((0, 6)), metric_logger.data["steps/residual_block_2/t2_calculation"]
        )

        self.assertEqual(
            1, len(metric_logger.data["steps/residual_block_2/t2_calculation"])
        )

        metric_logger.accumulate_metrics(
            layer_name="residual_block_2",
            calculate_fn=dummy_calculation_function,
            metrics={"t2": 5},
        )
        metric_logger.accumulate_metrics(
            layer_name="residual_block_2",
            calculate_fn=dummy_calculation_function,
            metrics={"t2": 7},
        )
        metric_logger.flush_accumulated_metrics(step=7)
        self.assertEqual(
            2, len(metric_logger.data["steps/residual_block_2/t2_calculation"])
        )
        self.assertIn(
            (12, 7), metric_logger.data["steps/residual_block_2/t2_calculation"]
        )
