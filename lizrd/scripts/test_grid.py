import unittest

from lizrd.scripts.grid_utils import timestr_to_minutes


class Test_timestr_to_minutes(unittest.TestCase):
    def test_basic(self):
        # Slurm manual states it accepts the following formats:
        # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
        for input_, expected_output in [
            # minutes
            ("0", 0),
            ("98", 98),
            ("98", 98),
            # minutes:seconds
            ("0:00", 0),
            ("3:120", 5),
            # hours:minutes:seconds
            ("0:00:00", 0),
            ("17:08:60", 17 * 60 + 9),
            # days-hours
            ("1-4", 1 * 24 * 60 + 4 * 60),
            # days-hours:minutes
            ("1-4:33", 1 * 24 * 60 + 4 * 60 + 33),
            # days-hours:minutes:seconds
            ("1-33:67:60", 1 * 24 * 60 + 33 * 60 + 67 + 1),
        ]:
            assert timestr_to_minutes(input_) == expected_output
