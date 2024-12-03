import numpy as np

from lizrd.support.test_utils import GeneralTestCase
from research.batch_size_rampup_config import BatchSizeRampupConfig


def check_configs_equality(target_config: dict, rampup_config: BatchSizeRampupConfig):
    for key in target_config.keys():
        assert np.allclose(target_config[key], rampup_config.key, atol=1e-5)


class TestBatchSizeRampupConfig(GeneralTestCase):
    def test_basic(self):
        target_config = {
            "transition_points": [1.0, 2.0, 3.0],
            "batch_sizes": [1, 10, 100],
        }

        # test transition points in B tokens option
        tps = [100, 110, 111]
        bszs = [1, 10, 100]

        bs_config_steps = BatchSizeRampupConfig(
            transition_points=tps,
            batch_sizes=bszs,
            target_batch_size=101,
            units="steps",
            seq_len=10000000,
        )
        check_configs_equality(target_config, bs_config_steps)

        # test transition points in M tokens option
        tps = [1000, 2000, 3000]
        bszs = [1, 10, 100]

        bs_config_steps = BatchSizeRampupConfig(
            transition_points=tps,
            batch_sizes=bszs,
            target_batch_size=101,
            units="steps",
            seq_len=10000000,
        )
        check_configs_equality(target_config, bs_config_steps)

        # test transition points in steps option
        tps = [100, 110, 111]
        bszs = [1, 10, 100]

        bs_config_steps = BatchSizeRampupConfig(
            transition_points=tps,
            batch_sizes=bszs,
            target_batch_size=101,
            units="steps",
            seq_len=10000000,
        )
        check_configs_equality(target_config, bs_config_steps)
