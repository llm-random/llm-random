import json
import unittest
from lizrd.hostname_setup.utils import MachineBackend
from lizrd.preparation.prepare_params import PrepareParams


class TestConfigPreparation(unittest.TestCase):
    def test_prepare_configs(self):
        config_path = "lizrd/test/configs/inh_a_and_b_ok.yaml"
        expected_grid_path = "lizrd/test/grid.json"
        git_branch = "git_branch"
        CLUSTER_NAME = MachineBackend.IDEAS
        prepare_params = PrepareParams(config_path, git_branch)
        generated_grid, _, _, _ = prepare_params.prepare_configs(CLUSTER_NAME)

        with open(expected_grid_path, "r") as f:
            expected_grid = json.load(f)

        # expected grid from file has a list of lists, while the generated grid has a list of tuples
        # so we need to convert the expected grid to a list of tuples
        expected_grid = [tuple(x) for x in expected_grid]

        self.assertEqual(generated_grid, expected_grid)

    def test_prepare_config_no_interactive_debug(self):
        config_path = "lizrd/test/configs/inh_a_and_b_error.yaml"
        git_branch = "git_branch"
        CLUSTER_NAME = MachineBackend.IDEAS
        prepare_params = PrepareParams(config_path, git_branch)
        with self.assertRaises(AssertionError):
            prepare_params.prepare_configs(CLUSTER_NAME)
