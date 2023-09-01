import os

from lizrd.support.test_utils import GeneralTestCase
from pathlib import Path
import subprocess


class TestIntegrated(GeneralTestCase):
    def test_dummy_train(self):
        """
        Test if the training loop runs without crashing, using dummy data
        from configs defined in research/conditional/train/configs/test/*.yaml
        """

        configs = (Path(__file__).parent.resolve() / "../train/configs/test/").glob(
            "**/*.yaml"
        )
        # get the current environment variables
        env = os.environ.copy()
        env[
            "NEPTUNE_API_TOKEN"
        ] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzY5NDlmYy1lNzNhLTQ4NjEtODY2Ny1kODM4ZGYyMWFkMmYifQ=="
        for path in configs:
            print(f"Running training loop with config from {str(path)}")
            exit_code = subprocess.call(
                ["python3", "-m", "lizrd.scripts.grid", str(path)], env=env
            )
            assert exit_code == 0
