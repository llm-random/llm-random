from src.support.test_utils import GeneralTestCase, heavy_test
from pathlib import Path
import subprocess


class TestIntegrated(GeneralTestCase):
    @heavy_test
    def test_dummy_train(self):
        """
        Test if the training loop runs without crashing, using dummy data
        from configs defined in research/conditional/train/configs/test/*.yaml
        """

        configs = (Path(__file__).parent.resolve() / "../train/configs/test/").glob(
            "**/*.yaml"
        )
        for path in configs:
            print(f"Running training loop with config from {str(path)}")
            exit_code = subprocess.call(
                ["python3", "-m", "src.scripts.grid", f"--config_path={str(path)}"]
            )
            assert exit_code == 0
