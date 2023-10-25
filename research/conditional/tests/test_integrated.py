from lizrd.support.test_utils import GeneralTestCase, skip_test
from pathlib import Path
import subprocess


@skip_test("datasets_are_bugged")
class TestIntegrated(GeneralTestCase):
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
                ["python3", "-m", "lizrd.scripts.grid", f"--config_path={str(path)}"]
            )
            assert exit_code == 0
