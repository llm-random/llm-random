from lizrd.support.test_utils import GeneralTestCase
import subprocess


class TestIntegrated(GeneralTestCase):
    def test_dummy_train(self):
        """
        Test if the training loop runs without crashing, using dummy data
        from configs defined in research/conditional/train/configs/test/*.yaml
        """
        configs = ["research/conditional/train/configs/test/test_baseline.yaml"]
        for path in configs:
            print(f"Running training loop with config from {str(path)}")
            exit_code = subprocess.call(
                ["python3", "-m", "lizrd.scripts.grid", str(path)]
            )
            assert exit_code == 0
