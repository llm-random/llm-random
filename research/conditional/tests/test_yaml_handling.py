from lizrd.support.misc import load_with_inheritance
from lizrd.support.test_utils import GeneralTestCase


class TestYaml(GeneralTestCase):
    def test_yaml(self):
        """
        Test if the yaml config with multiple configs and inheritance is loaded correctly
        """
        path_to_yaml = "research/conditional/train/configs/test/test.yaml"
        configs = load_with_inheritance(path_to_yaml)
        configs = list(configs)
        for config in configs:
            print(config)
        assert len(configs) == 2
        assert configs[0]["params"]["dhead"] == 21
        assert configs[1]["params"]["dhead"] == 210
        assert configs[0]["runs_multiplier"] == 20
        assert configs[1]["runs_multiplier"] == 1
