import yaml

from lizrd.support.misc import load_with_inheritance
from lizrd.support.test_utils import GeneralTestCase


class TestYaml(GeneralTestCase):
    def test_multiple(self):
        """
        Test if the yaml config with multiple configs is loaded correctly
        """
        path_to_yaml = "lizrd/scripts/tests/configs/multiple_yamls.yaml"
        configs = yaml.safe_load_all(open(path_to_yaml))
        configs = list(configs)
        for config in configs:
            print(config)
        assert len(configs) == 2
        assert configs[0]["params"]["dhead"] == 21
        assert configs[0]["runs_multiplier"] == 20
        assert configs[1]["runs_multiplier"] == 1

    def test_inherit(self):
        """
        Test if the yaml config with  inheritance is loaded correctly
        """
        path_to_yaml = "lizrd/scripts/tests/configs/inheritance_yaml.yaml"
        configs, paths_to_all_configs = load_with_inheritance(path_to_yaml)
        assert set(paths_to_all_configs) == {
            "lizrd/scripts/tests/configs/inheritance_yaml.yaml",
            "lizrd/scripts/tests/configs/inheritance_yaml2.yaml",
        }
        configs = list(configs)
        for config in configs:
            print(config)
        assert len(configs) == 2
        assert configs[0]["params"]["dhead"] == 21
        assert configs[0]["runs_multiplier"] == 20
        assert configs[1]["runs_multiplier"] == 233
        assert configs[1]["params"]["dhead"] == 1
