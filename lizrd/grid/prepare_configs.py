import hashlib
from typing import List, Set, Tuple
import yaml


def get_yaml_md5(file_path):
    with open(file_path, "rb") as f:
        file_data = f.read()  # read file data into memory
        hash_md5 = hashlib.md5(file_data).hexdigest()
    return hash_md5


def recursive_update(base_dict, update_dict):
    for key, value in base_dict.items():
        if isinstance(value, dict):
            update_dict[key] = recursive_update(value, update_dict.get(key, {}))
        elif key not in update_dict:
            update_dict[key] = value
    return update_dict


def load_with_inheritance(
    filepath: str, all_config_paths: Set[str] = None, is_parent=False
) -> Tuple[List[dict], List[str]]:
    """
    Load configs from a yaml file, with inheritance.
    This means that every config can include a "parent" field, which points to another yaml file.
    Parent yamls are loaded first, and then the child config is recursively updated with the parent config.
    Parent yaml can only include one configuration.
    """
    if all_config_paths is None:
        all_config_paths = set()
    all_config_paths.add(filepath)

    with open(filepath, "r") as f:
        configs = list(yaml.safe_load_all(f))

    if is_parent and len(configs) > 1:
        raise Exception("Parent yaml can only include one configuration!")

    for config in configs:
        if "parent" in config:
            assert "md5_parent_hash" in config
            assert (
                get_yaml_md5(config["parent"]) == config["md5_parent_hash"]
            ), f"md5 hash of {config['parent']} is in fact {get_yaml_md5(config['parent'])}, but is listed as {config['md5_parent_hash']}"
            parent_config_list, additional_paths = load_with_inheritance(
                config["parent"], all_config_paths, is_parent=True
            )
            parent_config = parent_config_list[0]
            all_config_paths.update(additional_paths)
            config = recursive_update(parent_config, config)

    all_config_paths = sorted(list(all_config_paths))
    return configs, all_config_paths


def split_configs(configs: List[dict]) -> List[Tuple[dict, dict]]:
    split = []
    for config in configs:
        training_config = config["params"]
        del config["params"]
        infrastructure_config = config
        split.append((infrastructure_config, training_config))
    return split


def check_interactive_debug_not_in_further_configs(configs: List[dict]):
    options = [config.get("interactive_debug", False) for config in configs[1:]]
    assert not any(options), "interactive_debug can only be set in the first config"


def prepare_configs(
    config_path: str, git_branch: str, CLUSTER
) -> List[Tuple[dict, dict]]:
    configs, all_config_paths = load_with_inheritance(config_path)

    for config in configs:
        config["params"]["git_branch"] = git_branch
        config["params"]["path_to_entry_config"] = config_path
        config["params"]["all_config_paths"] = ",".join(all_config_paths)

    for config in configs:
        default_params = CLUSTER.prepare_default_infrastructure_params(
            config["params"]["dataset_type"]
        )
        config.update({k: v for k, v in default_params.items() if k not in config})

        # Here we should be confident that all the necessary keys are present in the config
        # Arguments below are used both in the runner and in the infrastructure
        config["params"]["n_gpus"] = config["n_gpus"]
        config["params"]["train_dataset_path"] = config["train_dataset_path"]
        config["params"]["validation_dataset_path"] = config["validation_dataset_path"]

    validate_configs(configs)

    configs = split_configs(configs)
    return configs


def validate_configs(configs: List[dict]):
    check_interactive_debug_not_in_further_configs(configs)

    for config in configs:
        assert config["runner"] in [
            "research.conditional.train.cc_train",
            "research.blanks.train",
            "research.token_reduction.runner",
            "research.grad_norm.runner",
        ], f"Unknown runner: {config['runner']} \nIf a new one was implemented, include it here as well"
