import datetime
import hashlib
import random
import string
from typing import Optional, List, Tuple, Set

# import numpy as np
# import torch

import yaml


def tags_to_name(tags: Optional[List[str]]) -> str:
    return "_".join(tags) if tags else ""


def make_concise_datetime() -> str:
    now = datetime.datetime.now()
    return str(now.year)[-2:] + "_" + now.strftime("%m-%d_%H:%M:%S")


def count_parameters(model, args, VOCAB_SIZE):
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_embedding_and_head_params = 2 * VOCAB_SIZE * args.dmodel
    pos_embedding_params = args.cutoff * args.dmodel
    model_n_params -= input_embedding_and_head_params + pos_embedding_params
    return model_n_params


def generate_random_string(length: int) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def load_with_inheritance(
    filepath: str, all_config_paths: Set[str] = None, is_parent=False
) -> Tuple[List[dict], Set[str]]:
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

    return configs, all_config_paths


def recursive_update(base_dict, update_dict):
    for key, value in base_dict.items():
        if isinstance(value, dict):
            update_dict[key] = recursive_update(value, update_dict.get(key, {}))
        elif key not in update_dict:
            update_dict[key] = value
    return update_dict


def get_yaml_md5(file_path):
    with open(file_path, "rb") as f:
        file_data = f.read()  # read file data into memory
        hash_md5 = hashlib.md5(file_data).hexdigest()
    return hash_md5
