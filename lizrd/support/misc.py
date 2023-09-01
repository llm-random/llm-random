import datetime
import random
import string
from typing import Optional, List

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


def load_with_inheritance(filepath, is_parent=False):
    with open(filepath, "r") as f:
        configs = list(yaml.safe_load_all(f))

    if is_parent and len(configs) > 1:
        raise Exception("Parent yaml can only include one configuration!")

    for config in configs:
        if "parent" in config:
            parent_config = load_with_inheritance(config["parent"], is_parent=True)[0]
            config = recursive_update(parent_config, config)

    return configs


def recursive_update(base_dict, update_dict):
    for key, value in base_dict.items():
        if key not in update_dict:
            update_dict[key] = value
        elif isinstance(value, dict):
            update_dict[key] = recursive_update(value, update_dict.get(key, {}))
    return update_dict
