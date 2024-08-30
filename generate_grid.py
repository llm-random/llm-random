import itertools
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Set, Tuple

import yaml

GRID_OUTPUT = "configs/experiments/grad_norm/grid"
BASELINE_INPUT = "configs/experiments/grad_norm/medium.yaml"


GRAD_MODIF_PLACEMENT_COMBINATIONS: List[Tuple[Set[str], str]] = [
    ({"post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"}, "all"),
    ({"post_attn", "post_ff"}, "post_attn_and_ff"),
    ({"post_attn_norm", "post_ff_norm"}, "post_norm"),
    ({"post_attn_add", "post_ff_add"}, "post_add"),
]

STD_NORM_LAYER_TYPES: List[Tuple[str, str]] = [
    ("layer_type=v1", "v1"),
    ("layer_type=v2", "v2"),
    ("layer_type=v3", "v3"),
]

NAME_PREFIX = "grad_norm_formulas"


def main():
    with open(BASELINE_INPUT, "r") as f:
        baseline = yaml.safe_load(f)
        parent_md5_hash = "04709a070794b1108adf921b25da8a8c"

    for i, ((grad_placement, tag1), (layer_type, tag2)) in enumerate(
        itertools.product(GRAD_MODIF_PLACEMENT_COMBINATIONS, STD_NORM_LAYER_TYPES)
    ):
        config_fname = f"exp_{i}_{tag1}_{tag2}.yaml"

        config = deepcopy(baseline)
        config["md5_parent_hash"] = parent_md5_hash
        config["params"]["grad_modif_placement"] = list(grad_placement)
        config["params"]["grad_modif_params"].append(layer_type)
        config["params"]["tags"].append(tag1)
        config["params"]["tags"].append(tag2)
        config["params"]["name"] = f"{NAME_PREFIX}_{config_fname}"

        output_file = Path(GRID_OUTPUT) / config_fname
        with open(output_file, "w") as f:
            print(f"Writing to {output_file}")
            yaml.dump(config, f)


if __name__ == "__main__":
    sys.exit(main())
