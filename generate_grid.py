import itertools
import sys
from typing import List, Optional, Tuple

import yaml

from lizrd.grid.prepare_configs import get_yaml_md5

OUTPUT_FILE = "configs/experiments/grad_norm/std_norm_grid/small_c_lr_tune_grid_reduced_bn.yaml"
BASELINE_INPUT = "configs/experiments/grad_norm/medium_reduced_bs.yaml"


GRAD_MODIF_PLACEMENT_COMBINATIONS: List[Tuple[List[str], Optional[str]]] = [
    (["post_attn", "post_ff"], "post_attn_and_ff"),
    (["post_attn_norm", "post_ff_norm"], "post_norm"),
    (["post_attn_add", "post_ff_add"], "post_add"),
]

STD_NORM_MODIF_PARAMS_1: List[Tuple[List[str], Optional[str]]] = [
    (["layer_type=v1", "c=0.2", "eps=1e-6"], "layer_type_v1"),
    (["layer_type=v2", "c=0.2", "eps=1e-6"], "layer_type_v2"),
]

BASELINE_GRAD_MODIF_PLACEMENT = (
    ["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"],
    "all",
)
BASELINE_STD_NORM_MODIF_PARAMS = (["layer_type=v1", "c=0.0", "eps=0.0"], "baseline")

LR_MULTIPLIERS: List[Tuple[float, str]] = [
    (1 / 100, "div100"),
    (1 / 30, "div30"),
    (1 / 10, "div10"),
    (1 / 3, "div3"),
    (1, "mul1"),
    (3, "mul3"),
    (10, "mul10"),
]

NAME_PREFIX = "small_c_lr_tune"


def main():
    parent_md5_hash = get_yaml_md5(BASELINE_INPUT)
    configs = []

    grad_configs = itertools.chain(
        itertools.product(GRAD_MODIF_PLACEMENT_COMBINATIONS, STD_NORM_MODIF_PARAMS_1),
        ((BASELINE_GRAD_MODIF_PLACEMENT, BASELINE_STD_NORM_MODIF_PARAMS),),
    )

    for i, (((grad_placement, tag1), (layer_type, tag2)), (lr_multiplier, lr_tag)) in enumerate(
        itertools.product(grad_configs, LR_MULTIPLIERS)
    ):
        config_name = f"exp_{i}_{tag1}_{tag2}_lr_{lr_tag}"

        config = {
            "parent": BASELINE_INPUT,
            "time": "0-04:00:00",
            "md5_parent_hash": parent_md5_hash,
            "params": {
                "grad_modif_placement": grad_placement,
                "grad_modif_params": layer_type,
                "tags": [tag1, tag2, f"lr_{lr_tag}", "std_norm", "small_c_lr_tune", "grad_norm"],
                "name": f"{NAME_PREFIX}_{config_name}",
                "grad_modif_type": "std_norm",
                "learning_rate": 1e-4 * lr_multiplier,
                "n_steps": 2000,
            },
        }

        configs.append(config)

    with open(OUTPUT_FILE, "w") as f:
        print(f"Writing to {OUTPUT_FILE}")
        yaml.dump_all(configs, f)


if __name__ == "__main__":
    sys.exit(main())
