import copy
import os
import platform
from enum import Enum
from itertools import product
from typing import List, Tuple


class MachineBackend(Enum):
    ENTROPY = 1
    ATHENA = 2
    IDEAS = 3
    ENTROPY_GPU = 4
    LOCAL = 5


def get_machine_backend() -> MachineBackend:
    node = platform.uname().node
    if node == "asusgpu0":
        return MachineBackend.ENTROPY
    elif "athena" in node:
        return MachineBackend.ATHENA
    elif node == "login01":
        return MachineBackend.IDEAS
    elif node == "4124gs01":
        return MachineBackend.ENTROPY_GPU
    else:
        return MachineBackend.LOCAL


def get_common_directory(machine_backend: MachineBackend) -> str:
    if machine_backend == MachineBackend.ATHENA:
        return "/net/pr2/projects/plgrid/plggllmeffi"
    elif machine_backend == MachineBackend.IDEAS:
        return "/raid/NFS_SHARE/llm-random"
    elif machine_backend == MachineBackend.ENTROPY_GPU:
        return "/common/llm-random"
    else:
        return os.getenv("HOME")


def get_cache_path(machine_backend: MachineBackend) -> str:
    if machine_backend in [MachineBackend.LOCAL]:
        return f"{os.getenv('HOME')}/.cache/huggingface/datasets"
    else:
        common_dir = get_common_directory(machine_backend)
        return f"{common_dir}/.cache"


def get_sparsity_image(machine_backend: MachineBackend) -> str:
    image_name = "sparsity_2023.08.29_09.26.31.sif"
    common_dir = get_common_directory(machine_backend)
    return f"{common_dir}/images/{image_name}"


def get_grid_entrypoint(machine_backend: MachineBackend) -> str:
    if machine_backend in [MachineBackend.ENTROPY]:
        return "lizrd/scripts/grid_entrypoint.sh"
    elif machine_backend in [
        MachineBackend.ATHENA,
        MachineBackend.IDEAS,
        MachineBackend.ENTROPY_GPU,
    ]:
        return "lizrd/scripts/grid_entrypoint_athena.sh"
    elif machine_backend in [MachineBackend.LOCAL]:
        raise ValueError(f"Local machine should use main function directly. ")
    else:
        raise ValueError(f"Unknown machine backend: {machine_backend}")


def split_params(params: dict) -> Tuple[list, list, list]:
    functions = []
    grids = []
    normals = []
    for k, v in params.items():
        if k[0] == "^":
            grids.append((k[1:], v))
        elif k[0] == "*":
            functions.append((k[1:], v))
        elif "," in k:
            grids.append((k, v))
        else:
            normals.append((k, v))
    return grids, functions, normals


def shorten_arg(arg: str) -> str:
    ARG_TO_ABBR = {
        "reinit_dist": "rd",
        "ff_layer": "ff",
        "mask_loss_weight": "mlw",
        "class_loss_weight": "clw",
        "mask_percent": "mp",
        "n_steps": "ns",
        "n_steps_eval": "nse",
        "immunity": "im",
        "pruner": "pr",
        "pruner_prob": "prp",
        "pruner_delay": "prd",
        "pruner_n_steps": "prns",
    }
    return ARG_TO_ABBR.get(arg, arg)


def shorten_val(val: str) -> str:
    VAL_TO_ABBR = {
        # ff layers
        "regular": "r",
        "unstruct_prune": "up",
        "struct_prune": "sp",
        "unstruct_magnitude_prune": "ump",
        "struct_magnitude_prune": "smp",
        "unstruct_magnitude_recycle": "umr",
        "struct_magnitude_recycle_with_immunity": "smri",
        "masked_ff": "mf",
        "separate_direction_magnitude_ff": "sdmf",
        # reinit dist
        "zero": "0",
        "init": "i",
        "follow_normal": "fn",
    }
    if isinstance(val, bool):
        return "T" if val else "F"
    if isinstance(val, str):
        return VAL_TO_ABBR.get(val, val)
    if isinstance(val, int):
        if val % 1_000_000 == 0:
            return f"{val // 1_000_000}M"
        if val % 1_000 == 0:
            return f"{val // 1_000}k"
        return str(val)

    return str(val)


def make_tags(arg, val) -> str:
    if isinstance(val, list):
        val = "_".join([shorten_val(v) for v in val])
    return f"{shorten_arg(arg)}={shorten_val(val)}"


# parse time to minutes
def timestr_to_minutes(time: str) -> int:
    # Supported formats: "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
    def parse_time_part_no_days(time_part: str) -> Tuple[int, int, int]:
        if sum(c == ":" for c in time_part) == 0:
            return 0, int(time_part), 0
        elif sum(c == ":" for c in time_part) == 1:
            minutes, seconds = time_part.split(":")
            return 0, int(minutes), int(seconds)
        elif sum(c == ":" for c in time_part) == 2:
            hours, minutes, seconds = time_part.split(":")
            return int(hours), int(minutes), int(seconds)
        else:
            raise ValueError(f"Invalid time format: {time_part}")

    def parse_time_part_with_days(time_part: str) -> Tuple[int, int, int]:
        if sum(c == ":" for c in time_part) == 0:
            return int(time_part), 0, 0
        elif sum(c == ":" for c in time_part) == 1:
            hours, minutes = time_part.split(":")
            return int(hours), int(minutes), 0
        elif sum(c == ":" for c in time_part) == 2:
            hours, minutes, seconds = time_part.split(":")
            return int(hours), int(minutes), int(seconds)
        else:
            raise ValueError(f"Invalid time format: {time_part}")

    if "-" in time:
        days_part, time_part = time.split("-")
        days = int(days_part)
        hours, minutes, seconds = parse_time_part_with_days(time_part)
    else:
        days = 0
        hours, minutes, seconds = parse_time_part_no_days(time)

    return days * 24 * 60 + hours * 60 + minutes + round(seconds / 60)


def create_grid(params: dict) -> List[dict]:
    grids, functions, normals = split_params(params)
    base_params = {k: v for k, v in normals}
    out_params = []
    grids_keys = [k for k, v in grids]
    grids_values = product(*(v for k, v in grids))
    for value in grids_values:
        out_dict = copy.deepcopy(base_params)
        grid_dict = dict(zip(grids_keys, value))
        tags = [make_tags(k, v) for k, v in grid_dict.items()]
        if out_dict.get("tags") is None:
            out_dict["tags"] = []
        out_dict["tags"].extend(tags)
        out_dict = {**out_dict, **grid_dict}
        for func_name, func in functions:
            out_dict[func_name] = func(out_dict)
        out_params.append(out_dict)

    return out_params


def multiply_grid(param_sets: List[dict], runs_count: int) -> List[dict]:
    assert runs_count > 0

    if runs_count == 1:
        return param_sets

    out_params_sets = []
    for param_set in param_sets:
        for i in range(runs_count):
            out_dict = copy.deepcopy(param_set)
            out_dict["tags"].append(f"run={i+1}")
            out_dict["tags"].append(f"num_runs={runs_count}")
            out_params_sets.append(out_dict)
    return out_params_sets


def unpack_params(k, v):
    if "," in k:
        k = k.split(",")
        return k, v
    return [k], [v]


def param_to_str(param) -> str:
    if isinstance(param, str):
        return " ".join(param)
    else:
        return str(param)


def list_to_clean_str(l: List[str]) -> str:
    return " ".join([str(s) for s in l if s is not None])


def get_train_main_function(runner: str):
    from research.conditional.train.cc_train import main as cc_train_main

    if runner == "research.conditional.train.cc_train":
        return cc_train_main
    else:
        raise ValueError(f"Unknown runner: {runner}")


def get_setup_args_with_defaults(grid_args, CLUSTER_NAME):
    RUNS_MULTIPLIER = grid_args.get("runs_multiplier", 1)  ######
    TIME = grid_args.get("time", "1-00:00:00")  ######
    RUNNER = grid_args["runner"]
    GRES = grid_args.get("gres", "gpu:1")
    DRY_RUN = grid_args.get("dry_run", False)
    SINGULARITY_IMAGE = grid_args.get(
        "singularity_image", get_sparsity_image(CLUSTER_NAME)
    )
    HF_DATASETS_CACHE = grid_args.get("hf_datasets_cache", get_cache_path(CLUSTER_NAME))
    NODELIST = grid_args.get("nodelist", None)
    N_GPUS = grid_args.get("n_gpus", 1)
    CPUS_PER_GPU = grid_args.get("cpus_per_gpu", 8)
    CUDA_VISIBLE_DEVICES = grid_args.get("cuda_visible", None)

    if NODELIST is not None:
        NODELIST = "--nodelist=" + NODELIST

setup_args = {
    "gres": GRES,
    "time": TIME,
    "n_gpus": N_GPUS,
    "runner": RUNNER,
    "cpus_per_gpu": CPUS_PER_GPU,
    "nodelist": NODELIST,
    "cuda_visible": CUDA_VISIBLE_DEVICES,
    "hf_datasets_cache": HF_DATASETS_CACHE,
    "singularity_image": SINGULARITY_IMAGE,
    "runs_multiplier": RUNS_MULTIPLIER,
}
return setup_args


def translate_to_argparse(param_set: dict):
    runner_params = []

    for k_packed, v_packed in param_set.items():
        for k, v in zip(*unpack_params(k_packed, v_packed)):
            if isinstance(v, bool):
                if v:
                    runner_params.append(f"--{k}")
                else:
                    pass  # simply don't add it if v == False
                continue
            else:
                runner_params.append(f"--{k}")
                if isinstance(v, list):
                    runner_params.extend([str(s) for s in v])
                else:
                    runner_params.append(str(v))

    return runner_params
