import copy
from itertools import product
from typing import List, Tuple
from enum import Enum
import platform


class MachineBackend(Enum):
    ENTROPY = 1
    ATHENA = 2
    LOCAL = 3


def get_machine_backend() -> MachineBackend:
    node = platform.uname().node
    if node == "asusgpu0":
        return MachineBackend.ENTROPY
    elif "athena" in node:
        return MachineBackend.ATHENA
    else:
        return MachineBackend.LOCAL


def get_grid_entrypoint(machine_backend: MachineBackend) -> str:
    if machine_backend in [MachineBackend.ENTROPY, MachineBackend.LOCAL]:
        return "lizrd/scripts/grid_entrypoint.sh"
    elif machine_backend == MachineBackend.ATHENA:
        return "lizrd/scripts/grid_entrypoint_athena.sh"
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


def param_to_str(param) -> str:
    if isinstance(param, str):
        return " ".join(param)
    else:
        return str(param)
