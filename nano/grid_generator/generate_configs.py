from copy import deepcopy
import os
import yaml
from omegaconf import OmegaConf, DictConfig, ListConfig


def create_grid_config(cfg: DictConfig, path: str = ""):
    """
    Recursively expand an OmegaConf DictConfig (or a dict) that may contain
    '^'-prefixed keys (treated as a list of possible values for that key).

    Returns a list of tuples:
      [
        (expanded_dict: dict, overrides: List[str]),
        ...
      ]
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=False)

    if not isinstance(cfg, dict):
        # If it's a scalar or list at the top, return it as a single expansion
        return [(cfg, [])]

    # Separate normal keys from '^' keys
    normal_keys = {}
    caret_keys = {}
    for k, v in cfg.items():
        if k.startswith("^"):
            caret_keys[k.lstrip("^")] = v
        else:
            normal_keys[k] = v

    # If the same key is in both the normal and the caret set, ignore the normal key
    for k in list(normal_keys.keys()):
        if k in caret_keys:
            del normal_keys[k]

    # expansions will hold a list of (partial_dict, partial_overrides)
    expansions = [({}, [])]

    def merge_expansions(base_expansions, new_key, sub_expansions):
        """
        base_expansions: list of (dict, override_list)
        sub_expansions: list of (value, override_list_for_value)
        new_key: the key under which 'value' is placed in the dictionary
        """
        combined = []
        for base_dict, base_overrides in base_expansions:
            for val, sub_overrides in sub_expansions:
                new_dict = deepcopy(base_dict)
                new_dict[new_key] = val
                merged_overrides = base_overrides + sub_overrides
                combined.append((new_dict, merged_overrides))
        return combined

    def unique(mylist):
        return list(set(mylist))

    # First handle all normal (non-caret) keys
    for k, v in normal_keys.items():
        sub_expanded = expand_value(v, _join_path(path, k))
        expansions = merge_expansions(expansions, k, sub_expanded)

    # Then handle caret ('^') keys (these produce multiple expansions)
    for k, v in caret_keys.items():
        list_path = _join_path(path, k)
        # v must be a list, otherwise raise
        if not isinstance(v, list):
            raise ValueError(
                f"Key '{list_path}' is marked with '^' but its value is not a list."
            )
        # For each item in the caret list, expand it and gather its overrides
        new_expansions = []
        for item in v:
            # expand item (which might be a dict, scalar, or something else)
            item_expanded = expand_value(item, list_path)
            # For each expansion of that item, add overrides
            caretaker_item_sub = []
            for val, sub_overrides in item_expanded:
                # We already have sub_overrides from deeper expansions.
                # Additionally, gather overrides for all scalar leaves in 'val',
                # because we are inside a caret expansion.
                leaf_overrides = gather_scalar_leaves(val, list_path)
                merged_ovr = sub_overrides + leaf_overrides
                merged_ovr = unique(merged_ovr)
                caretaker_item_sub.append((val, merged_ovr))
            new_expansions.extend(caretaker_item_sub)

        # Now do the cartesian product with expansions so far
        expansions2 = []
        for base_dict, base_overrides in expansions:
            for val, val_overrides in new_expansions:
                new_dict = deepcopy(base_dict)
                new_dict[k] = val
                merged = base_overrides + val_overrides
                expansions2.append((new_dict, merged))
        expansions = expansions2

    return expansions


def expand_value(value, path):
    """
    Expand a single value (which might be a scalar, dict, or list) recursively.
    Returns a list of (expanded_value, overrides).
    """
    # If it's a dict, recursively expand
    if isinstance(value, dict):
        return create_grid_config(OmegaConf.create(value), path=path)

    # If it's a list, treat it as a single item (unless it's ^-prefixed outside)
    # Because a normal list that isn't ^ doesn't produce multiple expansions.
    return [(value, [])]


def gather_scalar_leaves(obj, prefix):
    """
    Recursively gather override lines for scalar leaves in obj:
      - If obj is a dict, gather from each key, building up prefix.
      - If obj is a list, gather from each item (with index) if you wish,
        or skip the index if you want simpler paths.
      - If obj is a scalar, return [f"{prefix}={obj}"].

    For your specific question, you likely only want to add overrides
    for dictionary keys (not list indexes). You can adapt as needed.
    """
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            sub_path = f"{prefix}.{k}"
            lines.extend(gather_scalar_leaves(v, sub_path))
        return lines
    elif isinstance(obj, list):
        # In many workflows, you might or might not want to record list indexes in overrides.
        # If you want to skip them, just gather from each item without the index:
        lines = []
        for item in obj:
            # We'll reuse the same prefix for all items in this list,
            # so we won't get perfect "path[0], path[1]" style overrides,
            # but we avoid confusion. Adapt if you prefer indexes in the path.
            lines.extend(gather_scalar_leaves(item, prefix))
        return lines
    else:
        # It's a scalar => produce a single override
        return [f"{prefix}={obj}"]


def _join_path(prefix, key):
    """Utility to safely build the path for overrides."""
    if not prefix:
        return key
    return f"{prefix}.{key}"
