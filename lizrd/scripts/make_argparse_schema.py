"""
Creates a json schema from argparse parser. Useful to get editor support for configs.
Example usage (VSCode):
    1. Install https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml
    2. $ python make_argparse_schema.py research/blanks/argparse.py > research/blanks/schema.json
    3. Add the following to ./.vscode/settings.json:
        "yaml.schemas": {
            "research/blanks/schema.json": ["research/blanks/configs/**/*.yaml"]
        }
    4. Ctrl+Shift+P -> "Developer: Reload Window" (or restart VSCode)
    5. In case the argparse changes, repeat steps 2 and 4.
"""

import argparse
import importlib
import json
import sys


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "parser_path",
        type=str,
        help="Path to python file containing `introduce_parser_arguments` function",
    )
    return parser


def action_to_schema_entry(action) -> dict:
    res = {}

    type_ = None
    if action.nargs in ["*", "+"]:
        type_ = "array"
    elif action.type == float:
        type_ = "number"
    elif action.type == int:
        type_ = "integer"
    elif action.type == str:
        type_ = "string"
    elif isinstance(action.default, bool):
        type_ = "boolean"
    if type_ is not None:
        res["type"] = type_

    if action.default is not None:
        res["default"] = action.default

    if action.choices is not None:
        res["enum"] = action.choices

    return res


def create_schema(parser: argparse.ArgumentParser) -> dict:
    res = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Training config",
        "description": "This document describes the grid training config",
        "type": "object",
        "properties": {
            "parent": {"description": "Path to the parent yaml", "type": "string"},
            "gres": {
                "description": "Resources (slurm)",
                "type": "string",
            },
            "md5_parent_hash": {
                "description": "MD5 hash of the parent yaml",
                "type": "string",
            },
            "time": {
                "description": "How long the training will take (slurm)",
                "type": "string",
            },
            "runs_multiplier": {
                "description": "How many times to run the training",
                "type": "integer",
            },
            "runner": {
                "description": "Which runner to use",
                "type": "string",
            },
            "cpus_per_gpu": {
                "description": "How many cpus to use per gpu (slurm)",
                "type": "integer",
            },
            "nodelist": {
                "description": "Which nodes to use (slurm)",
                "type": "string",
            },
            "singularity_image": {
                "description": "Path to singularity image",
                "type": "string",
            },
            "cuda_visible": {
                "description": "Which gpus can be used (entropy_gpu)",
                "type": "string",
            },
            "hf_datasets_cache": {
                "description": "Path to datasets cache (huggingface)",
                "type": "string",
            },
            "interactive_debug": {
                "description": "Whether to run in interactive debug mode",
                "type": "boolean",
            },
            "n_gpus": {
                "description": "How many gpus to use",
                "type": "integer",
            },
            "params": {
                "description": "Parameters that will be passed to the script",
                "type": "object",
            },
        },
    }
    params_properties = {}
    required_params = []
    for action in parser._actions:
        name = action.dest
        required = action.required
        if name == "help":
            continue
        if required:
            required_params.append(name)
        params_properties[action.dest] = action_to_schema_entry(action)

    res["properties"]["params"]["properties"] = params_properties
    res["properties"]["params"]["required"] = required_params

    return res


def main():
    parser = make_parser()
    args = parser.parse_args()
    parser_path = args.parser_path
    parser_module = importlib.util.spec_from_file_location(
        "parser_module", parser_path
    ).loader.load_module()
    parser_function = parser_module.introduce_parser_arguments
    parser = parser_function(argparse.ArgumentParser())
    schema = create_schema(parser)
    json.dump(schema, sys.stdout, indent=4)


if __name__ == "__main__":
    main()
