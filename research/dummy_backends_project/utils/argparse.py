import argparse


def introduce_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--name", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--git_branch", type=str)
    parser.add_argument("--path_to_entry_config", type=str)
    parser.add_argument("--all_config_paths", type=str)
    parser.add_argument("--n_gpus", type=int)
    parser.add_argument("--train_dataset_path", type=str)
    parser.add_argument("--validation_dataset_path", type=str)
    parser.add_argument("--tags", type=str, nargs="*")

    return parser
