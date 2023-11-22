import yaml
import argparse
import pandas as pd

from lizrd.support.misc import merge_dicts
from research.conditional.utils.model_utils import get_model_fit_gpu_info


def load_standard_model_config(model_name):
    assert model_name in ["mini", "small", "medium", "base", "large"]
    config_filepath = (
        f"research/conditional/train/configs/baselines/gpt/dense/{model_name}.yaml"
    )

    with open(config_filepath, "r") as f:
        baseline_config = yaml.safe_load(f)
        params = baseline_config["params"]

    return params


def main(
    models: [str],
    const_params: dict,
    param_to_compare: str,
    values_to_compare: [int],
    database_path: [str],
    output_report_path: str,
    batch_sizes: [int],
):
    """
    Database is a key-value database that stores values similar to this:
    {"batch_size": 2, "dff": 3072, "dmodel": 768, "group_size": 2, "n_att_heads": 12, "n_blocks": 12} -> "initialized/training/finished"
    """

    results = []
    for model_name in models:
        model_params = load_standard_model_config(model_name)
        model_params.update(const_params)

        model_results = [model_name]
        for value in values_to_compare:
            model_params[param_to_compare] = value
            max_batch_size = 1e9
            for batch_size in batch_sizes:
                model_params["batch_size"] = batch_size
                value = model_results.append(
                    get_model_fit_gpu_info(model_params, database_path)
                )
                if value < max_batch_size:
                    max_batch_size = value
            model_results.append(max_batch_size)

    df = pd.DataFrame(results, columns=["model_name \ group_size"] + values_to_compare)
    df.to_csv(output_report_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raport GPU model fits")
    parser.add_argument("--models", default="mini,small,medium,base", type=str)
    parser.add_argument(
        "--const_params",
        default="dff=3072,dmodel=768,n_att_heads=12,n_blocks=12",
        type=str,
    )
    parser.add_argument("--param_to_compare", default="group_size", type=str)
    parser.add_argument("--values_to_compare", default="32,64,128", type=str)
    parser.add_argument("--database_path", default="max_batch", type=str)
    parser.add_argument(
        "--output_report_path",
        default="max_batch_size_gpu_model_fits_report.csv",
        type=str,
    )
    parser.add_argument("--batch_sizes", default="32,64,128", type=str)
    args = parser.parse_args()
    values_to_compare = [int(x) for x in args.values_to_compare.split(",")]
    models = args.models.split(",")
    const_params = merge_dicts(
        *[dict([tuple(x.split("="))]) for x in args.const_params.split(",")]
    )
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    main(
        models,
        const_params,
        args.param_to_compare,
        values_to_compare,
        args.database_path,
        args.output_report_path,
        batch_sizes,
    )
