"""
    This script generates a report of max batch size for a given models and a given parameter.
    We assume that we already run grid of short experiments to test if model fits in a GPU memory with records in some database.
    (i.e. used 'model_fit_gpu_info_database_path' and 'model_fit_gpu_info_params' in the training).

    Args:
        --models : list of models to compare (e.g."mini,small,medium")
        --param_to_compare : parameter on we compare our model (e.g. "group_size")
        --values_to_compare : values of above parameter to compare, 
            probably you want to take them from a grid(e.g. "32,64,128") 
        --database_path : path to DiskCache database with recorderd results from a grid
        --output_report_path : path to output report

    In data base we have result saved in a following format:
    Key is serialized dictionary with params: e.g. {"batch_size": 2, "dmodel": 768, "group_size": 2}.
    Value is one of:
    - "initalized" is when we have a model with given params, but we didn't start get to start a training 
    (probably model itself did not fit in a GPU memory).
    - "failure" is when we have a model with given params, but we didn't finish training (probably OOM error ).
    - "success" otherwise.

    Generated report is a csv analogous to:
    model_name \ group_size	32	64	128	256	512
    base	training_finished	-1	-1	-1	-1
"""
import yaml
import argparse
import pandas as pd

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
    results = []
    for model_name in models:
        model_params = load_standard_model_config(model_name)
        model_params.update(const_params)

        model_results = [model_name]
        for value in values_to_compare:
            model_params[param_to_compare] = value
            max_batch_size = -1
            for batch_size in batch_sizes:
                model_params["batch_size"] = batch_size
                value = get_model_fit_gpu_info(database_path, model_params)
                if value == "success" and batch_size > max_batch_size:
                    max_batch_size = batch_size
            model_results.append(max_batch_size)
        results.append(model_results)

    df = pd.DataFrame(results, columns=["model_name \ group_size"] + values_to_compare)
    df.to_csv(output_report_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raport GPU model fits")
    parser.add_argument("--models", default="mini,small,medium,base", type=str)
    parser.add_argument(
        "--const_params",
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
    const_params = (
        {*[dict([tuple(x.split("="))]) for x in args.const_params.split(",")]}
        if args.const_params is not None and args.const_params != ""
        else {}
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
