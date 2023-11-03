import yaml
import argparse
import json
import pandas as pd

from lizrd.support.misc import merge_dicts


def main(
    gradient_checkpointing: bool,
    group_sizes: [int],
    models: [str],
    report_files: [str],
    output_file,
):
    params_per_model = {}

    for model_name in models:
        filepath = f"research/conditional/train/configs/baselines/gpt/{model_name}.yaml"

        with open(filepath, "r") as f:
            baseline_config = yaml.safe_load(f)
            params_per_model[model_name] = baseline_config["params"]

    result_dict = {}
    for report_file in report_files:
        with open(report_file) as file:
            report_dict = json.load(file)
            result_dict = merge_dicts(result_dict, report_dict)

    prepared_data = []
    for model_name in models:
        # Get from result_dict a current model results
        model_params = params_per_model[model_name]
        results = result_dict
        while "group_size" not in results.keys():
            param = list(results.keys())[0]
            results = results[param]
            results = results[str(model_params[param])]

        # Get the biggest batch_size that fit for each group_size
        model_results = results["group_size"]
        group_sizes_best_batch_sizes = []
        for group_size in group_sizes:
            group_results = model_results[str(group_size)]
            gradient_checkpointing_results = group_results["gradient_checkpointing"]
            gradient_checkpointing_results = gradient_checkpointing_results[
                str(gradient_checkpointing)
            ]

            max_true_batch_size = max(
                (
                    int(k)
                    for k, v in gradient_checkpointing_results["batch_size"].items()
                    if v
                ),
                default=-1,
            )
            group_sizes_best_batch_sizes.append(max_true_batch_size)

        prepared_data.append([model_name] + group_sizes_best_batch_sizes)

    df = pd.DataFrame(prepared_data, columns=["model_name \ group_size"] + group_sizes)
    df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raport model fits")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Is table for gradient_checkpointing=True",
    )
    parser.add_argument("--input_files", default="gpu_usage_report.json", type=str)
    parser.add_argument("--group_sizes", default="2,4,8,16,32,64,128,256", type=str)
    parser.add_argument(
        "--output_file", default="gpu_max_batch_size_report.csv", type=str
    )
    parser.add_argument("--models", default="mini,small,medium,base", type=str)
    args = parser.parse_args()
    group_sizes = group_sizes = [int(x) for x in args.group_sizes.split(",")]
    models = args.models.split(",")
    report_files = args.input_files.split(",")
    main(
        args.gradient_checkpointing,
        group_sizes,
        models,
        report_files,
        args.output_file,
    )
