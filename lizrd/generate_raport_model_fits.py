import yaml
import argparse
import json
import pandas as pd


def main(gradient_checkpointing, group_sizes, models, input_file, output_file):
    params_per_model = {}

    for model_name in models:
        filepath = f"research/conditional/train/configs/baselines/gpt/{model_name}.yaml"

        with open(filepath, "r") as f:
            baseline_config = yaml.safe_load(f)
            params_per_model[model_name] = baseline_config["params"]

    with open(input_file) as file:
        results_data = json.load(file)

    prepared_data = []
    for model_name in models:
        # Get from results_data the model results
        model_params = params_per_model[model_name]
        results = results_data
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
        help="Is table for gradient checkpointing",
    )
    parser.add_argument("--input_file", default="results.json", type=str)
    parser.add_argument("--group_sizes", default="2,4,8,16,32,64,128,256", type=str)
    parser.add_argument("--output_file", default="output_file", type=str)
    parser.add_argument("--models", default="mini,small,medium,base", type=str)
    args = parser.parse_args()
    group_sizes = group_sizes = [int(x) for x in args.group_sizes.split(",")]
    models = args.models.split(",")
    main(
        args.gradient_checkpointing,
        group_sizes,
        models,
        args.input_file,
        args.output_file,
    )
