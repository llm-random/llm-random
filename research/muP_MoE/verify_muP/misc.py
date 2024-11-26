import neptune as neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_neptune_table(tags, negative_tags=None, columns=None):
    """
    Fetches a Neptune runs table filtered by tags and returns it as a pandas DataFrame.

    Parameters:
    - tags (list): List of tags to filter the runs.
    - negative_tags (list, optional): List of tags to exclude from the runs.
    - columns (list, optional): Additional columns to include in the runs table.

    Returns:
    - pandas.DataFrame: The runs table with the specified filters and columns.
    """

    # Initialize the Neptune project
    project = neptune.init_project(
        project="pmtest/llm-random",
        mode="read-only",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMDY0ZDI5Ni05YWU3LTQyNGYtYmY4My1hZTFkY2EzYmUwMjgifQ==",
    )

    # Fetch the runs table with the specified tags and columns
    runs_table = project.fetch_runs_table(tag=tags, columns=columns).to_pandas()

    # Ensure 'sys/tags' is a list for each run
    runs_table["sys/tags"] = runs_table["sys/tags"].apply(
        lambda x: x.split(",") if isinstance(x, str) else x
    )

    # Exclude runs containing any of the negative tags
    if negative_tags:
        for neg_tag in negative_tags:
            runs_table = runs_table[
                ~runs_table["sys/tags"].apply(lambda x: neg_tag in x)
            ]

    print(f"Table downloaded\nShape: {runs_table.shape}")
    return runs_table


def get_activations(runs_table, metric=None):
    activation_dict = {}
    for i, run_row in runs_table.iterrows():
        # dmodel = 2 ** (i + 4)
        dmodel = run_row["args/dmodel"]
        run_id = run_row["sys/id"]  # Assuming 'sys/id' is the run identifier
        print(f"run ID: {run_id}")
        project_name = "pmtest/llm-random"
        # run_id = "LLMRANDOM-2078"
        run = neptune.init_run(
            project=project_name,
            with_id=run_id,
            mode="read-only",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMDY0ZDI5Ni05YWU3LTQyNGYtYmY4My1hZTFkY2EzYmUwMjgifQ==",
        )
        print(f'run keys\n{run["block_0/residual_attention"]}')
        single_run_data = {"dmodel": dmodel}
        print(f'n_blocks: {run["args/n_blocks"].fetch()}')
        for i in range(run_row["args/n_blocks"]):
            single_run_data[i] = {
                "attn": run[
                    f"block_{i}/residual_attention/update_norms/{metric}"
                ].fetch_values(),
                "FF": run[
                    f"block_{i}/residual_feedforward/update_norms/{metric}"
                ].fetch_values(),
            }
        activation_dict[run_id] = single_run_data
    return activation_dict


def pivot_dict(
    activations_dict: dict, steps: list, dmodels: list, layer_num: int, module: str
):
    """
    Constructs a dictionary that maps each training step to a dictionary containing lists of 'dmodel' values
    and their corresponding activation 'value's for the specified layer and module.

    Parameters:
    - activations_dict (dict): The dictionary containing the activation data from get_activations().
    - steps (list): A list of training steps to consider.
    - dmodels (list): A list of 'dmodel' values to include in the output.
    - layer_num (int): The specific layer (block) number to extract data from.
    - module (str): 'FF' or 'attn', specifying which module to extract data from.

    Returns:
    - result_dict (dict): A dictionary with structure {<step>: {'dmodel': [...], 'value': [...]}}
    """
    result_dict = {}

    for step in steps:
        result_dict[step] = {"dmodel": [], "value": []}

        for run_id, run_data in activations_dict.items():
            dmodel = run_data.get("dmodel")
            if dmodel in dmodels:
                layer_data = run_data.get(layer_num)
                if layer_data:
                    module_data = layer_data.get(module)
                    if module_data is not None and not module_data.empty:
                        # Filter the DataFrame for the specific step
                        step_data = module_data[module_data["step"] == step]
                        if not step_data.empty:
                            value = step_data["value"].values[0]
                            result_dict[step]["dmodel"].append(dmodel)
                            result_dict[step]["value"].append(value)
                        else:
                            print(
                                f"No data at step {step} for run {run_id}, dmodel {dmodel}, layer {layer_num}, module {module}."
                            )
                    else:
                        print(
                            f"No module data for run {run_id}, dmodel {dmodel}, layer {layer_num}, module {module}."
                        )
                else:
                    print(
                        f"No layer data for run {run_id}, dmodel {dmodel}, layer {layer_num}."
                    )
            else:
                print(
                    f"dmodel {dmodel} from run {run_id} not in specified dmodels list."
                )

    return result_dict


def get_steps_from_first_run(activations_dict):
    """
    Returns all the 'step' values from the first run in activations_dict.

    Parameters:
    - activations_dict (dict): The dictionary containing the activation data from get_activations().

    Returns:
    - steps_list (list): A sorted list of unique 'step' values from the first run.
    """
    # Get the first run_id in the activations_dict
    first_run_id = next(iter(activations_dict))
    first_run_data = activations_dict[first_run_id]

    # Initialize an empty set to collect unique steps
    steps_set = set()

    # Exclude the 'dmodel' key and focus on block numbers
    block_keys = [key for key in first_run_data.keys() if isinstance(key, int)]

    # Iterate over each block in the first run
    for block_num in block_keys:
        block_data = first_run_data[block_num]
        # Iterate over the modules 'attn' and 'FF'
        for module in ["attn", "FF"]:
            module_data = block_data.get(module)
            if module_data is not None and not module_data.empty:
                # Add the 'step' values to the set
                steps = module_data["step"].tolist()
                steps_set.update(steps)
            else:
                print(
                    f"Module data missing or empty for block {block_num}, module {module}."
                )

    # Convert the set to a sorted list
    steps_list = sorted(steps_set)
    return steps_list


def plot_module(pivoted_dict, module_keyword, layer_num):
    """
    Plots the activation values of a specific module in a specific layer across different model widths (dmodels) over training steps.
    The x-axis is set to logarithmic scale, and x-ticks match the dmodel values.

    Parameters:
    - pivoted_dict (dict): The dictionary returned by pivot_dict(), structured as {<step>: {'dmodel': [...], 'value': [...]}}.
    - module_keyword (str): 'FF' or 'attn', specifying which module to plot.
    - layer_num (int): The specific layer (block) number to plot.

    Returns:
    - None
    """

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Collect all unique dmodel values across steps for setting x-ticks
    all_dmodels = set()
    for data in pivoted_dict.values():
        all_dmodels.update(data["dmodel"])
    all_dmodels = sorted(all_dmodels)

    # Iterate over each training step in the pivoted dictionary
    for step, data in pivoted_dict.items():
        dmodels = data["dmodel"]
        values = data["value"]
        # Sort the data based on dmodels for consistent plotting
        sorted_indices = np.argsort(dmodels)
        sorted_dmodels = np.array(dmodels)[sorted_indices]
        sorted_values = np.array(values)[sorted_indices]
        plt.plot(sorted_dmodels, sorted_values, marker="o", label=f"Step {int(step)}")

    plt.xlabel("Model Width (dmodel)")
    plt.ylabel(f"Mean Activation Value ({module_keyword})")
    plt.title(f"Activation of {module_keyword} in Layer {layer_num}")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Set x-axis to logarithmic scale
    plt.xscale("log")

    # Set x-ticks to match dmodel values
    plt.xticks(all_dmodels, labels=[str(dm) for dm in all_dmodels])

    plt.show()


def plot_loss_vs_lr(runs_table):
    """
    For each model width in the runs table, plots a line where the y-axis is the final loss value
    and the x-axis is the learning rate (lr).

    Parameters:
    - runs_table (pd.DataFrame): The DataFrame returned by get_neptune_table(), containing run information.

    Returns:
    - None
    """
    # Ensure required columns are present in runs_table
    required_columns = ["sys/id", "args/dmodel", "args/learning_rate"]
    for col in required_columns:
        if col not in runs_table.columns:
            raise ValueError(
                f"Column '{col}' is missing from runs_table. Please include it in the 'columns' parameter when calling get_neptune_table()."
            )

    # Get final loss values for each run
    final_loss_df = get_final_loss_values(runs_table)

    # Remove entries with missing final loss
    final_loss_df = final_loss_df[final_loss_df["final_loss"].notnull()]

    # Plotting
    plt.figure(figsize=(10, 6))
    model_widths = sorted(final_loss_df["dmodel"].unique())
    for model_width in model_widths:
        df_subset = final_loss_df[final_loss_df["dmodel"] == model_width]
        df_subset = df_subset.sort_values("lr")
        lrs = df_subset["lr"].to_numpy()
        losses = df_subset["final_loss"].to_numpy()
        plt.plot(lrs, losses, marker="o", label=f"Model width {model_width}")

    plt.xlabel("Learning Rate (lr)")
    plt.ylabel("Final Loss Value")
    plt.title("Final Loss vs Learning Rate for Different Model Widths")
    plt.legend()
    plt.grid(True)
    plt.xscale(
        "log"
    )  # Set x-axis to logarithmic scale if learning rates vary exponentially
    plt.show()


def get_final_loss_values(runs_table):
    """
    Fetches the final loss value for each run in the runs table.

    Parameters:
    - runs_table (pd.DataFrame): The DataFrame containing run information.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['run_id', 'dmodel', 'lr', 'final_loss']
    """
    final_losses = []
    for _, run_row in runs_table.iterrows():
        run_id = run_row["sys/id"]
        model_width = run_row["args/dmodel"]
        lr = run_row["args/learning_rate"]
        loss = run_row["loss_interval/100"]

        final_losses.append(
            {"run_id": run_id, "dmodel": model_width, "lr": lr, "final_loss": loss}
        )

    return pd.DataFrame(final_losses)
