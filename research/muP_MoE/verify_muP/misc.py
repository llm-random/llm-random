import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
                            result_dict[step]["value"].append(value / np.sqrt(dmodel))
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


def plot_module(pivoted_dict, module_keyword, layer_num, step_interval=100):
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
        if step % step_interval == 0:
            dmodels = data["dmodel"]
            values = data["value"]
            # Sort the data based on dmodels for consistent plotting
            sorted_indices = np.argsort(dmodels)
            sorted_dmodels = np.array(dmodels)[sorted_indices]
            sorted_values = np.array(values)[sorted_indices]
            plt.plot(
                sorted_dmodels, sorted_values, marker="o", label=f"Step {int(step)}"
            )

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


def plot_loss_vs_lr(runs_table, ylim=None):
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
    if ylim is not None:
        plt.ylim(ylim)
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
