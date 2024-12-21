import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pivot_dict(
    activations_dict: dict, steps: list, dmodels: list, layer_num: int, module: str
):
    """
    Constructs a dictionary that maps each training step to a dictionary containing dicts: dmodel: values_list

    Parameters:
    - activations_dict (dict): The dictionary containing the activation data from get_activations().
    - steps (list): A list of training steps to consider.
    - dmodels (list): A list of 'dmodel' values to include in the output.
    - layer_num (int): The specific layer (block) number to extract data from.
    - module (str): 'FF' or 'attn', specifying which module to extract data from.

    Returns:
    - result_dict (dict): A dictionary with structure {<step>: {dmodel: [val1, val2, ...]}}
    """
    result_dict = {}

    for step in steps:
        result_dict[step] = {}

        for run_id, run_data in activations_dict.items():
            dmodel = run_data.get("dmodel")
            if dmodel in dmodels:
                layer_data = run_data.get(layer_num)
                if dmodel not in result_dict[step].keys():
                    result_dict[step][dmodel] = []
                if layer_data:
                    module_data = layer_data.get(module)
                    if module_data is not None and not module_data.empty:
                        # Filter the DataFrame for the specific step
                        step_data = module_data[module_data["step"] == step]
                        if not step_data.empty:
                            value = step_data["value"].values[0]
                            result_dict[step][dmodel].append(value)
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
    - pivoted_dict (dict): The dictionary returned by pivot_dict(), structured as {<step>: {dmodel: [val1, val2, ...]}}.
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
        all_dmodels.update(data.keys())
    all_dmodels = sorted(all_dmodels)

    # Iterate over each training step in the pivoted dictionary
    for step, data in pivoted_dict.items():
        if step % step_interval == 0:
            dmodels = np.array(list(data.keys()))
            # Sort the data based on dmodels for consistent plotting
            sorted_dmodels = np.sort(dmodels)
            sorted_values = []
            for dmodel in sorted_dmodels:
                val = np.array(data[dmodel]).mean()
                sorted_values.append(val)
            sorted_values = np.array(sorted_values)
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
    plt.yscale("log")

    # Set x-ticks to match dmodel values
    plt.xticks(all_dmodels, labels=[str(dm) for dm in all_dmodels])

    plt.show()


def plot_module_grid(
    pivoted_dict, module_keyword, layer_num, step_interval=100, fig=None, ax=None
):
    """
    Plots the activation values of a specific module in a specific layer across different model widths (dmodels)
    over training steps. The x-axis is set to a logarithmic scale, and x-ticks match the dmodel values.

    Parameters:
    - pivoted_dict (dict): The dictionary returned by pivot_dict(), structured as {<step>: {dmodel: [val1, val2, ...]}}.
    - module_keyword (str): 'FF' or 'attn', specifying which module to plot.
    - layer_num (int): The specific layer (block) number to plot.
    - step_interval (int): Interval at which to sample steps for plotting.
    - fig (matplotlib.figure.Figure, optional): Figure object to plot on. If None, a new figure is created.
    - ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new axes is created.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object for the plot.
    """

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all unique dmodel values across steps for setting x-ticks
    all_dmodels = set()
    for data in pivoted_dict.values():
        all_dmodels.update(data.keys())
    all_dmodels = sorted(all_dmodels)

    # Iterate over each training step in the pivoted dictionary
    for step, data in pivoted_dict.items():
        if step % step_interval == 0:
            dmodels = np.array(list(data.keys()))
            # Sort the data based on dmodels for consistent plotting
            sorted_dmodels = np.sort(dmodels)
            sorted_values = []
            for dmodel in sorted_dmodels:
                val = np.array(data[dmodel]).mean()
                sorted_values.append(val)
            sorted_values = np.array(sorted_values)
            ax.plot(
                sorted_dmodels, sorted_values, marker="o", label=f"Step {int(step)}"
            )

    ax.set_xlabel("Model Width (dmodel)")
    ax.set_ylabel(f"Mean Activation Value ({module_keyword})")
    ax.set_title(f"Activation of {module_keyword} in Layer {layer_num}")
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    # Set x-axis to logarithmic scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set x-ticks to match dmodel values
    ax.set_xticks(all_dmodels)
    ax.set_xticklabels([str(dm) for dm in all_dmodels])

    fig.tight_layout()
    return fig, ax


def plot_multiple_modules(
    activations_dict,
    module_keywords,
    layer_nums,
    dmodels,
    step_interval=100,
    figsize=(15, 10),
):
    """
    Creates a grid of subplots, each plotting the activation values for a specified module and layer combination.
    Uses the plot_module function for each combination of module_keyword and layer_num.

    Parameters:
    - module_keywords (list of str): List of module keywords (e.g., ['FF', 'attn']).
    - layer_nums (list of int): List of layer numbers to plot.
    - step_interval (int): Interval at which to sample steps for plotting.
    - figsize (tuple): Size of the figure.

    Returns:
    - None (just shows the plot)
    """

    # Determine the number of rows and columns in the grid
    n_rows = len(module_keywords)
    n_cols = len(layer_nums)
    steps = get_steps_from_first_run(activations_dict)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, mk in enumerate(module_keywords):
        for j, ln in enumerate(layer_nums):
            pivoted_dict = pivot_dict(
                activations_dict=activations_dict,
                steps=steps,
                dmodels=dmodels,
                layer_num=ln,
                module=mk,
            )
            # Use the plot_module function to plot on the given Axes object
            plot_module_grid(
                pivoted_dict=pivoted_dict,
                module_keyword=mk,
                layer_num=ln,
                step_interval=step_interval,
                fig=fig,
                ax=axs[i, j],
            )

    plt.show()


def plot_loss_vs_lr(runs_table, ylim=None, title=None, figsize=(10, 6)):
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
    plt.figure(figsize=figsize)
    model_widths = sorted(final_loss_df["dmodel"].unique())
    for model_width in model_widths:
        df_subset = final_loss_df[final_loss_df["dmodel"] == model_width]
        df_subset = df_subset.sort_values("lr")
        lrs = df_subset["lr"].to_numpy()
        losses = df_subset["final_loss"].to_numpy()
        plt.plot(lrs, losses, marker="o", label=f"Model width {model_width}")

    plt.xlabel("Learning Rate (lr)")
    plt.ylabel("Final Loss Value")
    if title is None:
        title = "Final Loss vs Learning Rate for Different Model Widths"
    plt.title(title)
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
