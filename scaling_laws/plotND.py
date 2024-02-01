import neptune
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


api_token = os.environ["NEPTUNE_API_TOKEN"]
project = neptune.init_project(api_token=api_token, project="SECRET_PROJECT_NAME")


def calculate_total_params(dmodel, expansion_rate, n_blocks, **_):
    # assume no params in routing and embeddings
    return dmodel**2 * 4 * (2 * expansion_rate + 1) * n_blocks


def calculate_active_params(dmodel, n_blocks, **_):
    return calculate_total_params(dmodel, 1, n_blocks)


# def predict_loss(n_steps, n_params, granularity):
#     n_steps = n_steps * 256 * 2048
#     return (1.8524*granularity**-0.6044+15.9259)*n_params**-0.1024 + 440.3686*n_steps**-0.2714 + 0.500


def predict_loss(n_steps, n_params, granularity):
    n_steps = n_steps * 256 * 2048
    return (
        (18.6961 * granularity**-0.1106 + 22.9111) * n_params**-0.1680
        + 20.0235 * n_steps**-0.1088
        + 0.500
    )  # (1.2167*granularity**-0.7143+14.3546)*n_params**-0.0890 + 50774.7539*n_steps**-0.5015 + 0.500 # (1.2167*granularity**-0.7143+14.3546)*n_params**-0.0890 + 50774.7539*n_steps**-0.5015 + 0.500 # (2.0760*granularity**-0.5843+18.0712)*n_params**-0.1146 + 30.8082*n_steps**-0.1465 + 0.472


TAG = "fixed_grid_wd"  # CHANGE TAG HERE
gran = 4

table = project.fetch_runs_table(tag=f"{TAG}").to_pandas()
# assert False
table = table[table["args/expansion_rate"] == 64]
table = table[table["sys/state"] == "Inactive"]
table = table[table["step"] > 7000]
table = table[table["args/n_steps"] == table["step"]]
table = table[table["args/granularity"] > 0.5]

# add the non-embedding params
# table["params"] = table.apply(lambda row: get_nonemb_params(row["args/dmodel"], row["args/n_att_heads"], row["args/n_blocks"]), axis=1)
table["params"] = table.apply(
    lambda row: calculate_total_params(
        row["args/dmodel"], row["args/expansion_rate"], row["args/n_blocks"]
    ),
    axis=1,
)
diff_params = sorted(table["params"].unique())
diff_steps = sorted(table["step"].unique())
print(f"Params: {diff_params}")
print(f"Params: {diff_steps}")
table["params"].max()
table["args/granularity"] = table["args/granularity"].astype(int)
table = table[table["params"] != 270532608]

# table = table[table["params"] == 16777216]
# table = table[table["params"] != 56623104]
# table = table[table["step"] < 200000]
# table = table[table["step"] < 100000]
# table = table[table["step"] > 10000]
table = table[table["step"] < 200000]
sizes = table["params"].unique().tolist()
print(table["params"].max())


steps = table["step"].unique().tolist()

for step in steps:
    if step == 20044:
        steps.pop(steps.index(step))

table = table[table["args/granularity"] != 32]

# table = table[table["step"] == 31250]
# table = table[table["params"] == table["params"].max()]


def plot_c(N, D, const):
    data = table
    data = data[data["step"] == D]
    data = data[data["params"] == N]

    X = sorted(data["args/granularity"])
    Y = sorted(data["loss_interval/1000"] - const, reverse=True)

    # Plotting the line
    fig = plt.scatter(X, Y, s=70)

    # X_sorted = X.values[sorted_indices]
    # Y_sorted = Y.values[sorted_indices]
    # plt.plot(X, Y, linewidth=2)

    # plt.plot(np.unique(X), np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)),'--', linewidth=1)
    x = np.array(X).reshape(-1, 1)
    y = np.array(Y).reshape(-1, 1)

    plt.plot(
        x,
        np.exp(LinearRegression().fit(np.log(x), np.log(y)).predict(np.log(x))),
        ":",
        linewidth=2,
    )

    # set log scale on x and y
    plt.xscale("log")
    plt.yscale("log")
    # set title with const
    plt.xlabel(f"Granularity")
    plt.ylabel(f"loss - {const}")
    plt.title(f"Loss vs Granularity for N={N}, D={D}")
    plt.tight_layout()
    plt.show()


plot_c(N=diff_params[5], D=diff_steps[2], const=3.02)
plot_c(N=diff_params[4], D=diff_steps[3], const=3.03)
plot_c(N=diff_params[5], D=diff_steps[3], const=2.88)
plot_c(N=diff_params[4], D=diff_steps[2], const=3.12)

pass
# plot_c(N=diff_params[5], D=diff_steps[2], const=3.002)
# 3 3.002)
