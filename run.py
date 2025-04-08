import os
from typing import Optional
import nemo_run as run


def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes, and devices args for using this function."
        )

    mounts = []
    # Custom mounts are defined here.
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This will package the train.py script in the current working directory to the remote cluster.
    # If you are inside a git repo, you can also use https://github.com/NVIDIA/NeMo-Run/blob/main/src/nemo_run/core/packaging/git.py.
    # If the script already exists on your container and you call it with the absolute path, you can also just use `run.Packager()`.
    packager = run.PatternPackager(include_pattern="train.py", relative_path=os.getcwd())

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir, # This is where the results of the run will be stored by default.
            # identity="/path/to/identity/file" OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
        packager=packager,
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


if __name__ == "__main__":
    training_job = run.Script(
        inline="""
    # This string will get saved to a sh file and executed with bash
    # Run any preprocessing commands

    # Run the training command
    python _train.py

    # Run any post processing commands
    """
    )

    # Run it locally
    # executor = run.LocalExecutor()
    executor = slurm_executor(...) # pass in args relevant to your cluster

    with run.Experiment("nemo_2.0_training_experiment", log_level="INFO") as exp:
        exp.add(training_job, executor=executor, tail_logs=True, name="training")
        # Add more jobs as needed

        # Run the experiment
        exp.run(detach=False)


