from lizrd.hostname_setup.utils import (
    MachineBackend,
    get_grid_entrypoint,
    make_singularity_env_arguments,
    make_singularity_mount_paths,
)
from lizrd.scripts.grid_utils import translate_to_argparse


def get_subprocess_args(
    CLUSTER_NAME, slurm_command, setup_args, training_args, neptune_key, wandb_key
):
    job_name = training_args["name"]
    training_args["n_gpus"] = setup_args["n_gpus"]

    singularity_env_arguments = make_singularity_env_arguments(
        hf_datasets_cache_path=setup_args["hf_datasets_cache"],
        neptune_key=neptune_key,
        wandb_key=wandb_key,
    )

    singularity_mount_paths = make_singularity_mount_paths(setup_args, training_args)

    env = None
    runner_params = translate_to_argparse(training_args)
    if CLUSTER_NAME == MachineBackend.ENTROPY:
        subprocess_args = [
            slurm_command,
            "--partition=a100",
            f"--gres=gpu:a100:{setup_args['n_gpus']}",
            f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
            f"--mem={1000 // setup_args['n_gpus']}G",
            f"--job-name={job_name}",
            f"--time={setup_args['time']}",
            get_grid_entrypoint(CLUSTER_NAME),
            "singularity",
            "run",
            *singularity_env_arguments,
            singularity_mount_paths,
            "--nv",
            setup_args["singularity_image"],
            "python3",
            "-m",
            setup_args["runner"],
            *runner_params,
        ]
    elif CLUSTER_NAME == MachineBackend.ATHENA:
        subprocess_args = [
            slurm_command,
            f"--gres=gpu:{setup_args['n_gpus']}",
            "--partition=plgrid-gpu-a100",
            f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
            "--account=plgplggllmeffi-gpu-a100",
            f"--job-name={job_name}",
            f"--time={setup_args['time']}",
            get_grid_entrypoint(CLUSTER_NAME),
            "singularity",
            "run",
            "--bind=/net:/net",
            *singularity_env_arguments,
            singularity_mount_paths,
            "--nv",
            setup_args["singularity_image"],
            "python3",
            "-m",
            setup_args["runner"],
            *runner_params,
        ]
    elif CLUSTER_NAME == MachineBackend.IDEAS:
        subprocess_args = [
            slurm_command,
            f"--gres=gpu:ampere:{setup_args['n_gpus']}",
            f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
            f"--job-name={job_name}",
            f"--time={setup_args['time']}",
            "--mem=32G",
            setup_args["nodelist"],
            get_grid_entrypoint(CLUSTER_NAME),
            "singularity",
            "run",
            *singularity_env_arguments,
            singularity_mount_paths,
            "--nv",
            setup_args["singularity_image"],
            "python3",
            "-m",
            setup_args["runner"],
            *runner_params,
        ]
    return subprocess_args
