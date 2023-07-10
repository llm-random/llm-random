import argparse
import random
import subprocess
import time
import re


def write_file(file_path, session_name):
    with open(file_path, "w") as file:
        file.write(str(session_name))


def machines_to_gpu_ids(available_machines):
    """
    This function reserves machines by writing the session name to the gpu log files.
    It is A BIT fragile: it assumes that the gpu log files are in the format "*0.txt", "*1.txt", etc.
    the fact that the [-5] index is the number of the gpu in nvidia-smi is what makes it fragile.
    """
    ids = ",".join([m[-5] for m in available_machines])
    return ids


def lock_files(control_file_path="/tmp/gpu_control_file.txt"):
    "writes 'locked' to the control file"
    write_file(control_file_path, "locked")


def release_files(control_file_path="/tmp/gpu_control_file.txt"):
    "makes the control file empty"
    write_file(control_file_path, "")


def can_check_files(control_file_path="/tmp/gpu_control_file.txt"):
    "checks if the control file is empty. If does not exist, creates it and returns True"
    try:
        with open(control_file_path, "r") as file:
            return file.read() == ""
    except FileNotFoundError:
        write_file(control_file_path, "")
        return True


def reserve_machines(
    session_name,
    socket_path,
    gpu_log_files,
    no_machines,
):
    while True:
        if not can_check_files():
            time.sleep(random.random() * 10)
            continue
        lock_files()
        available_machine_files = []
        for file in gpu_log_files:
            if file_is_free(file, socket_path):
                available_machine_files.append(file)
        if len(available_machine_files) >= no_machines:
            available_machine_files = available_machine_files[:no_machines]
            available_machines_cuda_format = machines_to_gpu_ids(
                available_machine_files
            )
            for file in available_machine_files:
                write_file(file, session_name)
            # this is the output in bash
            release_files()
            print(available_machines_cuda_format)
            break
        else:
            release_files()
            time.sleep(random.random() * 10)


def call_nvidia_smi():
    result = subprocess.run(["nvidia-smi", "-q"], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8")


def gpu_has_no_processes(gpu_number):
    """parse nvidia-smi output to see if gpu has no processes running"""

    nvidia_output = call_nvidia_smi()
    chunks = re.split("Product Name", nvidia_output)
    number_pattern = re.compile(r"\s*Minor Number\s*:\s*(\d+)")
    gpu_exists = False

    for chunk in chunks:
        minor_number_match = re.search(number_pattern, chunk)
        if minor_number_match:
            minor_number = minor_number_match.group(1).split(":")[-1].strip()
            if minor_number == gpu_number:
                gpu_exists = True
                if "Process ID" in chunk:
                    return False

    if not gpu_exists:
        raise ValueError(f"GPU with minor number {gpu_number} not found in string")

    return True


def file_is_free(file_path, socket_path):
    # Read file to get session name
    with open(file_path, "r") as f:
        x = f.read().strip()

    gpu_num = file_path[-5]

    # Run the tmux ls command and get output
    result = subprocess.run(["tmux", "-S", socket_path, "ls"], stdout=subprocess.PIPE)
    output = result.stdout.decode()

    for line in output.split("\n"):
        session_name = line.split(":")[0].strip()
        if session_name == x:
            return False
    if not gpu_has_no_processes(gpu_num):
        return False

    return True


def main(session_name, socket_path, gpu_log_files, no_machines):
    reserve_machines(session_name, socket_path, gpu_log_files, no_machines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session_name", type=str, help="tmux session name", required=True
    )
    parser.add_argument(
        "--socket_path", type=str, help="tmux socket path", required=True
    )
    parser.add_argument(
        "--gpu_log_files",
        type=str,
        help="GPU log files in one big string",
        required=True,
    )
    parser.add_argument(
        "--no_machines", type=int, help="Number of machines to reserve", required=True
    )
    args = parser.parse_args()
    log_files = args.gpu_log_files.split(" ")
    main(args.session_name, args.socket_path, log_files, args.no_machines)
