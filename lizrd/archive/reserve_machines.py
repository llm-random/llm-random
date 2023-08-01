import argparse
import os
import random
import subprocess
import time


def write_file(file_path, text):
    with open(file_path, "w") as file:
        file.write(str(text))


def machines_to_gpu_ids(available_machines):
    """
    This function reserves machines by writing the session name to the gpu log files.
    It is A BIT fragile: it assumes that the gpu log files are in the format "*0.txt", "*1.txt", etc.
    the fact that the [-5] index is the number of the gpu in nvidia-smi is what makes it fragile.
    """
    ids = ",".join([m[-5] for m in available_machines])
    return ids


def lock_files(control_file, pid):
    "writes pid to the control file"
    write_file(control_file, pid)


def release_files(control_file):
    "makes the control file empty"
    write_file(control_file, "")


def can_check_files(control_file):
    "checks if the control file is empty. If does not exist, creates it and returns True"
    try:
        with open(control_file, "r") as file:
            pid_str = file.read()
        if pid_str == "":
            return True
        else:
            return int(pid_str) in [pid for pid in os.listdir("/proc") if pid.isdigit()]
    except FileNotFoundError:
        with open(control_file, "w") as file:
            file.write("")
        os.chmod(control_file, 0o777)
        return True


def reserve_machines(
    session_name,
    socket_path,
    gpu_log_files,
    no_machines,
):
    while True:
        lock_register = os.path.join(os.path.dirname(gpu_log_files[0]), "register.txt")
        if not can_check_files(lock_register):
            time.sleep(random.random() * 10)
            continue
        process_id = str(os.getpid())
        lock_files(lock_register, process_id)
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
            release_files(lock_register)
            print(available_machines_cuda_format)
            break
        else:
            release_files(lock_register)
            time.sleep(random.random() * 100)


def call_nvidia_smi():
    result = subprocess.run(
        ["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")


def gpu_has_no_processes(gpu_number):
    sp = subprocess.Popen(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("\n")

    gpu_info = {}

    for item in out_list:
        if "python3" in item:
            item_list = item.split()
            gpu_info[item_list[1]] = item_list[4]

    return gpu_number not in gpu_info


def file_is_free(file_path, socket_path):
    # Read file to get session name
    with open(file_path, "r") as f:
        x = f.read().strip()

    gpu_num = file_path[-5]

    # Run the tmux ls command and get output
    result = subprocess.run(
        ["tmux", "-S", socket_path, "ls"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout.decode()

    for line in output.split("\n"):
        session_name = line.split(":")[0].strip()
        if session_name == x:
            return False
    if gpu_has_no_processes(gpu_num):
        return True
    else:
        return False


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
