import argparse
import subprocess
import time


def write_file(file_path, session_name):
    with open(file_path, "w") as file:
        file.write(str(session_name))


def machines_to_gpu_ids(available_machines):
    ids = ",".join([m[-5] for m in available_machines])
    return ids


def reserve_machines(
    session_name, socket_path, gpu_log_files, no_machines, sleep_time=14.1314151617
):
    while True:
        available_machines = []
        for file in gpu_log_files:
            if file_is_free(file, socket_path):
                available_machines.append(file)
        if len(available_machines) >= no_machines:
            available_gpu_files = available_machines[:no_machines]
            actual_available_machines = machines_to_gpu_ids(available_gpu_files)
            for file in available_gpu_files:
                write_file(file, session_name)
            print(actual_available_machines)
            break
        time.sleep(sleep_time)


def file_is_free(file_path, socket_path):
    # Read file to get session name
    with open(file_path, "r") as f:
        x = f.read().strip()

    # Run the tmux ls command and get output
    result = subprocess.run(["tmux", "-S", socket_path, "ls"], stdout=subprocess.PIPE)
    output = result.stdout.decode()

    for line in output.split("\n"):
        session_name = line.split(":")[0].strip()
        if session_name == x:
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
