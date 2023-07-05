import argparse
import os
import subprocess
import time


def process_exists(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_tmux_pid_by_name(session_name):
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}///#{pane_pid}"],
        stdout=subprocess.PIPE,
    )

    sessions = result.stdout.decode("utf-8").split("\n")

    for session in sessions:
        name, pid = session.split("///")
        if name == session_name:
            return int(pid)

    return None  # In case there are no sessions with the given name


def write_file(file_path, pid):
    with open(file_path, "w") as file:
        file.write(str(pid))


def machines_to_gpu_ids(available_machines):
    ids = ",".join([m[-5] for m in available_machines])
    return ids


def reserve_machines(process_id, gpu_log_files, no_machines, sleep_time=14.1314151617):
    while True:
        assert process_exists(process_id)
        available_machines = []
        for file in gpu_log_files:
            if file_is_free(file):
                available_machines.append(file)
        if len(available_machines) >= no_machines:
            available_gpu_files = available_machines[:no_machines]
            actual_available_machines = machines_to_gpu_ids(available_gpu_files)
            for file in available_gpu_files:
                write_file(file, process_id)
            print(actual_available_machines)
            break
        time.sleep(sleep_time)


def file_is_free(file_path):
    if os.stat(file_path).st_size == 0:  # returns True is the file is empty
        return True
    else:
        with open(file_path, "r") as file:
            pid = int(file.read())
        if not process_exists(pid):  # if process does not exist, returns True
            return True
        else:
            return False


def main(process_id, gpu_log_files, no_machines):
    reserve_machines(process_id, gpu_log_files, no_machines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_id", type=int, help="Process ID", required=True)
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
    main(args.process_id, log_files, args.no_machines)
