import re
import subprocess


def experiment_code_versioning(remote_url, branch_name):
    remote_url = remote_url.strip()
    # Check if remote_url is already among remote repos
    check_remote = subprocess.run(
        ["git", "remote", "-v"], capture_output=True, text=True
    )
    check_remote_output = check_remote.stdout
    remote_present = False
    lines = check_remote_output.split("\n")
    for line in lines:
        nickname, url, _ = re.split("[\t ]", line)
        if url == remote_url:
            remote_name = nickname
            remote_present = True
            break
    if remote_present:
        # Create a new branch
        subprocess.run(
            ["git", "checkout", "-b", branch_name], capture_output=True, text=True
        )
        # Push the current code to the repo
        subprocess.run(
            ["git", "push", remote_name, branch_name], capture_output=True, text=True
        )
    else:
        # Add the repo as a remote
        subprocess.run(
            ["git", "remote", "add", "code_image_cemetery", remote_url],
            capture_output=True,
            text=True,
        )
        # Create a new branch
        subprocess.run(
            ["git", "checkout", "-b", branch_name], capture_output=True, text=True
        )
        # Push the current code to the repo
        subprocess.run(
            ["git", "push", "origin", branch_name], capture_output=True, text=True
        )


experiment_code_versioning(
    "  https://github.com/Simontwice/sparsity.git  ", "code_image_cemetery"
)
