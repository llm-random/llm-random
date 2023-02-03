import re
import subprocess


def experiment_code_versioning(branch_name,remote_url="https://github.com/Simontwice/sparsity.git",):
    # Record the current branch
    current_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    current_branch_output = current_branch.stdout.strip()

    remote_url = remote_url.strip()
    # Check if remote_url is already among remote repos
    check_remote = subprocess.run(
        ["git", "remote", "-v"], capture_output=True, text=True
    )
    check_remote_output = check_remote.stdout
    remote_present = False
    lines = check_remote_output.split("\n")
    for line in lines:
        line = line.strip()
        if line == "":
            continue
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

    # Switch back to the original branch
    subprocess.run(["git", "checkout", current_branch_output], capture_output=True, text=True)

