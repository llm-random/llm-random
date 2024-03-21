from contextlib import contextmanager
import copy
import getpass
import os
import datetime
from typing import Generator
from fabric import Connection
from argparse import ArgumentParser
from git import Repo
import paramiko.ssh_exception

from lizrd.support.misc import generate_random_string

_SSH_HOSTS_TO_PASSPHRASES = {}


@contextmanager
def ConnectWithPassphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host using a passphrase if the key is encrypted. The passphrase is preserved for subsequent connections to the same host."""
    try:
        connection = Connection(*args, **kwargs)
        connection.run('echo "Connection successful."')
        yield connection
    except paramiko.ssh_exception.PasswordRequiredException as e:
        if connection.host not in _SSH_HOSTS_TO_PASSPHRASES:
            passphrase = getpass.getpass(
                f"SSH key encrypted, provide the passphrase ({connection.host}): "
            )
            _SSH_HOSTS_TO_PASSPHRASES[connection.host] = passphrase
        else:
            passphrase = _SSH_HOSTS_TO_PASSPHRASES[connection.host]
        kwargs["connect_kwargs"] = copy.deepcopy(
            kwargs.get("connect_kwargs", {})
        )  # avoid modifying the original connect_kwargs
        kwargs["connect_kwargs"]["passphrase"] = passphrase
        connection = Connection(*args, **kwargs)
        yield connection
    finally:
        connection.close()


def cd_to_root_dir():
    repo = Repo()
    assert repo.remotes.origin.url in [
        "git@github.com:llm-random/llm-random.git",
    ], "You're not in the right repo! Move to the llm-random folder, and make sure your origin is the llm-random repo. Aborting..."
    os.chdir(repo.working_dir)


def rsync_to_remote(host, local_dir):
    try:
        with ConnectWithPassphrase(host) as connection:
            base_dir = get_base_directory(connection)
            proxy_command = get_proxy_command(connection)
            rsync_command = [
                "rsync",
                "--compress",
                "--recursive",
                "--links",
                "--perms",
                "--human-readable",
                "--stats",
                f"--rsh={proxy_command}",
                "--exclude=*.pyc",
                local_dir,
                f"{host}:{base_dir}",
            ]
            print(f"Syncing {local_dir} to {host}:{base_dir}...")
            connection.local(" ".join(rsync_command), echo=True, warn=True)
            print("Sync complete.")
            return base_dir
    except Exception as e:
        raise Exception(f"[RSYNC ERROR]: An error occurred during rsync: {str(e)}")


def athena_user_to_workdir(connection):
    d = {
        "plglizard": "jaszczur",
        "plgcrewtool": "crewtool",
        "plgmaciejpioro": "maciejpioro",
        "plgludziej": "ludziej",
        "plgsimontwice": "simontwice",
    }
    user = connection.user
    return d[user] if user in d else user[3:]


def get_base_directory(connection):
    if connection.host == "athena.cyfronet.pl":
        base_dir = f"/net/pr2/projects/plgrid/plggsubgoal/{athena_user_to_workdir(connection)}/llm-random"
    else:
        base_dir = f"~/llm-random"
    return base_dir


def get_proxy_command(connection):
    if connection.host == "4124gs01":
        with ConnectWithPassphrase(connection.ssh_config["proxyjump"]) as cc:
            proxy_command = f"'ssh -A -J {cc.user}@{cc.host}'"
    else:
        proxy_command = "ssh"
    return proxy_command


def set_up_permissions(host):
    try:
        with ConnectWithPassphrase(host) as connection:
            path = f"{get_base_directory(connection)}/lizrd/grid/grid_entrypoint.sh"
            print(f"Changing permissions for {path}...")
            connection.run(f"chmod +x {path}")
            print("The permissions for the script have been changed successfully.")
    except Exception as e:
        raise Exception(
            f"The permissions change for the script failed. Error: {str(e)}"
        )


if __name__ == "__main__":
    # set up argparse for hostname, config
    parser = ArgumentParser()
    parser.add_argument("--host", type=str)
    # create parser
    args = parser.parse_args()

    cd_to_root_dir()
    working_dir = os.getcwd()
    base_dir = rsync_to_remote(args.host, working_dir + "/lizrd")
    _ = rsync_to_remote(args.host, working_dir + "/research")
    _ = rsync_to_remote(args.host, working_dir + "/configs")
    _ = rsync_to_remote(args.host, working_dir + "/.versioningignore")
    # WRITE root_dir to temp file for run_exp_remotely.sh
    with open("/tmp/base_dir.txt", "w") as f:
        f.write(base_dir)
    set_up_permissions(args.host)
    name_for_branch = (
        "exp_"
        + f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        + generate_random_string(10)
    )
    with open("/tmp/git_branch.txt", "w") as f:
        f.write(name_for_branch)
