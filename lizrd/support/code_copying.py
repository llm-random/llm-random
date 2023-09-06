import os
import shutil


def copy_code(newdir_name):
    cemetery_path = "llm_random_cemetery"
    # Copy code
    root_dir = os.getcwd()
    assert (
        os.path.basename(root_dir) == "llm-random"
    ), "You need to be in the llm-random directory to copy code."
    newdir_path = f"{os.path.dirname(root_dir)}/{cemetery_path}/{newdir_name}"

    ignore_patterns_file = os.path.join(root_dir, ".versioningignore")
    versioning_ignore_patterns = make_ignore_patterns(ignore_patterns_file)

    print(f"Copying code to {newdir_path}...")
    # Copy the project root directory to a new directory, ignoring files described in versioning_ignore_patterns
    shutil.copytree(root_dir, newdir_path, ignore=versioning_ignore_patterns)
    print(f"Code copied successfully to {newdir_path}")

    # Change to the new directory
    os.chdir(newdir_path)


def make_ignore_patterns(filepath):
    # Set up ignore patterns
    with open(filepath) as f:
        patterns = f.read().splitlines()
        patterns = [
            pattern for pattern in patterns if pattern != "" and pattern[0] != "#"
        ]
        patterns = [p.strip() for p in patterns]
        patterns = shutil.ignore_patterns(*patterns)
    return patterns
