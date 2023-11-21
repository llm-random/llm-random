import subprocess


def find_lines_with(name, data, start):
    lines = [i for i in range(len(data)) if data[i].startswith(start)]
    if len(lines) == 0:
        return None, None
    elif len(lines) > 1:
        raise ValueError(f"More than one line in {name} starts with {start} - {lines}")
    else:
        return lines[0], data[lines[0]][len(start) : -1]


def fix_one_iteration():
    cmd = ["bash", "lizrd/scripts/list_md5sum.sh"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    hashes = {}
    for line in proc.stdout.readlines():
        hash, filename = line.decode("utf-8").split()
        hashes[filename] = hash

    for filename in hashes.keys():
        with open(filename, "r") as file:
            data = file.readlines()

        h_line, h_val = find_lines_with(file, data, "md5_parent_hash: ")
        p_line, p_val = find_lines_with(file, data, "parent: ")
        if p_val is None or h_val is None:
            continue
        proper_hash = hashes.get(p_val)
        if proper_hash is None:
            print(
                f"Parent {p_val} of {filename} is not in the list of hashes, probably bad parent path in config"
            )
            print("Skipping..")
            continue
        if h_val != proper_hash:
            print(f"Fixing reference to parent {p_val} in {filename}")
            print(f"Proper hash is {proper_hash}, while {h_val} is in the file")
            decide = input("Fix (y/n)?")
            if decide.lower() == "y":
                data[h_line] = f"md5_parent_hash: {proper_hash}\n"
                with open(filename, "w") as file:
                    file.writelines(data)
                print("Fixed")
                return True
            else:
                print("Skipping all")
                return False
    return False


if __name__ == "__main__":
    while fix_one_iteration():
        pass
