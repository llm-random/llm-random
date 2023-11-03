import json


def record_gpu_ram_availability(filename: str, params: [str], value: bool):
    """
    This function is used to records whether a model with given params fits in gpu.
    """

    if filename is None or params is None:
        return

    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
        print(f"File not found or could not decode it, creating new dict.")

    current = data
    for param in params[:-1]:
        if param not in current:
            current[param] = {}
        current = current[param]

    last_param = params[-1]
    if last_param not in current:
        current[last_param] = {}
    current[last_param] = value

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
