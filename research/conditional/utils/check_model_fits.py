import json


def mark_model_fits(filename, params, value):
    """
    This function is used to mark whether a model fits a certain condition in a json file.
    """

    if filename is None or params is None:
        return

    try:
        with open(filename, "r") as file:
            data = json.load(file)
            print(f"Found file with data: {data}")
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
        print(f"Did not find file with data")

    current = data
    for param_name, param_value in params[:-1]:
        if param_name not in current:
            current[param_name] = {}
        current = current[param_name]
        if param_value not in current:
            current[param_value] = {}
        current = current[param_value]
    last_param_name, last_param_value = params[-1]

    if last_param_name not in current:
        current[last_param_name] = {}
    current = current[last_param_name]

    current[last_param_value] = value

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
