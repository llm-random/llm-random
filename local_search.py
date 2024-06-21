import yaml
import os
import copy
import random
import string
import numpy as np
from tqdm import trange
import time
from pathlib import Path
import argparse
import time
import subprocess


def parse_config(config_path):
    with open(config_path, "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    tune_config = base_config["tune_config"]
    del base_config["tune_config"]
    tune_config["base_config"] = base_config
    return tune_config


def get_random_UUID(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def run_command(command):
    return os.popen(command).read()


def init_neptune_connection():
    return None  # TODO
    #raise Exception("Not implemented")


class TrainRun:
    def __init__(self, search, command, pid, val):
        self.command = command
        self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.pid = pid
        self.connection = search.connection
        self.server = search.server_name
        self.val = val

    def is_submitted(self):
        return self.process.poll() is not None

    def is_queued(self):
        out = run_command(f"ssh -qt {self.server} 'squeue --start --states=R -o \"%.50j\" | grep {self.pid} ; squeue --states=R -o \"%.50j\"' | grep {self.pid} 2>/dev/null")
        return len(out) > 1

    def is_running(self):
        out = run_command(f"ssh -qt {self.server} 'squeue --states=R -o \"%.50j\"' | grep {self.pid} 2>/dev/null")
        return len(out) > 1

    def is_in_neptune(self):
        raise Exception("Not implemented")

    def is_finished(self):
        raise Exception("Not implemented")

    def get_results(self):
        raise Exception("Not implemented")


class LocalSearch:
    def __init__(self, server_name, base_config, iters, tune_params, wait_time=5, param_change=(0.1, 0.5, 3, 10), configs_directory="local_search_configs"):
        self.iters = iters
        self.server_name = server_name
        self.current_config = base_config
        self.param_change = param_change
        self.tune_params = tune_params
        self.connection = init_neptune_connection()
        self.configs_directory = configs_directory
        self.last_score = None
        self.wait_time = wait_time
        self.last_param = ""
        self.exp_name = self.get_param_val('name')
        Path(self.configs_directory).mkdir(parents=True, exist_ok=True)
        assert not self.current_config["interactive_debug_session"]

    def write_yaml(self, config_path, config):
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def get_new_config(self):
        return copy.deepcopy(self.current_config)

    def get_param_val(self, param, default=None):
        if param not in self.current_config["params"]:
            if default is None:
                raise Exception(f"Parameter {param} not found in config")
            else:
                return default
        return self.current_config["params"][param]

    def set_param_val(self, config, param, val):
        config["params"][param] = val
        return config

    def get_run_command(self, config_path):
        return f"bash scripts/run_exp_remotely.sh {self.server_name} {config_path}"

    def run_config_dict(self, config, param, val, iter):
        # set name
        uuid = get_random_UUID()
        name = f"{self.exp_name}_{iter}_{param}_{uuid}"
        config_path = f"{self.configs_directory}/{name}.yaml"
        self.set_param_val(config, 'name', name)

        # add tags
        tags = self.get_param_val('tags', [])
        tags += ["local_search", f"search_{self.exp_name}", f"tuning_{param}", f"iter_{iter}"]
        self.set_param_val(config, 'tags', tags)

        self.write_yaml(config_path, config)
        command = self.get_run_command(config_path)
        return TrainRun(self, command, name, val)

    def wait_until_true(self, trange, func, pids, desc):
        all = len(pids)
        while len(pids) > 0:
            for pid in pids:
                if func(pid):
                    pids.remove(pid)
                    trange.set_description(desc + f" ({all-len(pids)}/{all})")
                    trange.update(1)
            time.sleep(self.wait_time)

    def wait_for_runs_to_finish_and_get_best(self, pids, param):
        range = trange(5*len(pids), desc="Starting..", leave=True)
        self.wait_until_true(range, TrainRun.is_submitted, list(pids), "Submitting exps..")
        self.wait_until_true(range, TrainRun.is_queued, list(pids), "Exps submitted, waiting for them to queue..")
        self.wait_until_true(range, TrainRun.is_running, list(pids), "Exps queued, waiting for them to start")
        self.wait_until_true(range, TrainRun.is_in_neptune, list(pids), "Exps started, waiting for them to appear in Neptune")
        self.wait_until_true(range, TrainRun.is_finished, list(pids), "Exps running, waiting for results")

        results = [run.get_results() for run in pids]

        results_str = ", ".join([f"{run.val}: {res}" for run, res in zip(pids, results)])
        trange.set_description(f"Results ({param}): {results_str} ")
        range.close()
        best_run = np.argmin(results)
        if self.last_score is None or results[best_run] < self.last_score:
            self.last_score = results[best_run]
            self.set_param_val(self.current_config, param, pids[best_run].val)
            print(f"New best value [{param}] = {results[best_run]} with loss {results[best_run]}")
            return True
        return False

    def run_param_tuning(self, param, i):
        val = self.get_param_val(param)
        new_vals = [val*change for change in self.param_change]
        test_configs = [(self.set_param_val(self.get_new_config(), param, new_val), new_val) for new_val in new_vals]
        pids = [self.run_config_dict(config, param, new_val, i) for config, new_val in test_configs]
        return pids

    def run_baseline_score(self, param):
        val = self.get_param_val(param)
        return self.run_config_dict(self.current_config, param, val, iter=0)

    def run_iteration(self, i):
        while (perm := np.random.permutation(self.tune_params))[0] == self.last_param: pass
        self.last_param = perm[-1]
        changed = False
        for param in perm:
            pids = self.run_param_tuning(param, i)
            if self.last_score is None:
                pids.append(self.run_baseline_score(param))
            has_changed = self.wait_for_runs_to_finish_and_get_best(pids, param)
            changed = changed or has_changed
        return changed

    def run(self):
        for i in range(self.iters):
            if not self.run_iteration(i):
                print(f"Early stopping at iteration {i}")
                break

        name = f"{self.exp_name}_best_config"
        config_path = f"{self.configs_directory}/{name}.yaml"
        self.set_param_val(self.current_config, 'name', name)
        self.write_yaml(config_path, self.current_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    local_search = LocalSearch(server_name=args.server_name, **parse_config(args.config_path))
    local_search.run()



