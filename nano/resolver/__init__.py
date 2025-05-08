import random
from omegaconf import OmegaConf
import platform
import os
import re


def get_cluster_name(hostname=None, username=None) -> str:
    if "LLMRANDOM_CLUSTER" in os.environ:
        return os.environ.get("LLMRANDOM_CLUSTER")

    if hostname is None:
        hostname = platform.uname().node

    conf = OmegaConf.load("configs/clusters.yaml")

    for cluster in conf:
        if "hosts" in cluster:
            for host_pattern in cluster.hosts:
                if re.match(host_pattern, hostname):
                    return cluster.name

    return "default"


OmegaConf.register_new_resolver("__llmrandom_cluster_config", get_cluster_name)

OmegaConf.register_new_resolver("random_seed", lambda: random.randint(0, 100000))

OmegaConf.register_new_resolver("eval", eval)
