import time

import numpy as np
import torch
from torch import nn as nn

GLOBAL_TIMERS = dict()
GLOBAL_NAMES = []
GLOBAL_DEPTHS = dict()
CURRENT_DEPTH = [0]
DISABLED = False


class TimerLayer(nn.Module):
    def __init__(self, name, layer, off=False):
        super(TimerLayer, self).__init__()
        self.name = name
        self.layer = layer
        self.off = off

    def forward(self, *args, **kwargs):
        if self.off:
            result = self.layer(*args, **kwargs)
        else:
            with Timer(self.name):
                result = self.layer(*args, **kwargs)
        return result


class Timer(object):
    def __init__(self, name, disable_inner=False):
        global DISABLED
        self.name = name
        self.disable_inner = disable_inner
        self.i_disabled = False
        if not DISABLED:
            if name not in GLOBAL_TIMERS:
                GLOBAL_TIMERS[self.name] = []
                # GLOBAL_TIMERS[self.name+'T'] = []
            if self.name not in GLOBAL_NAMES:
                GLOBAL_NAMES.append(name)
        # self.start = torch.cuda.Event(enable_timing=True)
        # self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        global DISABLED
        if not DISABLED:
            if self.disable_inner:
                DISABLED = True
                self.i_disabled = True
            torch.cuda.synchronize()
            GLOBAL_DEPTHS[self.name] = CURRENT_DEPTH[0]
            CURRENT_DEPTH[0] += 1
            self.start_time = time.time()
            # self.start.record()

    def __exit__(self, *args):
        global DISABLED
        if not DISABLED or self.i_disabled:
            if self.disable_inner:
                DISABLED = False
                self.i_disabled = False
            # self.end.record()
            torch.cuda.synchronize()
            self.end_time = time.time()

            CURRENT_DEPTH[0] -= 1
            GLOBAL_TIMERS[self.name].append(self.end_time - self.start_time)
            # GLOBAL_TIMERS[self.name+'T'].append(self.start.elapsed_time(self.end)/1000)


def reset_times():
    global GLOBAL_TIMERS, GLOBAL_DEPTHS, GLOBAL_NAMES
    GLOBAL_TIMERS = dict()
    GLOBAL_DEPTHS = dict()
    GLOBAL_NAMES = []


def print_times(reset=True):
    for name in GLOBAL_NAMES:
        values = GLOBAL_TIMERS[name]
        depth = GLOBAL_DEPTHS[name]
        print(f'{" "*depth + name:18}: {round(sum(values), 3)} +/- {round(np.std(values)*len(values)**0.5, 2)}')
    print('\n\n\n')
    if reset:
        reset_times()