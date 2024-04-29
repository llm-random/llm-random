from time import time
from datasets import load_from_disk

from random import randint

load_start = time()
dataset = load_from_disk("/net/tscratch/people/plgkciebiera/datasets/c4/train")
load_time = time() - load_start
print(f"Load time: {load_time:.3f}s")
times = []
curr_time = time()
for i in range(20):
    document = dataset[randint(0, len(dataset))]["text"]
    # print(document[:100])
    document = document["text"]
    read_time = time() - curr_time
    print(f"Time: {read_time}s")
    times.append(read_time)
    # if i == 20:
    #    break
    curr_time = time()
print(f"Average time: {sum(times) / len(times):.3f}s")
