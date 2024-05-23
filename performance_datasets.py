from time import time

from datasets import load_dataset

from random import randint

load_start = time()
dataset = load_dataset(
    "parquet",
    data_files="/net/tscratch/people/plgkciebiera/datasets2/c4/validation/validation.parquet",
    split="validation",
)
load_time = time() - load_start
print(f"Load time: {load_time:.3f}s")
times = []
curr_time = time()
for i in range(100):
    document = dataset[randint(0, len(dataset) - 1)]
    document = document["text"]
    read_time = time() - curr_time
    times.append(read_time)
    if i == 100:
        break
    curr_time = time()
print(f"Average time: {sum(times[1:]) / len(times[1:]):.5f}s")
