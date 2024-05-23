from time import time

# from datasets import load_dataset
import dask.dataframe as dd
from random import randint

load_start = time()
dataset = dd.read_parquet(
    "/net/tscratch/people/plgkciebiera/datasets2/c4/validation/validation.parquet"
)
load_time = time() - load_start
print(f"Load time: {load_time:.3f}s")
times = []
curr_time = time()
for i in range(100):
    document = dataset["text"][randint(0, len(dataset) - 1)]
    read_time = time() - curr_time
    times.append(read_time)
    if i == 100:
        break
    curr_time = time()
print(f"Average time: {sum(times[1:]) / len(times[1:]):.5f}s")
