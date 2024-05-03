from time import time
import dask.dataframe

# from datasets import load_from_disk
import dask

from random import randint

load_start = time()
dataset = dask.dataframe.read_parquet(
    "/net/tscratch/people/plgkciebiera/datasets/c4/train",
    parquet_file_extension=(".arrow"),
    engine="pyarrow",
)
load_time = time() - load_start
print(f"Load time: {load_time:.3f}s")
times = []
curr_time = time()
for i in range(100):
    # print(document[:100])
    document = dataset[randint(0, len(dataset) - 1)]
    document = document["text"]
    read_time = time() - curr_time
    # print(f"Time: {read_time}s")
    times.append(read_time)
    if i == 100:
        break
    curr_time = time()
print(f"Average time: {sum(times[1:]) / len(times[1:]):.5f}s")
