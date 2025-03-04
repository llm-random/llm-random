#!/usr/bin/env python
from datasets import load_dataset
from datasets import Dataset

N_EXAMPLES = 300
TRAIN_DATASET_NAME = "data"
EVAL_DATASET_NAME = "data_eval"

hf_dataset = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

hf_dataset_eval = load_dataset(
    "allenai/c4",
    "en",
    split="validation",
    streaming=True,
    trust_remote_code=True,
)


class SamplesGenerator:
    def __init__(self, dataset, n_examples):
        self.dataset = dataset
        self.n_examples = n_examples

    def __call__(self):
        for i, ex in enumerate(self.dataset):
            if i >= self.n_examples:
                break
            yield ex


new_train_dataset = Dataset.from_generator(SamplesGenerator(hf_dataset, N_EXAMPLES))
new_train_dataset.save_to_disk(TRAIN_DATASET_NAME)

new_eval_dataset = Dataset.from_generator(SamplesGenerator(hf_dataset_eval, N_EXAMPLES))
new_eval_dataset.save_to_disk(EVAL_DATASET_NAME)
