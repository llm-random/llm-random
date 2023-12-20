from datasets import load_dataset
import os

os.environ["HF_DATASETS_CACHE"] = "<Twoja sciezka na dataset>"
print(f"Should save to {os.environ['HF_DATASETS_CACHE']}")
d = load_dataset("wikipedia", "20220301.simple")["train"]
print(f"Number of examples: {len(d)}")
print(f"First example: {d[0]}")
