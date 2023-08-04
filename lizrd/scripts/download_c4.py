from datasets import load_dataset

d = load_dataset("c4", "en", split="train")
print(f"Number of examples: {len(d)}")
print(f"First example: {d[0]}")
