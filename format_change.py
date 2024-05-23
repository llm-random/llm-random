from datasets import load_from_disk


dataset = load_from_disk(
    "/net/tscratch/people/plgkciebiera/datasets/c4/validation",
)
dataset.to_parquet(
    "/net/tscratch/people/plgkciebiera/datasets2/c4/validation/validation.parquet"
)
