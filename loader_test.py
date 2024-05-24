from research.datasets import get_processed_dataset

loader = get_processed_dataset(
    batch_size=512,
    sequence_length=16,
    device="cpu",
    num_workers=8,
    seed=42,
    model_type="gpt",
    dataset_type="c4",
    use_dummy_dataset=False,
    dataset_path="/net/tscratch/people/plgkciebiera/datasets/c4/train",
    dataset_split="train",
    num_gpus=1,
)

print("GETTING BATCH")
batch = loader.get_batch()
print("GOT BATCH")
print(batch)
