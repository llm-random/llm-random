import os
from datasets import load_dataset
from argparse import ArgumentParser


# write argument parser for one argument: dataset directory
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory to save the dataset, e.g. /net/data/datasets/c4",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Cache directory for the dataset, e.g. /net/tscratch/tmp/kciebiera/c4_cache. This directory should be deleted after the dataset is downloaded.",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    cache_dir = args.cache_dir
    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "validation")
    print(f"Downloading C4 dataset train")
    ds_train = load_dataset("c4", "en", split="train", cache_dir=cache_dir)
    print(f"Number of train examples: {len(ds_train)}")
    print(f"Downloading C4 dataset validation")
    ds_val = load_dataset("c4", "en", split="validation", cache_dir=cache_dir)
    print(f"Number of validation examples: {len(ds_val)}")
    print(f"Saving dataset to {train_path}")
    ds_train.save_to_disk(train_path)
    print(f"Saving dataset to {val_path}")
    ds_val.save_to_disk(val_path)
    print(f"Dataset downloaded to {dataset_dir}")


if __name__ == "__main__":
    main()
