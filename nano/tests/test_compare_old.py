from dataclasses import dataclass
import itertools
import unittest
from unittest.mock import patch
import torch

from model import get_dataloader

from old_datasets import get_processed_dataset


class TestComparison(unittest.TestCase):
    """
    This test case compares the old and new dataloaders implementation
    """

    def patch_randint_in_get_document(self, dataset):
        """
        Patch the randint function in the get_document method of the dataset so order of documents is fixed, but sampled doucment from cyclic buffer is from seed
        """
        original_get_document = dataset.get_document

        def randint_generator():
            for i in range(1000):
                yield i

        randint_gen = randint_generator()

        def patched_get_document():
            with patch.object(
                dataset.py_rng, "randint", side_effect=lambda a, b: next(randint_gen)
            ):
                return original_get_document()

        dataset.get_document = patched_get_document

    def test_compare(self):

        with patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0"}):
            @dataclass
            class Config:
                num_workers: int
                dataset: str
                total_batch_size: int
                training_dataset_path: str
                eval_dataset_path: str
                world_size_independent: bool
                use_new_sampling_method: bool
                shuffle: bool

            config = Config(
                num_workers=0,
                dataset="c4",
                total_batch_size=10,
                training_dataset_path="data",
                eval_dataset_path="data",
                world_size_independent=False,
                use_new_sampling_method=False,
                shuffle=False,
            )
            train_dataloader = get_dataloader(
                dataloader_config=config,
                batch_size_per_device=10,
                sequence_length=32,
                seed=2311,
                dataset_split="train",
            )

            old_train_dataset = get_processed_dataset(
                batch_size=10,
                sequence_length=33,
                device="cpu",
                num_workers=0,
                seed=2311,
                model_type="gpt",
                dataset_type="c4",
                use_dummy_dataset=True,
                dataset_split="train",
                dataset_path="data",
            )
            self.patch_randint_in_get_document(
                old_train_dataset.generator.dataset.dataset
            )

        for i, (a, b) in enumerate(
            itertools.islice(zip(old_train_dataset, train_dataloader), 10), start=1
        ):
            assert torch.equal(a.input_ids, b), f"Samples number:{i} are not equal "

if __name__ == "__main__":
    unittest.main()
