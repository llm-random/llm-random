from abc import abstractmethod
import random
from typing import Optional

from datasets import load_dataset


class AbstractDataset:
    @abstractmethod
    def get_document(self) -> str:
        raise NotImplementedError()


class WikiBookDataset(AbstractDataset):
    def __init__(
        self,
        seed: Optional[int] = None,
        use_dummy_dataset: bool = False,
        split: str = "train",
    ):
        assert split in ["train", "eval"]
        self.split = split

        self.dataset_wiki = load_dataset(
            "wikipedia", f"20220301.{'simple' if use_dummy_dataset else 'en'}"
        )["train"]
        self.dataset_book = (
            load_dataset("bookcorpus")["train"]
            if not use_dummy_dataset
            else self.dataset_wiki
        )
        self.seed = seed
        self.rng = random.Random(seed)

        self.bookcorpus_chance = len(self.dataset_book) / len(self.dataset_wiki)

    def get_document(self) -> str:
        document = None
        while document is None or not self._belongs_to_split(document):
            document = self._get_document()
        return document

    def _belongs_to_split(self, document: str) -> bool:
        eval_percentage = 5

        if self.split == "train":
            return hash(document) % 100 >= eval_percentage
        elif self.split == "eval":
            return hash(document) % 100 < eval_percentage
        else:
            raise ValueError("split must be either 'train' or 'eval'")

    def _get_document(self) -> str:
        selector = self.rng.random()
        if selector < self.bookcorpus_chance:
            return self._get_random_book_example()
        else:
            return self._get_random_wiki_example()

    def _get_random_book_example(self) -> str:
        document = self.dataset_book[self.rng.randint(0, len(self.dataset_book) - 1)]
        return document["text"]

    def _get_random_wiki_example(self) -> str:
        document = self.dataset_wiki[self.rng.randint(0, len(self.dataset_wiki) - 1)]
        return document["text"]


class C4(AbstractDataset):
    def __init__(self, seed: Optional[int] = None, split: str = "train"):
        self.dataset = load_dataset("c4", "en", split=split)
        self.rng = random.Random(seed)

    def get_document(self) -> str:
        return self.dataset["train"][self.rng.randint(0, len(self.dataset) - 1)]
