import random
from typing import Literal
import os
import subprocess

import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, IterableDataset

from lizrd.datasets import wikibookdata
from lizrd.datasets.processor import BERTSentenceProcessor, GPTSentenceProcessor
import lizrd.datasets.processor
from lizrd.datasets.processed_batch import (
    ProcessedBERTBatch,
    ProcessedBatch,
    ProcessedGPTBatch,
)
from lizrd.datasets.c4 import C4Dataset
import lizrd.datasets.processor
from lizrd.datasets.utils import get_random_chunk


class AbstractDataset:
    def get_example(self) -> str:
        raise


class WikiBookDataset(AbstractDataset):
    def __init__(self, seed, use_dummy_dataset=False):
        self.examples_buffer = []
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

        self.buffer_refill_to = 10000
        self.buffer_refill_from = 0
        self.min_sentence_length = 40
        self.bookcorpus_chance = 0.5
        self.bookcorpus_lines = len(self.dataset_book) // len(self.dataset_wiki) + 1
        self.bookcorpus_chance = self.bookcorpus_chance / 100 * self.bookcorpus_lines
        self.bookcorpus_lines = 100  # the above is very approximate
        self.wikipedia_chance = 1.0 - self.bookcorpus_chance
        print("bookcorpus_lines:", self.bookcorpus_lines)
        print("bookcorpus_chance:", self.bookcorpus_chance)

    def get_example(self) -> str:
        if len(self.examples_buffer) <= self.buffer_refill_from:
            self._refill_buffer()
        example = self.examples_buffer.pop()
        return example

    def _refill_buffer(self):
        while len(self.examples_buffer) <= self.buffer_refill_to:
            self._add_examples(self._get_random_document())
        self.rng.shuffle(self.examples_buffer)

    def _get_random_document(self):
        if self.rng.random() < self.wikipedia_chance:
            document_text = self.dataset_wiki[
                self.rng.randint(0, len(self.dataset_wiki) - 1)
            ]["text"]
            documents_sentences = wikibookdata.process_wiki_text(document_text)
            assert isinstance(documents_sentences, list)
            assert isinstance(documents_sentences[0], str)
        else:
            linebegin = self.rng.randint(
                0, len(self.dataset_wiki) - 1 - self.bookcorpus_lines
            )
            lineend = linebegin + self.bookcorpus_lines
            documents_sentences = self.dataset_book[linebegin:lineend]["text"]
            documents_sentences = wikibookdata.process_book_text(documents_sentences)
            assert isinstance(documents_sentences, list)
            assert isinstance(documents_sentences[0], str)
        return documents_sentences

    def _add_examples(self, param):
        """This version simply filters out all sentences that are too short, then adds all remaining sentences to the buffer."""

        document_sentences = [
            sentence for sentence in param if len(sentence) > self.min_sentence_length
        ]
        self.examples_buffer += document_sentences


class C4(AbstractDataset):
    def __init__(self, seed, split: str = "train"):
        self.dataset = load_dataset("c4", "en", split=split)

        # print(f"Using C4 dataset consisting of {NUM_C4_TOKENS} tokens")
        # print(
        #     f"One epoch with batch {batch_size} and sequence length {seq_length} will take {NUM_C4_TOKENS // (batch_size * seq_length)} steps"
        # )

        self.rng = random.Random(seed)

    def get_example(self) -> str:
        return self.dataset["train"][self.rng.randint(0, len(self.dataset) - 1)]
